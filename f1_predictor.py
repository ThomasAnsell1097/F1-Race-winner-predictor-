"""
F1 race winner predictor.

Learns from past races: grid position, each driver's and team's recent form,
how a driver goes at this track, and how they do in the wet and in the heat
compared with their usual. Then predicts the next race, mixing a wet and a
dry prediction by the chance of rain in the forecast.

Data (free, no API key):
  - Race results:  Jolpica F1 API (the Ergast replacement)  https://api.jolpi.ca
  - Weather:       Open-Meteo historical + forecast         https://open-meteo.com
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

F1_API = "https://api.jolpi.ca/ergast/f1"
WEATHER_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST = "https://api.open-meteo.com/v1/forecast"
CACHE = Path(__file__).parent / "cache"

WET_MM = 0.5     # rain in the race window (mm) that counts as a wet race
HOT_C = 28.0     # air temperature that counts as a hot race
DNF_POS = 20     # finishing position given to a driver who didn't finish
RACE_HOURS = 2   # length of the race window used for weather

FEATURES = [
    "grid", "form_finish", "form_points", "team_form", "track_finish",
    "season_points", "season_wins", "dnf_rate",
    "is_wet", "wet_skill", "wet_x_skill",
    "temp", "hot_skill", "hot_x_skill",
]


# ---------------------------------------------------------------- downloading

class QuotaUsedUp(Exception):
    """The API's hourly or daily request limit has been reached."""


def _get_json(url: str, params: dict, cache_name: str | None,
              cache_only: bool = False) -> dict:
    """GET JSON, caching it on disk so reruns are quick and polite to the APIs."""
    path = CACHE / cache_name if cache_name else None
    if path and path.exists():
        return json.loads(path.read_text())
    if cache_only:
        raise QuotaUsedUp("not downloaded yet")
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            reason = r.text[:200]
            if "Daily" in reason or "Hourly" in reason:
                raise QuotaUsedUp(reason)   # waiting a few seconds won't help
            time.sleep(2 ** attempt)        # short-term limit: back off and retry
            continue
        r.raise_for_status()
        data = r.json()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"Gave up on {url} after repeated rate limits")


def _f1_paged(endpoint: str, cache_prefix: str | None) -> list[dict]:
    """Fetch every page of a Jolpica endpoint and return the list of races."""
    races, offset, limit = {}, 0, 100
    while True:
        name = f"{cache_prefix}_{offset}.json" if cache_prefix else None
        data = _get_json(f"{F1_API}/{endpoint}.json",
                         {"limit": limit, "offset": offset}, name)
        mr = data["MRData"]
        for race in mr["RaceTable"]["Races"]:
            # one race can be split across two pages, so merge by round
            key = race["round"]
            if key in races:
                races[key]["Results"] += race.get("Results", [])
            else:
                races[key] = race
        offset += limit
        if offset >= int(mr["total"]):
            break
        time.sleep(0.3)
    return list(races.values())


def load_results(seasons: list[int]) -> pd.DataFrame:
    """One row per driver per race, for the given seasons."""
    this_year = dt.date.today().year
    rows = []
    for season in seasons:
        print(f"  results {season}...", file=sys.stderr)
        # don't cache the season that's still going on
        prefix = None if season >= this_year else f"results_{season}"
        for race in _f1_paged(f"{season}/results", prefix):
            loc = race["Circuit"]["Location"]
            for res in race["Results"]:
                pos_text = res["positionText"]
                finished = pos_text.isdigit()
                rows.append({
                    "season": season,
                    "round": int(race["round"]),
                    "race": race["raceName"],
                    "circuit": race["Circuit"]["circuitId"],
                    "date": race["date"],
                    "time": race.get("time", "13:00:00Z"),
                    "lat": float(loc["lat"]),
                    "lon": float(loc["long"]),
                    "driver": res["Driver"]["driverId"],
                    "name": f'{res["Driver"]["givenName"]} {res["Driver"]["familyName"]}',
                    "team": res["Constructor"]["constructorId"],
                    "grid": int(res["grid"]) or 21,   # 0 means pit lane start
                    "finish": int(res["position"]) if finished else DNF_POS,
                    "finished": finished,
                    "points": float(res["points"]),
                })
    df = pd.DataFrame(rows)
    df["win"] = (df["finish"] == 1) & df["finished"]
    return df.sort_values(["season", "round", "finish"]).reset_index(drop=True)


def race_start(date: str, time_utc: str) -> dt.datetime:
    return dt.datetime.fromisoformat(f"{date}T{time_utc.rstrip('Z')}")


def _hourly_window(hourly: dict, start: dt.datetime) -> pd.DataFrame:
    h = pd.DataFrame(hourly)
    h["time"] = pd.to_datetime(h["time"])
    s = pd.Timestamp(start).floor("h")
    return h[(h["time"] >= s) & (h["time"] < s + pd.Timedelta(hours=RACE_HOURS))]


def race_weather(lat: float, lon: float, date: str, time_utc: str,
                 cache_only: bool = False) -> dict:
    """Actual rain (mm) and temperature during a past race."""
    data = _get_json(WEATHER_ARCHIVE, {
        "latitude": lat, "longitude": lon, "start_date": date, "end_date": date,
        "hourly": "precipitation,temperature_2m", "timezone": "GMT",
    }, f"weather_{lat}_{lon}_{date}.json", cache_only)
    w = _hourly_window(data["hourly"], race_start(date, time_utc))
    return {"rain_mm": float(w["precipitation"].fillna(0).sum()),
            "temp": float(w["temperature_2m"].mean())}


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    races = df[["season", "round", "lat", "lon", "date", "time"]].drop_duplicates(["season", "round"])
    print(f"  weather for {len(races)} races...", file=sys.stderr)
    out, quota_hit, missing = [], False, 0
    for r in races.itertuples():
        w = None
        try:   # once the limit is hit, still use weather that's already saved
            w = race_weather(r.lat, r.lon, r.date, r.time, cache_only=quota_hit)
        except QuotaUsedUp:
            quota_hit = True
        except Exception as e:
            print(f"    no weather for {r.season} R{r.round}: {e}", file=sys.stderr)
        if w is None:                   # weather missing: assume dry, mild
            missing += 1
            w = {"rain_mm": 0.0, "temp": 22.0}
        out.append({"season": r.season, "round": r.round, **w})
    if missing:
        print(f"\n  WARNING: no weather for {missing} of {len(races)} races, so they count as dry."
              + ("\n  Open-Meteo's free request limit was reached. Run again later (or tomorrow);"
                 "\n  weather already downloaded is saved, so it carries on where it stopped."
                 if quota_hit else ""), file=sys.stderr)
    df = df.merge(pd.DataFrame(out), on=["season", "round"])
    df["temp"] = df["temp"].fillna(22.0)
    df["is_wet"] = (df["rain_mm"] >= WET_MM).astype(float)
    return df


# ------------------------------------------------------------------ features

def _mean_last(values: list, n: int, default: float) -> float:
    return float(np.mean(values[-n:])) if values else default


def _skill(finish_in: list, finish_out: list, shrink: float = 3.0) -> float:
    """How many places better (+) a driver finishes in some condition (wet, hot)
    than they do otherwise. Shrunk towards 0 when there are few such races."""
    if not finish_in or not finish_out:
        return 0.0
    n = len(finish_in)
    return float((np.mean(finish_out) - np.mean(finish_in)) * n / (n + shrink))


class History:
    """Everything known about drivers and teams up to a point in time.
    Races are fed in order, so features never use the future."""

    def __init__(self):
        self.finish, self.points, self.dnf = {}, {}, {}
        self.team_finish = {}
        self.track = {}                        # (driver, circuit) -> finishes
        self.cond = {}                         # driver -> [(finish, wet, hot)]
        self.season_pts, self.season_wins = {}, {}

    def features(self, driver, team, circuit, season, grid, is_wet, temp) -> dict:
        fin = self.finish.get(driver, [])
        form = _mean_last(fin, 5, 12.0)
        g = self.cond.get(driver, [])
        wet_in = [x for x, w, _ in g if w]
        wet_out = [x for x, w, _ in g if not w]
        hot_in = [x for x, _, h in g if h]
        hot_out = [x for x, _, h in g if not h]
        wet_skill = _skill(wet_in, wet_out)
        hot_skill = _skill(hot_in, hot_out)
        is_hot = float(temp >= HOT_C)
        return {
            "grid": grid,
            "form_finish": form,
            "form_points": _mean_last(self.points.get(driver, []), 5, 0.0),
            "team_form": _mean_last(self.team_finish.get(team, []), 10, 12.0),
            "track_finish": _mean_last(self.track.get((driver, circuit), []), 5, form),
            "season_points": self.season_pts.get((driver, season), 0.0),
            "season_wins": self.season_wins.get((driver, season), 0),
            "dnf_rate": _mean_last(self.dnf.get(driver, []), 10, 0.15),
            "is_wet": is_wet,
            "wet_skill": wet_skill,
            "wet_x_skill": is_wet * wet_skill,
            "temp": temp,
            "hot_skill": hot_skill,
            "hot_x_skill": is_hot * hot_skill,
        }

    def update(self, race: pd.DataFrame):
        for r in race.itertuples():
            self.finish.setdefault(r.driver, []).append(r.finish)
            self.points.setdefault(r.driver, []).append(r.points)
            self.dnf.setdefault(r.driver, []).append(0.0 if r.finished else 1.0)
            self.team_finish.setdefault(r.team, []).append(r.finish)
            self.track.setdefault((r.driver, r.circuit), []).append(r.finish)
            if r.finished:   # a breakdown says nothing about liking the rain
                self.cond.setdefault(r.driver, []).append(
                    (r.finish, bool(r.is_wet), r.temp >= HOT_C))
            k = (r.driver, r.season)
            self.season_pts[k] = self.season_pts.get(k, 0.0) + r.points
            self.season_wins[k] = self.season_wins.get(k, 0) + int(r.win)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, History]:
    hist, rows = History(), []
    for (_, _), race in df.groupby(["season", "round"], sort=True):
        for r in race.itertuples():
            f = hist.features(r.driver, r.team, r.circuit, r.season,
                              r.grid, r.is_wet, r.temp)
            rows.append({"season": r.season, "round": r.round, "race": r.race,
                         "driver": r.driver, "name": r.name, "win": int(r.win), **f})
        hist.update(race)
    return pd.DataFrame(rows), hist


# --------------------------------------------------------------------- model

def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        # small and cautious: tested on 2021-2025, bigger models overfit
        max_iter=150, learning_rate=0.03, max_depth=3,
        min_samples_leaf=80, l2_regularization=5.0, random_state=0)


def race_probs(model, X: pd.DataFrame, groups: pd.Series) -> np.ndarray:
    """Win probability per driver, scaled so each race sums to 100%."""
    p = model.predict_proba(X[FEATURES])[:, 1]
    return p / pd.Series(p, index=X.index).groupby(groups.values).transform("sum").values


def backtest(feats: pd.DataFrame, test_season: int) -> None:
    train = feats[feats["season"] < test_season]
    test = feats[feats["season"] == test_season].copy()
    if train.empty or test.empty:
        print("Not enough seasons to backtest.")
        return
    model = make_model().fit(train[FEATURES], train["win"])
    test["p"] = race_probs(model, test, test["round"])
    hits = hits_pole = top3 = 0
    n = 0
    for _, race in test.groupby("round"):
        if race["win"].sum() == 0:
            continue
        n += 1
        winner = race.loc[race["win"] == 1, "driver"].iloc[0]
        ranked = race.sort_values("p", ascending=False)["driver"].tolist()
        hits += ranked[0] == winner
        top3 += winner in ranked[:3]
        hits_pole += race.loc[race["grid"].idxmin(), "driver"] == winner
    print(f"\nBacktest on {test_season} ({n} races), trained on earlier seasons:")
    print(f"  model picked the winner:     {hits}/{n}  ({hits / n:.0%})")
    print(f"  winner in model's top 3:     {top3}/{n}  ({top3 / n:.0%})")
    print(f"  'pole sitter wins' baseline: {hits_pole}/{n}  ({hits_pole / n:.0%})")


# ---------------------------------------------------------------- next race

def next_race(season: int) -> dict | None:
    data = _get_json(f"{F1_API}/{season}.json", {"limit": 100}, None)
    today = dt.date.today().isoformat()
    for race in data["MRData"]["RaceTable"]["Races"]:
        if race["date"] >= today:
            return race
    return None


def forecast_rain(lat: float, lon: float, date: str, time_utc: str) -> tuple[float, float] | None:
    """(chance of rain 0-1, temperature) for the race window, if within forecast range."""
    try:
        data = _get_json(WEATHER_FORECAST, {
            "latitude": lat, "longitude": lon, "start_date": date, "end_date": date,
            "hourly": "precipitation_probability,temperature_2m", "timezone": "GMT",
        }, None)
    except (requests.HTTPError, QuotaUsedUp):
        return None   # race too far ahead for a forecast, or the daily limit is used up
    w = _hourly_window(data["hourly"], race_start(date, time_utc))
    if w.empty or w["precipitation_probability"].isna().all():
        return None
    return float(w["precipitation_probability"].max()) / 100, float(w["temperature_2m"].mean())


def qualifying_grid(season: int, rnd: str) -> dict[str, int]:
    data = _get_json(f"{F1_API}/{season}/{rnd}/qualifying.json", {"limit": 100}, None)
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return {}
    return {q["Driver"]["driverId"]: int(q["position"]) for q in races[0]["QualifyingResults"]}


def predict_next(df: pd.DataFrame, feats: pd.DataFrame, hist: History,
                 rain: float | None, grid_arg: str | None, top: int) -> None:
    season = int(df["season"].max())
    race = next_race(season) or next_race(season + 1)
    if race is None:
        print("Couldn't find an upcoming race.")
        return
    loc = race["Circuit"]["Location"]
    lat, lon = float(loc["lat"]), float(loc["long"])
    date, time_utc = race["date"], race.get("time", "13:00:00Z")
    circuit = race["Circuit"]["circuitId"]

    print(f"\n{race['raceName']} {race['season']}  -  {race['Circuit']['circuitName']}, {date}")

    # weather: the user's number wins, else the forecast, else assume dry
    fc = forecast_rain(lat, lon, date, time_utc)
    temp = fc[1] if fc else 22.0
    if rain is None:
        rain = fc[0] if fc else 0.0
        print(f"  chance of rain: {rain:.0%} (" +
              ("forecast" if fc else "no forecast yet, assuming dry; set --rain") + ")")
    else:
        print(f"  chance of rain: {rain:.0%} (you set it)")
    print(f"  temperature:    {temp:.0f}°C" + ("" if fc else " (guess)"))

    # grid: the user's order, else real qualifying, else expected from recent form
    last = df[(df["season"] == season) & (df["round"] == df.loc[df["season"] == season, "round"].max())]
    entrants = last[["driver", "name", "team"]].drop_duplicates("driver")
    if grid_arg:
        order = [d.strip().lower() for d in grid_arg.split(",")]
        grid = {d: i + 1 for i, d in enumerate(order)}
        print("  grid: the order you gave")
    else:
        grid = qualifying_grid(int(race["season"]), race["round"])
        print("  grid: " + ("real qualifying result" if grid else
                            "not qualified yet, using each driver's average recent grid"))
    if not grid:
        recent = df.groupby("driver")["grid"].apply(lambda s: s.tail(5).mean())
        order = recent.reindex(entrants["driver"]).fillna(15).sort_values().index
        grid = {d: i + 1 for i, d in enumerate(order)}

    model = make_model().fit(feats[FEATURES], feats["win"])
    rows = []
    for e in entrants.itertuples():
        g = grid.get(e.driver, 21)
        for wet in (0.0, 1.0):
            rows.append({"driver": e.driver, "name": e.name, "wet": wet,
                         **hist.features(e.driver, e.team, circuit, int(race["season"]),
                                         g, wet, temp)})
    X = pd.DataFrame(rows)
    X["p"] = race_probs(model, X, X["wet"])
    dry = X[X["wet"] == 0].set_index("driver")
    wet = X[X["wet"] == 1].set_index("driver")
    out = pd.DataFrame({
        "name": dry["name"], "grid": dry["grid"].astype(int),
        "dry": dry["p"], "wet": wet["p"],
        "win": (1 - rain) * dry["p"] + rain * wet["p"],
        "wet_skill": dry["wet_skill"],
    }).sort_values("win", ascending=False).head(top)

    print(f"\n  {'':2} {'Driver':22} {'Grid':>4} {'If dry':>7} {'If wet':>7} {'WIN %':>7}  rain skill")
    for i, r in enumerate(out.itertuples(), 1):
        skill = f"{r.wet_skill:+.1f}" if abs(r.wet_skill) >= 0.3 else "  ·"
        print(f"  {i:>2} {r.name:22} {r.grid:>4} {r.dry:>7.1%} {r.wet:>7.1%} {r.win:>7.1%}  {skill}")
    print("\n  rain skill = how many places better (+) or worse (-) a driver finishes in the wet than the dry")


# ----------------------------------------------------------------------- CLI

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Predict the next F1 race winner.")
    ap.add_argument("--from", dest="start", type=int, default=2014,
                    help="first season to learn from (default 2014)")
    ap.add_argument("--rain", type=float, default=None,
                    help="chance of rain in %%, e.g. --rain 60 (default: use the forecast)")
    ap.add_argument("--grid", default=None,
                    help="starting grid as driver ids, e.g. norris,verstappen,leclerc")
    ap.add_argument("--backtest", type=int, default=None,
                    help="test the model on a past season, e.g. --backtest 2024")
    ap.add_argument("--top", type=int, default=10, help="how many drivers to show")
    a = ap.parse_args(argv)

    end = dt.date.today().year
    print("Loading data (first run takes a few minutes, then it's cached)...", file=sys.stderr)
    df = add_weather(load_results(list(range(a.start, end + 1))))
    df = df[df["season"] >= a.start]
    feats, hist = build_features(df)
    # drop the first season from training: its form features have no history yet
    train = feats[feats["season"] > a.start]

    if a.backtest:
        backtest(train, a.backtest)
    else:
        rain = None if a.rain is None else max(0.0, min(a.rain, 100.0)) / 100
        predict_next(df, train, hist, rain, a.grid, a.top)


if __name__ == "__main__":
    main()
