# F1 Race Winner Predictor

Predicts who will win the next Formula 1 race. It learns from past races since 2018 and looks at:

- **Grid position**: where each driver starts (from real qualifying once it's happened)
- **Form**: average finish and points over each driver's last 5 races, and the team's last 10
- **Track history**: how each driver has finished at this circuit before
- **Season so far**: championship points and wins this year
- **Reliability**: how often each driver failed to finish recently
- **Rain**: whether it rained during each past race (real weather records), and each driver's **rain skill**, meaning how many places better or worse they finish in the wet than in the dry
- **Heat**: race temperature, and whether each driver does better or worse in hot races

For the next race it checks the weather forecast for the circuit. It makes one prediction for a dry race and one for a wet race, then mixes them by the chance of rain. A 60% chance of rain counts the wet prediction for 60%.

All the data is free and needs no account:
[Jolpica F1 API](https://api.jolpi.ca) for results and [Open-Meteo](https://open-meteo.com) for weather.

## Setup

```bash
pip install -r requirements.txt
```

## Use it

Predict the next race:

```bash
python f1_predictor.py
```

The first run downloads about 8 seasons of results and weather, which takes a few minutes. After that the data is saved in `cache/`, so later runs are quick.

Set the chance of rain yourself, for example when the race is too far away for a forecast:

```bash
python f1_predictor.py --rain 70
```

Give the starting grid yourself (driver ids, in grid order):

```bash
python f1_predictor.py --grid norris,piastri,verstappen,leclerc
```

Test how good it is. This trains on earlier seasons only, predicts every race of 2025, and compares the results with simply picking the pole sitter:

```bash
python f1_predictor.py --backtest 2025
```

Other options: `--from 2014` to learn from more seasons, `--top 20` to show every driver.

## Example output

```
Las Vegas Grand Prix 2026  -  Las Vegas Strip Street Circuit, 2026-11-21
  chance of rain: 10% (forecast)
  temperature:    14°C
  grid: not qualified yet, using each driver's average recent grid

     Driver                 Grid  If dry  If wet   WIN %  rain skill
   1 ...
```

## How it works

1. Every race is processed in date order, so each prediction only uses races that came before it, never the future.
2. A gradient-boosted decision tree model (scikit-learn) learns how those features relate to winning.
3. Each driver's win chance is scaled so that a race adds up to 100%.

F1 is unpredictable, and nothing here knows about upgrades, penalties or crashes before they happen. Treat the output as a smart guess, not a sure thing.
