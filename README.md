# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/itzalisha99/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/itzalisha99/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z — as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per trophy. Built for the [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty on make.fun.

## How it works

`main.ts` (Node 22, standard library only, no network) reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case. Deterministic: same input, byte-identical output. Runs in ~0.2s.

The model, per trophy:

1. **Growth.** On the logit scale a rate moves roughly linearly against log time since release, `x = ln(1 + hours)`. The median is a weighted fit of `logit(rate)` against `x` over the case's distinct page states, with gentle recency weights and slopes shrunk toward a tier prior (early bronze story trophies climb fastest, the platinum slowest). An earners-over-players fit is kept only as a fallback for trophies whose direct fit is degenerate.
2. **Launch dilution.** At launch Exophase's tracked players grow much faster than earners, so a refresh after a long stale stretch cuts most rates. When a case's history has barely changed, a dilution prior takes over: players keep growing at the launch rate over the stale stretch plus the horizon, earners keep up with only a share of that growth. It fades out as real history accumulates.
3. **Spread.** Built from honest parts — a random-walk term growing with the horizon, the fit's residual, the slope's standard error times the extrapolation distance, a refresh-jump term measured from the history's own change points, and the dilution model's own uncertainty — with an asymmetric band (refreshes at launch mostly push rates down). The legs are scaled by 0.8, calibrated on backtests.
4. **Ordering.** Story chapters must not rise (finishing Chapter 2 implies Chapter 1), and the platinum — which needs every other trophy — must not exceed any other trophy.

## Run it locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Backtests (my own, not official scores)

Practice cases p00/p01 (the answers are not in the kit, so three plausible refresh severities were scored — rates falling to 45% / 30% / 20% of launch-night values after the first refresh, players 638 → 3,868):

| case | this entry | house baseline (starter) |
| --- | --- | --- |
| p00 mild | 0.210 | 0.857 |
| p00 mid | 0.170 | 1.275 |
| p00 hard | 0.253 | 1.661 |
| p01 mid | 0.187 | 1.097 |

Synthetic final (a simulated 2-day history cut at the deadline, target a week after launch, lognormal time-to-earn truth, Exophase refreshes every 3–5h): this entry **0.201**, house curve baseline **0.256**, house baseline **0.385**. Across 15 simulated worlds (slower/faster adoption, refreshes every 1.5–9h) this entry beat the house curve baseline in all 15, with 90% coverage 1.00 throughout.

The practice pays nothing and the final is what counts — but the same machinery (trend fit, dilution prior, calibrated spread) is what the final case will run on.

Unofficial fan project. Grand Theft Auto VI is developed and published by Rockstar Games. No Rockstar artwork is used.
