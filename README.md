# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/itzalisha99/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/itzalisha99/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z — as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per trophy. Built for the [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty on make.fun.

## How it works (V2 — ensemble)

`main.ts` (Node 22, standard library only, no network) reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case. Deterministic: same input, byte-identical output. Runs in ~0.4s per case.

V2 rebuilds V1's single-model forecast as an ensemble — the fix the sister Twitch forecast's top 10 all used (their losses 0.066–0.080 with 90% coverage 0.83–1.00, against our single model's 0.199 / 0.48 there). V1 here scored 0.170 on the practice mid scenario with coverage 1.00: its median was decent but its band was far too wide, so the loss was mostly spread cost. V2 attacks both halves:

1. **Median — three sound members.** Six views of each trophy are fitted on the logit scale against `x = ln(1 + hours)`: direct WLS trend with slopes shrunk to tier priors (V1's model), a log-ratio to a mid-rate reference trophy (ratios cancel the common dilution from Exophase's player growth), a concave quadratic that can saturate, a kernel-weighted local-linear fit, the house-curve OLS over the last 24h, and a tier-pooled slope. The median is carried only by the first three — the ones whose extrapolation is sound at 4.5 days — weighted by long-horizon rolling-origin scores (each member refitted on history up to each past snapshot, scored predicting the last snapshot). The other three set the spread, not the median: origin scoring cannot validate a 4.5-day extrapolation, and on the synthetic final it ranked the steep-growth chasers first while they overshot the flat week-one regime by up to +0.7 logit. No bias correction is applied: the rolling-origin signed error measures early-growth behaviour that does not carry over to the flat week-one regime.
2. **Launch dilution.** At launch Exophase's tracked players grow much faster than earners, so a refresh after a long stale stretch cuts most rates. When a case's history has barely changed, a dilution prior takes over: players keep growing at the launch rate over the stale stretch plus the horizon, earners keep up with only a share of that growth (capped at 0.70 for the 100% first trophy, with a little catch-up over the horizon itself). It fades out as real history accumulates.
3. **Spread — calibrated, not just wide.** Built from honest parts: the rolling-origin residual scale, the ensemble's own disagreement (when the members disagree the band widens), a smaller random-walk term, a refresh-jump term measured from the history's own change points, slope uncertainty, and the dilution model's own uncertainty — with an asymmetric band (refreshes at launch mostly push rates down, fading as the checkpoint ages). The legs are scaled by 0.45, chosen as a robustness compromise: 0.35 was best on clean synthetic worlds (0.076) but 0.45 holds coverage 0.98 when the truth is shifted ×0.7/×1.4, for 0.095 clean.
4. **Ordering.** Story chapters must not rise (finishing Chapter 2 implies Chapter 1), and the platinum — which needs every other trophy — must not exceed any other trophy.

## Run it locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Backtests (my own, not official scores)

Practice cases p00/p01 (the answers are not in the kit, so three plausible refresh severities were scored — rates falling to 45% / 30% / 20% of launch-night values after the first refresh, players 638 → 3,868):

| case | V2 | V1 | house baseline (starter) |
| --- | --- | --- | --- |
| p00 mild | 0.138 | 0.210 | 0.857 |
| p00 mid | 0.122 | 0.170 | 1.275 |
| p00 hard | 0.294 | 0.253 | 1.661 |
| p01 mild | 0.254 | 0.310 | — |
| p01 mid | 0.112 | 0.187 | 1.097 |
| p01 hard | 0.140 | 0.185 | — |

Practice mid mean: V2 **0.117** vs V1 0.178. (The hard/mild tails trade off against each other — no single median multiplier wins both; V2's sits near the geometric centre of the range.)

Synthetic final (a simulated 2-day history cut at the deadline, target a week after launch, lognormal time-to-earn truth, Exophase refreshes every 3–5h): V2 **0.105** mean loss across 15 simulated worlds (slower/faster adoption, refreshes every 1.5–9h), beating the house curve baseline in all 15 (house curve 0.262, house baseline 0.385, V1 0.214). Typical miss 0.4–2.9pp, median logit error ~0.00. Coverage is 1.00 on these worlds because the median is that accurate inside the lognormal family — the 0.45 spread scale is kept as a hedge against the real truth deviating from that family, not because the synthetic worlds need it.

The practice pays nothing and the final is what counts — but the same machinery (ensemble median, dilution prior, calibrated spread) is what the final case will run on.

Try make.fun: https://make.fun/r/y05d5qm6

Unofficial fan project. Grand Theft Auto VI is developed and published by Rockstar Games. No Rockstar artwork is used.
