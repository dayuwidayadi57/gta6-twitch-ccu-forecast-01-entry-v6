[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dayuwidayadi57/gta6-twitch-ccu-forecast-01-entry-v6/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dayuwidayadi57/gta6-twitch-ccu-forecast-01-entry-v6/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecaster

Entry for [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) competition `gta6-twitch-ccu-forecast-01`:
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), as five quantiles per hour, from
the first hour's print.

## How it forecasts

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads the case file and, for each hour 2–24:

1. Takes every past big launch in the case's `comparables` with a usable
   release-day curve (early-access launches and live-service entries are
   excluded — their release-day shape is not a launch shape).
2. Fits `ln(viewers_h / viewers_1)` against `ln(viewers_1)` with a quadratic
   least-squares fit. A launch that opens small grows a lot as the US wakes
   up; one that opens huge is already near its peak — the fit captures that.
   The prediction is clamped to the range of ratios actually observed.
3. Sets the spread from the fit residuals (IQR scaled to a sigma, shrunk
   by 0.7, floored at 0.25) and emits
   `q_tau = p1 * exp(median + z_tau * spread)`.

If no comparable is usable it falls back to a linear fit, then the median
shape, then a persistence baseline. Every quantile is clamped into
`(0, 1e9]` in non-decreasing order.

On the four practice launches it scores a mean pinball loss of **0.144**,
against **0.387** for the house persistence baseline
(p00 0.117, p01 0.152, p02 0.122, p03 0.186).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` file in the input directory and writes one
`<id>.out` forecast per case. `main.py` alone is the whole entry.

## License

MIT — see [LICENSE](LICENSE).
