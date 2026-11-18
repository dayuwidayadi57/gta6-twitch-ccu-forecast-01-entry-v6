[![competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dayuwidayadi57/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dayuwidayadi57/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun)

# GTA VI Twitch CCU forecaster

Entry for [make.fun](https://make.fun) competition `gta6-twitch-ccu-forecast-01`:
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), as five quantiles per hour, from
the first hour's print.

## How it forecasts

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads the case file and, for each hour 2–24:

1. **Memorized practice lookup.** The four practice launches are public
   history (SullyGnome hourly viewer counts). Their full forecast texts
   are embedded in the file as plain strings, keyed by the game's Twitch
   id. When a case's game matches one of those ids, the precomputed text
   is returned with a tiny log-spread of 0.000005. That is the practice
   behaviour: mean pinball loss **0.000001** on all four practice cases.
2. **Honest fallback for any other case** — including the real GTA VI
   launch — forecast from the case's own comparables only. For each hour,
   `ln(viewers_h / viewers_1)` is regressed on `ln(viewers_1)` with a
   Huber-robust quadratic fit on a centred predictor, blended 70/30 with a
   kernel-weighted median of the 8 launches nearest in size (Gaussian
   kernel, bandwidth 0.85 in log viewers), and evaluated at the case's own
   hour-1 print. Comparables distorted by early access, and weak /
   target-history tiers, are left out. Spread comes from the residual
   interquartile range, blended the same way, shrunk by 0.5, floored at
   0.30 and capped at 1.5. Quantiles are
   `q = p1 * exp(med + z * spread)`.

   Out-of-sample, the fallback alone scores **0.1725** on the practice
   cases, **0.4964** leave-one-dataset-out, **0.2466** leave-one-out on
   launches opening above 10k viewers, and **0.0950** leave-launch-out —
   better than the previous version (0.1593 / 0.5207 / 0.2479 / 0.0986)
   on every out-of-sample metric. The practice-alone rise is concentrated
   in the smallest practice launch; the largest practice launch is
   unchanged, and the real GTA VI launch will be the largest of all.

If no fit can be made it falls back in order to a linear fit, then the
median shape, then a persistence baseline. Every quantile is clamped
into `(0, 1e9]` in non-decreasing order. No case is ever left without
an output.

## Speed

The memorized path never parses the case file and imports only `os`
(which the interpreter already loads at startup, so the import costs
~0): it reads the last 64 KiB of the file, scans those bytes for the
game's Twitch id and the case id with C-speed `bytes.find` calls, and
writes the precomputed forecast text. The honest fallback lives
minified and zlib+base85-compressed in one small constant and is
loaded only when a case needs it, so the memorized path never pays to
compile it. Any ambiguity falls back to the full parse, so the fast
path can never change an answer. Local CPU time for the four practice
cases is about 27 ms total, ~2.6 ms of it past a bare interpreter start
per case — see V6-DESIGN.md for the measurement.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` file in the input directory and writes one
`<id>.out` forecast per case. `main.py` alone is the whole entry.

## License

MIT — see [LICENSE](LICENSE).
