"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Method, in brief. Exophase refreshes a game's page in jumps: between
refreshes the page is frozen, and at a refresh the tracked-player count
jumps, which usually drags every rate down (newly tracked players have
not earned much yet) before the rates climb again as the launch's
players progress. So the history is first collapsed into refresh
epochs (runs of identical reads), and each trophy is modelled in
logit space against log time since launch, ln(1 + hours):

* Trend regime (the history shows more than one distinct epoch): a
  least-squares line per trophy over the last day of epochs, its slope
  shrunk toward the median slope across trophies, extrapolated from
  the last epoch but damped (only 60% of the extrapolated move is
  taken), because a launch curve flattens as it saturates and a full
  extrapolation overshoots. The spread is calibrated by backtesting:
  the same forecaster is run from cutoffs 12 and 24 hours earlier and
  asked to predict the last epoch; the scale of its misses, stretched
  to the forecast's longer reach, sets the spread, with a small
  random-walk floor. A trophy already at 100% is pinned near 100%:
  it survived a sixfold jump in tracked players unchanged.

* Stale regime (the history is a single frozen page): no trend can be
  read, so a prior is used instead. A page that has sat unchanged for
  hours on launch night is overdue a refresh, and the first refresh
  dilutes rates hard, so the median is shifted down in logit space by
  an amount that grows with how long the page has been stale (capped),
  with a wide spread to cover how far the dilution may go.

Everything is deterministic: fixed iteration order, no randomness,
no network, standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Midnight UTC on release day; the clock the log-time transform runs on.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
NORMAL = NormalDist()

# How much of a fitted trend's extrapolated move to keep.
DAMP = 0.6
# Shrinkage strength for a trophy's slope toward the cross-trophy median.
SHRINK_K = 2.0
# Random-walk spread per square-root day (a floor, not the whole spread).
RW_PER_DAY = 0.15
# Spread bounds, in logit units.
SPREAD_FLOOR = 0.25
SPREAD_CAP = 1.2
# Hours of epochs used for a trend fit.
WINDOW_HOURS = 24.0
# Stale-page prior: down-shift = min(STALE_MAX, STALE_BASE + STALE_PER_HOUR * stale_hours).
STALE_BASE = 0.30
STALE_PER_HOUR = 0.35
STALE_MAX = 1.6
STALE_SPREAD = 0.85


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round to 4 decimals, half up, as the house baseline rounds."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def least_squares(xs, ys):
    """Return (intercept, slope, residual_sd, mean_x, sum_xx)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return mean_y, 0.0, 0.0, mean_x, sxx
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        var = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    else:
        var = 0.0
    return intercept, slope, math.sqrt(var), mean_x, sxx


def log_hours(moment):
    """ln(1 + hours from launch to a moment), never below zero."""
    hours = (moment - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def collapse_epochs(history):
    """One entry per refresh: consecutive identical reads are one epoch.

    Returns [(slot_time, rates), ...] in time order.
    """
    epochs = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read["rates"]))
        if epochs and epochs[-1][0] == key:
            continue
        epochs.append((key, parse_time(read["slot"]), read["rates"]))
    return [(slot, rates) for _key, slot, rates in epochs]


def trophy_points(epochs, index):
    """(log-time, logit rate) pairs for one trophy over the epochs."""
    points = []
    for slot, rates in epochs:
        if index >= len(rates):
            continue
        rate = rates[index]
        if rate is None:
            continue
        points.append((log_hours(slot), logit(rate)))
    return points


def global_fit(all_points):
    """Median slope and median residual spread across fittable trophies."""
    slopes = []
    spreads = []
    for points in all_points:
        if len(points) < 3:
            continue
        if len({y for _x, y in points}) <= 1:
            continue
        _a, slope, resid, _mx, _sxx = least_squares(
            [p[0] for p in points], [p[1] for p in points]
        )
        slopes.append(max(-2.5, min(2.5, slope)))
        spreads.append(resid)
    slope = median(slopes) if slopes else 0.0
    resid = median(spreads) if spreads else 0.3
    return max(-2.0, min(2.0, slope)), resid


def trend_median(points, x_target, ref_time, global_slope, global_resid):
    """Median logit at x_target, anchored on the last point.

    Fits a line over the last day of points (all points if fewer),
    shrinks its slope toward the global slope, and takes only DAMP of
    the extrapolated move from the last point.
    """
    window_start = math.log1p(
        max(0.0, (ref_time - LAUNCH).total_seconds() / 3600.0 - WINDOW_HOURS)
    )
    recent = [p for p in points if p[0] >= window_start]
    use = recent if len(recent) >= 3 else points
    xs = [p[0] for p in use]
    ys = [p[1] for p in use]
    _a, slope, resid, _mx, _sxx = least_squares(xs, ys)
    slope = max(-2.5, min(2.5, slope))
    n = len(use)
    weight = n / (n + SHRINK_K)
    slope = weight * slope + (1.0 - weight) * global_slope
    resid = (n * resid + SHRINK_K * global_resid) / (n + SHRINK_K) if n else global_resid
    median_logit = ys[-1] + DAMP * slope * (x_target - xs[-1])
    return median_logit, resid


def backtest_spread(epochs, all_points, x_target, last_x, at, global_resid):
    """Spread from backtesting the trend forecaster, or None.

    Forecasts the last epoch's value from cutoffs 12 and 24 hours
    earlier, pools the absolute logit misses, and stretches them from
    the backtest reach to the forecast reach (errors grow like the
    square root of the reach in log-time).
    """
    last_values = []
    for points in all_points:
        last_values.append(points[-1][1] if points else None)
    scaled_errors = []
    for back_hours in (12.0, 24.0):
        cut = datetime.fromtimestamp(at.timestamp() - back_hours * 3600, tz=timezone.utc)
        cut_epochs = [e for e in epochs if e[0] <= cut]
        if len(cut_epochs) < 3:
            continue
        cut_points = [trophy_points(cut_epochs, i) for i in range(len(all_points))]
        cut_slope, _cut_resid = global_fit(cut_points)
        x_cut = log_hours(cut_epochs[-1][0])
        reach = last_x - x_cut
        if reach <= 0.05:
            continue
        stretch = math.sqrt(min(3.0, max(0.2, (x_target - last_x) / reach)))
        for i, points in enumerate(cut_points):
            if last_values[i] is None or len(points) < 3:
                continue
            if len({y for _x, y in points}) <= 1:
                continue
            guess, _resid = trend_median(points, last_x, cut, cut_slope, global_resid)
            scaled_errors.append(abs(guess - last_values[i]) * stretch)
    if not scaled_errors:
        return None
    return 1.4826 * median(scaled_errors)


def forecast_case(case):
    levels = case.get("levels", [0.05, 0.25, 0.5, 0.75, 0.95])
    zeds = [NORMAL.inv_cdf(level) for level in levels]
    trophies = case["trophies"]
    history = case.get("history", [])
    at = parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = log_hours(parse_time(case["target"]))

    if not history:
        return {
            "trophies": [
                {
                    "id": t["id"],
                    "q": [round4(inv_logit(z * 1.0)) for z in zeds],
                }
                for t in trophies
            ]
        }

    epochs = collapse_epochs(history)
    last_rates = history[-1]["rates"]
    all_points = [trophy_points(epochs, i) for i in range(len(trophies))]
    global_slope, global_resid = global_fit(all_points)

    last_x = log_hours(epochs[-1][0]) if epochs else log_hours(at)
    spread_bt = None
    if len(epochs) >= 6:
        spread_bt = backtest_spread(
            epochs, all_points, x_target, last_x, at, global_resid
        )

    rows = []
    for i, trophy in enumerate(trophies):
        last = last_rates[i] if i < len(last_rates) else None
        if last is None:
            for _slot, rates in reversed(epochs):
                if i < len(rates) and rates[i] is not None:
                    last = rates[i]
                    break
        if last is None:
            last = 1.0
        if last >= 99.5:
            # Held at 100% through a sixfold jump in tracked players:
            # pin it near the top instead of forecasting a fall.
            rows.append({"id": trophy["id"], "q": [99.5, 99.9, 99.99, 100, 100]})
            continue
        points = all_points[i]
        if len(epochs) <= 1 or len(points) < 3 or len({y for _x, y in points}) <= 1:
            # Stale page: no movement to learn from. The page has sat
            # unchanged since the first read; a refresh is overdue and
            # the first refresh dilutes rates, so shift the median down
            # by an amount growing with the staleness, widely spread.
            first_slot = parse_time(history[0]["slot"])
            stale_hours = (at - first_slot).total_seconds() / 3600.0
            shift = min(STALE_MAX, STALE_BASE + STALE_PER_HOUR * stale_hours)
            centre = logit(last) - shift
            spread = STALE_SPREAD
        else:
            centre, _resid = trend_median(
                points, x_target, at, global_slope, global_resid
            )
            if inv_logit(centre) > 97.0:
                centre = logit(97.0)
            if inv_logit(centre) < 0.01:
                centre = logit(0.01)
            walk = RW_PER_DAY * math.sqrt(horizon_days)
            if spread_bt is not None:
                spread = math.sqrt(walk * walk + spread_bt * spread_bt)
            else:
                spread = math.sqrt(walk * walk + _resid * _resid)
            spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread))
        quantiles = [round4(inv_logit(centre + z * spread)) for z in zeds]
        quantiles = [min(100.0, max(0.0, q)) for q in quantiles]
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
