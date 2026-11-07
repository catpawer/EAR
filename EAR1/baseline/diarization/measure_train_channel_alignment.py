#!/usr/bin/env python3
"""Measure D1/R1 timing at the beginning and end of each EAR-1 training session."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate

from analyze_channel_alignment import STEP_SECONDS, energy_envelope
from analyze_der import select_wav
from ear1_baseline import read_split


def best_delay(d1: np.ndarray, r1: np.ndarray, max_lag_s: float) -> tuple[float, float]:
    """Positive delay means that an event occurs later in R1 than in D1."""
    count = min(len(d1), len(r1))
    d1, r1 = d1[:count], r1[:count]
    if count < 100:
        raise ValueError("Too few audio frames for alignment")
    cross = correlate(d1, r1, mode="full", method="fft")
    d1_sum = np.r_[0.0, np.cumsum(d1)]
    r1_sum = np.r_[0.0, np.cumsum(r1)]
    d1_square = np.r_[0.0, np.cumsum(d1 * d1)]
    r1_square = np.r_[0.0, np.cumsum(r1 * r1)]
    max_frames = min(round(max_lag_s / STEP_SECONDS), count - 100)
    best = (0, -math.inf)
    for lag in range(-max_frames, max_frames + 1):
        if lag < 0:
            d_start, d_end = 0, count + lag
            r_start, r_end = -lag, count
        else:
            d_start, d_end = lag, count
            r_start, r_end = 0, count - lag
        shared = d_end - d_start
        sum_d = d1_sum[d_end] - d1_sum[d_start]
        sum_r = r1_sum[r_end] - r1_sum[r_start]
        var_d = (d1_square[d_end] - d1_square[d_start]) - sum_d * sum_d / shared
        var_r = (r1_square[r_end] - r1_square[r_start]) - sum_r * sum_r / shared
        if var_d <= 0 or var_r <= 0:
            continue
        covariance = cross[lag + count - 1] - sum_d * sum_r / shared
        correlation = covariance / math.sqrt(var_d * var_r)
        if correlation > best[1]:
            best = lag, float(correlation)
    if abs(best[0]) == max_frames:
        raise ValueError(f"Best delay reached search boundary: {best[0] * STEP_SECONDS}s")
    return -best[0] * STEP_SECONDS, best[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-lag-seconds", type=float, default=60.0)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    sessions = sorted(read_split(data_root, args.split.resolve())["train"])
    rows = []
    for session in sessions:
        d1 = select_wav(data_root / session, "D1")
        r1 = select_wav(data_root / session, "R1")
        duration = min(sf.info(str(d1)).duration, sf.info(str(r1)).duration)
        if duration < 180:
            raise ValueError(f"Session too short for separate 120-second windows: {session}")
        windows = ((0.0, 120.0), (duration - 150.0, duration - 30.0))
        measurements = [
            best_delay(
                energy_envelope(d1, start, end),
                energy_envelope(r1, start, end),
                args.max_lag_seconds,
            )
            for start, end in windows
        ]
        early, late = (measurement[0] for measurement in measurements)
        row = {
            "session": session,
            "r1_delay_early_s": early,
            "r1_delay_late_s": late,
            "reference_shift_s": (early + late) / 2,
            "aligned_corr_early": measurements[0][1],
            "aligned_corr_late": measurements[1][1],
            "early_late_difference_s": late - early,
        }
        rows.append(row)
        print(
            f"{session}: early {early:+.2f}s, late {late:+.2f}s, "
            f"reference shift {row['reference_shift_s']:+.2f}s, "
            f"correlation {measurements[0][1]:.2f}/{measurements[1][1]:.2f}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Measured {len(rows)} training sessions: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
