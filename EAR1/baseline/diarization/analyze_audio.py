#!/usr/bin/env python3
"""Lightweight channel-level audio diagnostics for EAR-1 DER analysis."""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_right
from pathlib import Path

import numpy as np
import soundfile as sf


def select_wav(session_dir: Path, track: str) -> Path:
    matches = [
        path
        for path in session_dir.iterdir()
        if path.is_file()
        and path.stem.upper() == track.upper()
        and path.suffix.lower() == ".wav"
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {track} WAV in {session_dir}: {matches}")
    return matches[0]


def speech_support(path: Path) -> tuple[list[tuple[float, float]], list[float]]:
    intervals = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if parts and parts[0] == "SPEAKER":
            start, duration = float(parts[3]), float(parts[4])
            intervals.append((start, start + duration))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged, [row[0] for row in merged]


def contains(intervals: list[tuple[float, float]], starts: list[float], time: float) -> bool:
    index = bisect_right(starts, time) - 1
    return index >= 0 and intervals[index][0] <= time < intervals[index][1]


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks", nargs="+", default=["D1", "R1"])
    args = parser.parse_args()

    sessions = sorted(path.stem for path in args.reference_root.glob("*.rttm"))
    rows = []
    for session in sessions:
        intervals, starts = speech_support(args.reference_root / f"{session}.rttm")
        for track in args.tracks:
            wav = select_wav(args.data_root / session, track)
            total_squares = 0.0
            total_samples = 0
            clipped_samples = 0
            block_levels = []
            speech_levels = []
            silence_levels = []
            with sf.SoundFile(str(wav)) as stream:
                sample_rate = stream.samplerate
                for block_index, block in enumerate(
                    stream.blocks(blocksize=sample_rate, dtype="float32", always_2d=False)
                ):
                    values = np.asarray(block, dtype=np.float64)
                    square_sum = float(values @ values)
                    total_squares += square_sum
                    total_samples += values.size
                    clipped_samples += int(np.count_nonzero(np.abs(values) >= 0.999))
                    level = dbfs(math.sqrt(square_sum / values.size))
                    block_levels.append(level)
                    center = block_index + 0.5 * values.size / sample_rate
                    (speech_levels if contains(intervals, starts, center) else silence_levels).append(level)
            percentiles = np.percentile(block_levels, [10, 50, 90])
            speech_median = float(np.median(speech_levels)) if speech_levels else None
            silence_median = float(np.median(silence_levels)) if silence_levels else None
            rows.append(
                {
                    "session": session,
                    "track": track,
                    "sample_rate": sample_rate,
                    "duration_s": total_samples / sample_rate,
                    "overall_rms_dbfs": dbfs(math.sqrt(total_squares / total_samples)),
                    "one_second_rms_p10_dbfs": percentiles[0],
                    "one_second_rms_p50_dbfs": percentiles[1],
                    "one_second_rms_p90_dbfs": percentiles[2],
                    "speech_center_blocks": len(speech_levels),
                    "silence_center_blocks": len(silence_levels),
                    "speech_median_dbfs": speech_median,
                    "silence_median_dbfs": silence_median,
                    "speech_silence_contrast_db": (
                        speech_median - silence_median
                        if speech_median is not None and silence_median is not None
                        else None
                    ),
                    "clipped_sample_pct": 100.0 * clipped_samples / total_samples,
                }
            )
            print(f"Analyzed {session}/{track}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audio diagnostics written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
