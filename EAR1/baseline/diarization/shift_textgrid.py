#!/usr/bin/env python3
"""Shift every IntervalTier in a long TextGrid onto an audio file's timeline."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import soundfile as sf

from ear1_baseline import INTERVAL_RE, NAME_RE, TIER_RE


def load_tiers(path: Path) -> list[tuple[str, list[tuple[float, float, str]]]]:
    source = path.read_text(encoding="utf-8-sig")
    chunks = TIER_RE.split(source)[1:]
    if not chunks:
        raise ValueError(f"No TextGrid tiers found: {path}")
    tiers = []
    for chunk in chunks:
        if not re.search(r'^\s*class\s*=\s*"IntervalTier"\s*$', chunk, re.M):
            raise ValueError("Only IntervalTier TextGrids are supported")
        name = NAME_RE.search(chunk)
        if not name:
            raise ValueError(f"Missing tier name in {path}")
        intervals = [
            (float(item.group(1)), float(item.group(2)), item.group(3).replace('""', '"'))
            for item in INTERVAL_RE.finditer(chunk)
        ]
        declared = re.search(r"intervals:\s*size\s*=\s*(\d+)", chunk)
        if not declared or len(intervals) != int(declared.group(1)):
            raise ValueError(f"Interval count mismatch in tier {name.group(1)}")
        if not intervals or not math.isclose(intervals[0][0], 0.0, abs_tol=1e-6):
            raise ValueError(f"Tier {name.group(1)} does not begin at zero")
        for previous, current in zip(intervals, intervals[1:]):
            if not math.isclose(previous[1], current[0], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Discontinuous tier {name.group(1)}")
        tiers.append((name.group(1), intervals))
    return tiers


def shift_tier(
    intervals: list[tuple[float, float, str]], shift: float, duration: float
) -> list[tuple[float, float, str]]:
    result: list[tuple[float, float, str]] = []
    if shift > 0:
        result.append((0.0, min(shift, duration), ""))
    for start, end, label in intervals:
        left, right = max(0.0, start + shift), min(duration, end + shift)
        if right <= left:
            continue
        if result:
            previous_end = result[-1][1]
            if left > previous_end + 1e-6:
                result.append((previous_end, left, ""))
            elif left < previous_end - 1e-6:
                raise ValueError("Shifted intervals overlap")
            left = result[-1][1]
        if result and not label and not result[-1][2]:
            result[-1] = (result[-1][0], right, "")
        else:
            result.append((left, right, label))
    if not result:
        result.append((0.0, duration, ""))
    elif result[-1][1] < duration - 1e-6:
        if not result[-1][2]:
            result[-1] = (result[-1][0], duration, "")
        else:
            result.append((result[-1][1], duration, ""))
    if not math.isclose(result[0][0], 0.0, abs_tol=1e-6):
        raise ValueError("Shifted tier does not start at zero")
    if not math.isclose(result[-1][1], duration, abs_tol=1e-6):
        raise ValueError("Shifted tier does not cover the audio duration")
    for previous, current in zip(result, result[1:]):
        if not math.isclose(previous[1], current[0], rel_tol=0, abs_tol=1e-6):
            raise ValueError("Shifted tier is discontinuous")
    return result


def write_textgrid(
    path: Path, tiers: list[tuple[str, list[tuple[float, float, str]]]], duration: float
) -> None:
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration:.6f}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for index, (name, intervals) in enumerate(tiers, 1):
        escaped_name = name.replace('"', '""')
        lines.extend([
            f"    item [{index}]:",
            '        class = "IntervalTier"',
            f'        name = "{escaped_name}"',
            "        xmin = 0",
            f"        xmax = {duration:.6f}",
            f"        intervals: size = {len(intervals)}",
        ])
        for number, (start, end, label) in enumerate(intervals, 1):
            escaped_label = label.replace('"', '""')
            lines.extend([
                f"        intervals [{number}]:",
                f"            xmin = {start:.6f}",
                f"            xmax = {end:.6f}",
                f'            text = "{escaped_label}"',
            ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--shift-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    duration = sf.info(str(args.audio)).duration
    source = load_tiers(args.input)
    aligned = [
        (name, shift_tier(intervals, args.shift_seconds, duration))
        for name, intervals in source
    ]
    write_textgrid(args.output, aligned, duration)
    # Parse the written Praat file again to catch malformed interval blocks.
    reparsed = load_tiers(args.output)
    if len(reparsed) != len(source):
        raise ValueError("Written TextGrid lost tiers")
    for name, intervals in reparsed:
        if not math.isclose(intervals[-1][1], duration, abs_tol=1e-6):
            raise ValueError(f"Tier {name} exceeds audio duration")
    print(f"Written {args.output.resolve()}: {len(aligned)} tiers, shift {args.shift_seconds:+.3f}s, duration {duration:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
