#!/usr/bin/env python3
"""Stage D1 originals and R1 time-aligned TextGrids for EAR-1 dev sessions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import soundfile as sf

from analyze_der import select_wav
from package_train_aligned_textgrids import verify_shifted
from shift_textgrid import load_tiers, shift_tier, write_textgrid


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--alignment-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    with args.alignment_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 7 or len({row["session"] for row in rows}) != 7:
        raise ValueError("Expected seven distinct development sessions")

    for row in rows:
        session = row["session"]
        shift = float(row["reference_shift_s"])
        session_dir = data_root / session
        target_dir = output_dir / session
        target_dir.mkdir(parents=True, exist_ok=True)
        source_d1 = session_dir / "D1.TextGrid"
        target_d1 = target_dir / "D1.TextGrid"
        shutil.copyfile(source_d1, target_d1)
        if sha256(source_d1) != sha256(target_d1):
            raise ValueError(f"D1 copy changed: {session}")

        duration = sf.info(str(select_wav(session_dir, "R1"))).duration
        source = load_tiers(source_d1)
        aligned = [
            (name, shift_tier(intervals, shift, duration))
            for name, intervals in source
        ]
        target_r1 = target_dir / "R1.TextGrid"
        write_textgrid(target_r1, aligned, duration)
        retained = verify_shifted(source, load_tiers(target_r1), shift, duration)
        print(
            f"{session}: D1 unchanged; R1 shift {shift:+.2f}s, "
            f"duration {duration:.3f}s, {retained} speech intervals",
            flush=True,
        )
    print(f"Staged {len(rows) * 2} TextGrids under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
