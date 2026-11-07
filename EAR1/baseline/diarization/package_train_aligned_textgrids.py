#!/usr/bin/env python3
"""Package per-session R1 TextGrids shifted onto the R1 training audio timeline."""

from __future__ import annotations

import argparse
import csv
import io
import math
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

from analyze_der import select_wav
from ear1_baseline import read_split
from shift_textgrid import load_tiers, shift_tier, write_textgrid


def expected_speech(intervals, shift: float, duration: float):
    return [
        (max(0.0, start + shift), min(duration, end + shift), label)
        for start, end, label in intervals
        if label and min(duration, end + shift) > max(0.0, start + shift)
    ]


def verify_shifted(source, generated, shift: float, duration: float) -> int:
    if [name for name, _ in source] != [name for name, _ in generated]:
        raise ValueError("The generated TextGrid changed its speaker tiers")
    retained = 0
    for (name, old_intervals), (_, new_intervals) in zip(source, generated):
        if not math.isclose(new_intervals[0][0], 0.0, abs_tol=1e-6):
            raise ValueError(f"{name} does not start at zero")
        if not math.isclose(new_intervals[-1][1], duration, abs_tol=1e-6):
            raise ValueError(f"{name} does not end at the R1 audio boundary")
        actual = [(start, end, label) for start, end, label in new_intervals if label]
        wanted = expected_speech(old_intervals, shift, duration)
        if len(actual) != len(wanted):
            raise ValueError(f"{name} lost or gained speech intervals")
        for (start, end, label), (want_start, want_end, want_label) in zip(actual, wanted):
            if label != want_label or not math.isclose(start, want_start, abs_tol=1e-6) or not math.isclose(end, want_end, abs_tol=1e-6):
                raise ValueError(f"{name} speech or text changed after time shifting")
        retained += len(actual)
    return retained


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--alignment-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    expected_sessions = set(read_split(data_root, args.split.resolve())["train"])
    with args.alignment_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(expected_sessions) or {row["session"] for row in rows} != expected_sessions:
        raise ValueError("Alignment CSV does not contain exactly the training split")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with tempfile.TemporaryDirectory(prefix="ear1_train_textgrids_", dir=args.output.parent) as temporary:
        staging = Path(temporary)
        archive_temp = staging / "aligned_textgrids.zip"
        with ZipFile(archive_temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for row in sorted(rows, key=lambda item: item["session"]):
                session = row["session"]
                shift = float(row["reference_shift_s"])
                source_path = data_root / session / "D1.TextGrid"
                audio_path = select_wav(data_root / session, "R1")
                duration = sf.info(str(audio_path)).duration
                source = load_tiers(source_path)
                aligned = [(name, shift_tier(intervals, shift, duration)) for name, intervals in source]
                target = staging / session / "R1.TextGrid"
                write_textgrid(target, aligned, duration)
                retained = verify_shifted(source, load_tiers(target), shift, duration)
                archive.write(target, f"{session}/R1.TextGrid")
                manifest.append({
                    "session": session,
                    "r1_delay_early_s": row["r1_delay_early_s"],
                    "r1_delay_late_s": row["r1_delay_late_s"],
                    "reference_shift_s": row["reference_shift_s"],
                    "aligned_corr_early": row["aligned_corr_early"],
                    "aligned_corr_late": row["aligned_corr_late"],
                    "r1_audio_duration_s": duration,
                    "speech_intervals_retained": retained,
                    "source": "D1.TextGrid",
                })
                print(f"{session}: shift {shift:+.2f}s, retained {retained} speech intervals", flush=True)
            text = io.StringIO()
            writer = csv.DictWriter(text, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
            archive.writestr("alignment_manifest.csv", text.getvalue())
            archive.writestr(
                "README.txt",
                "EAR-1 training R1 TextGrids aligned to R1 audio.\n"
                "Each session/R1.TextGrid is derived from that session's D1.TextGrid,\n"
                "shifted by reference_shift_s in alignment_manifest.csv, and clipped\n"
                "to the R1 WAV duration. Positive shifts move labels later.\n"
                "The offsets are acoustic estimates, not organizer-issued corrections.\n"
                "Original audio and TextGrid files are unchanged.\n",
            )
        with ZipFile(archive_temp) as archive:
            if archive.testzip() is not None or len(archive.namelist()) != len(rows) + 2:
                raise ValueError("ZIP integrity check failed")
        archive_temp.replace(args.output)
    print(f"Packaged {len(manifest)} aligned TextGrids: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
