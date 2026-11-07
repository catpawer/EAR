#!/usr/bin/env python3
"""Estimate R1-versus-D1 timing from audio energy and rescore shifted references."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from analyze_der import read_rttm, select_wav


STEP_SECONDS = 0.02
MAX_LAG_SECONDS = 3.0


def energy_envelope(path: Path, start: float, end: float) -> np.ndarray:
    with sf.SoundFile(str(path)) as audio:
        start = max(0.0, start)
        end = min(end, audio.frames / audio.samplerate)
        audio.seek(round(start * audio.samplerate))
        samples = audio.read(round((end - start) * audio.samplerate), dtype="float32")
        frame_size = round(STEP_SECONDS * audio.samplerate)
    frames = samples[: len(samples) // frame_size * frame_size].reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return np.clip(20 * np.log10(np.maximum(rms, 1e-10)), -90, -20)


def best_lag(d1: np.ndarray, r1: np.ndarray) -> tuple[float, float, float]:
    count = min(len(d1), len(r1))
    d1, r1 = d1[:count], r1[:count]
    d1 = (d1 - d1.mean()) / (d1.std() + 1e-12)
    r1 = (r1 - r1.mean()) / (r1.std() + 1e-12)
    zero_corr = float(np.corrcoef(d1, r1)[0, 1])
    winner = (0, -math.inf)
    max_frames = round(MAX_LAG_SECONDS / STEP_SECONDS)
    for lag in range(-max_frames, max_frames + 1):
        if lag < 0:
            left, right = d1[:lag], r1[-lag:]
        elif lag > 0:
            left, right = d1[lag:], r1[:-lag]
        else:
            left, right = d1, r1
        correlation = float(np.corrcoef(left, right)[0, 1])
        if correlation > winner[1]:
            winner = lag, correlation
    # Positive R1 delay means that R1 audio occurs later than D1.
    return -winner[0] * STEP_SECONDS, winner[1], zero_corr


def make_annotation(segments, session: str, duration: float, shift: float) -> Annotation:
    annotation = Annotation(uri=session)
    for item in segments:
        start, end = item.start + shift, item.end + shift
        if end > 0 and start < duration:
            annotation[Segment(max(0.0, start), min(duration, end)), item.line_no] = item.speaker
    return annotation


def score(reference, hypothesis, session: str, duration: float, shift: float) -> dict:
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    uem = Timeline([Segment(0.0, duration)], uri=session)
    result = metric(
        make_annotation(reference, session, duration, shift),
        make_annotation(hypothesis, session, duration, 0.0),
        uem=uem,
        detailed=True,
    )
    return {
        "der_pct": 100 * result["diarization error rate"],
        "reference_s": result["total"],
        "miss_s": result["missed detection"],
        "false_alarm_s": result["false alarm"],
        "confusion_s": result["confusion"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/der_analysis/channel_alignment_diagnostics.csv"))
    args = parser.parse_args()
    data_root, results_root, output = (
        args.data_root.resolve(), args.results_root.resolve(), args.output.resolve()
    )
    sessions = sorted(path.stem for path in (results_root / "prepare" / "dev").glob("*.rttm"))
    rows = []
    for session in sessions:
        d1_path = select_wav(data_root / session, "D1")
        r1_path = select_wav(data_root / session, "R1")
        duration = sf.info(str(r1_path)).duration
        early = (0.0, 120.0)
        late = (max(0.0, duration - 150.0), max(0.0, duration - 30.0))
        correlations = []
        for start, end in (early, late):
            correlations.append(best_lag(
                energy_envelope(d1_path, start, end),
                energy_envelope(r1_path, start, end),
            ))
        shift = float(np.median([item[0] for item in correlations]))
        reference = read_rttm(results_root / "prepare" / "dev" / f"{session}.rttm", session, math.inf)
        hypothesis = read_rttm(results_root / "infer" / "dev" / "R1" / f"{session}.rttm", session, math.inf)
        original = score(reference, hypothesis, session, duration, 0.0)
        shifted = score(reference, hypothesis, session, duration, shift)
        rows.append({
            "session": session,
            "r1_delay_early_s": correlations[0][0],
            "r1_delay_late_s": correlations[1][0],
            "reference_shift_s": shift,
            "aligned_corr_early": correlations[0][1],
            "zero_lag_corr_early": correlations[0][2],
            "aligned_corr_late": correlations[1][1],
            "zero_lag_corr_late": correlations[1][2],
            "original_r1_der_pct": original["der_pct"],
            "shifted_r1_der_pct_diagnostic_only": shifted["der_pct"],
            "original_miss_s": original["miss_s"],
            "shifted_miss_s": shifted["miss_s"],
            "original_false_alarm_s": original["false_alarm_s"],
            "shifted_false_alarm_s": shifted["false_alarm_s"],
            "original_confusion_s": original["confusion_s"],
            "shifted_confusion_s": shifted["confusion_s"],
        })
        print(
            f"{session}: R1 delay {shift:+.2f}s; "
            f"DER {original['der_pct']:.2f}% -> {shifted['der_pct']:.2f}% (diagnostic)",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Diagnostics: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
