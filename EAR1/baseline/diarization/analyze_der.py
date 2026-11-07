#!/usr/bin/env python3
"""Detailed, globally-mapped DER diagnostics for the EAR-1 baseline.

The official baseline recomputes an optimal one-to-one speaker mapping once per
session.  This script keeps that mapping fixed when it drills down into time
windows, reference turns, and atomic error regions.  Re-optimizing a mapping in
each small window would hide speaker confusion and make the local DER misleading.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class RttmSegment:
    start: float
    end: float
    speaker: str
    line_no: int


@dataclass
class Atom:
    start: float
    end: float
    ref: Counter
    hyp_raw: Counter
    hyp_mapped: Counter
    ref_count: int
    hyp_count: int
    correct_count: int
    miss_count: int
    fa_count: int
    conf_count: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def read_rttm(path: Path, session: str, duration: float) -> list[RttmSegment]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.strip().split()
        if not parts or parts[0] != "SPEAKER":
            continue
        if len(parts) < 8 or parts[1] != session:
            raise ValueError(f"Invalid RTTM row at {path}:{line_no}")
        start = max(0.0, float(parts[3]))
        end = min(duration, float(parts[3]) + float(parts[4]))
        if end > start:
            rows.append(RttmSegment(start, end, parts[7], line_no))
    return rows


def raw_atoms(
    ref_segments: list[RttmSegment],
    hyp_segments: list[RttmSegment],
    duration: float,
) -> list[tuple[float, float, Counter, Counter]]:
    events: dict[float, dict[str, Counter]] = defaultdict(
        lambda: {"ref": Counter(), "hyp": Counter()}
    )
    events[0.0]
    events[duration]
    for kind, segments in (("ref", ref_segments), ("hyp", hyp_segments)):
        for segment in segments:
            events[segment.start][kind][segment.speaker] += 1
            events[segment.end][kind][segment.speaker] -= 1

    times = sorted(events)
    active_ref: Counter = Counter()
    active_hyp: Counter = Counter()
    rows = []
    for index, start in enumerate(times[:-1]):
        for label, delta in events[start]["ref"].items():
            active_ref[label] += delta
            if active_ref[label] == 0:
                del active_ref[label]
        for label, delta in events[start]["hyp"].items():
            active_hyp[label] += delta
            if active_hyp[label] == 0:
                del active_hyp[label]
        end = times[index + 1]
        if end > start:
            rows.append((start, end, active_ref.copy(), active_hyp.copy()))
    return rows


def global_mapping(
    atoms: list[tuple[float, float, Counter, Counter]],
) -> tuple[dict[str, str], dict[tuple[str, str], float], dict[str, float], dict[str, float]]:
    ref_duration: Counter = Counter()
    hyp_duration: Counter = Counter()
    cooccurrence: Counter = Counter()
    for start, end, ref, hyp in atoms:
        length = end - start
        for ref_label, count in ref.items():
            ref_duration[ref_label] += length * count
        for hyp_label, count in hyp.items():
            hyp_duration[hyp_label] += length * count
        for hyp_label, hyp_count in hyp.items():
            for ref_label, ref_count in ref.items():
                cooccurrence[hyp_label, ref_label] += length * min(hyp_count, ref_count)

    hyp_labels = sorted(hyp_duration)
    ref_labels = sorted(ref_duration)
    mapping: dict[str, str] = {}
    if hyp_labels and ref_labels:
        matrix = [
            [-cooccurrence[hyp_label, ref_label] for ref_label in ref_labels]
            for hyp_label in hyp_labels
        ]
        row_indices, col_indices = linear_sum_assignment(matrix)
        for row_index, col_index in zip(row_indices, col_indices):
            overlap = -matrix[row_index][col_index]
            if overlap > 0.0:
                mapping[hyp_labels[row_index]] = ref_labels[col_index]
    return mapping, dict(cooccurrence), dict(ref_duration), dict(hyp_duration)


def mapped_atoms(
    atoms: list[tuple[float, float, Counter, Counter]], mapping: dict[str, str]
) -> list[Atom]:
    rows = []
    for start, end, ref, hyp_raw in atoms:
        hyp_mapped: Counter = Counter()
        for label, count in hyp_raw.items():
            hyp_mapped[mapping.get(label, f"HYP::{label}")] += count
        ref_count = sum(ref.values())
        hyp_count = sum(hyp_mapped.values())
        correct_count = sum(
            min(count, hyp_mapped.get(label, 0)) for label, count in ref.items()
        )
        rows.append(
            Atom(
                start=start,
                end=end,
                ref=ref,
                hyp_raw=hyp_raw,
                hyp_mapped=hyp_mapped,
                ref_count=ref_count,
                hyp_count=hyp_count,
                correct_count=correct_count,
                miss_count=max(0, ref_count - hyp_count),
                fa_count=max(0, hyp_count - ref_count),
                conf_count=min(ref_count, hyp_count) - correct_count,
            )
        )
    return rows


def blank_components() -> dict[str, float]:
    return {
        "wall_s": 0.0,
        "ref_s": 0.0,
        "hyp_s": 0.0,
        "correct_s": 0.0,
        "miss_s": 0.0,
        "fa_s": 0.0,
        "conf_s": 0.0,
        "ref_speech_wall_s": 0.0,
        "hyp_speech_wall_s": 0.0,
        "ref_overlap_wall_s": 0.0,
        "speech_miss_wall_s": 0.0,
        "speech_fa_wall_s": 0.0,
    }


def add_atom(target: dict[str, float], atom: Atom, length: float) -> None:
    target["wall_s"] += length
    target["ref_s"] += length * atom.ref_count
    target["hyp_s"] += length * atom.hyp_count
    target["correct_s"] += length * atom.correct_count
    target["miss_s"] += length * atom.miss_count
    target["fa_s"] += length * atom.fa_count
    target["conf_s"] += length * atom.conf_count
    target["ref_speech_wall_s"] += length * (atom.ref_count > 0)
    target["hyp_speech_wall_s"] += length * (atom.hyp_count > 0)
    target["ref_overlap_wall_s"] += length * (atom.ref_count > 1)
    target["speech_miss_wall_s"] += length * (
        atom.ref_count > 0 and atom.hyp_count == 0
    )
    target["speech_fa_wall_s"] += length * (
        atom.ref_count == 0 and atom.hyp_count > 0
    )


def aggregate_atoms(
    atoms: list[Atom], start: float = 0.0, end: float = math.inf
) -> dict[str, float]:
    result = blank_components()
    for atom in atoms:
        overlap = min(atom.end, end) - max(atom.start, start)
        if overlap > 0:
            add_atom(result, atom, overlap)
    return result


def finish_components(values: dict[str, float]) -> dict[str, float | None]:
    result = dict(values)
    error = values["miss_s"] + values["fa_s"] + values["conf_s"]
    result["error_s"] = error
    result["der_pct"] = 100.0 * error / values["ref_s"] if values["ref_s"] else None
    for key in ("miss", "fa", "conf"):
        result[f"{key}_pct_ref"] = (
            100.0 * values[f"{key}_s"] / values["ref_s"]
            if values["ref_s"]
            else None
        )
        result[f"{key}_pct_error"] = (
            100.0 * values[f"{key}_s"] / error if error else None
        )
    result["ref_overlap_pct_speech_wall"] = (
        100.0 * values["ref_overlap_wall_s"] / values["ref_speech_wall_s"]
        if values["ref_speech_wall_s"]
        else None
    )
    result["speech_miss_pct_wall"] = (
        100.0 * values["speech_miss_wall_s"] / values["ref_speech_wall_s"]
        if values["ref_speech_wall_s"]
        else None
    )
    silence = values["wall_s"] - values["ref_speech_wall_s"]
    result["speech_fa_pct_silence_wall"] = (
        100.0 * values["speech_fa_wall_s"] / silence if silence > 0 else None
    )
    return result


def format_labels(counter: Counter) -> str:
    labels = []
    for label in sorted(counter):
        labels.extend([str(label)] * counter[label])
    return "+".join(labels)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_wav(session_dir: Path, track: str) -> Path:
    candidates = [
        path
        for path in session_dir.iterdir()
        if path.is_file()
        and path.stem.upper() == track.upper()
        and path.suffix.lower() == ".wav"
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {track} WAV in {session_dir}; found {candidates}"
        )
    return candidates[0]


def pyannote_scores(ref_path: Path, hyp_path: Path, session: str, duration: float) -> dict:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    def load(path: Path) -> Annotation:
        annotation = Annotation(uri=session)
        for segment in read_rttm(path, session, duration):
            annotation[Segment(segment.start, segment.end), segment.line_no] = segment.speaker
        return annotation

    reference, hypothesis = load(ref_path), load(hyp_path)
    uem = Timeline([Segment(0.0, duration)], uri=session)
    values = {}
    for collar in (0.0, 0.25):
        for skip_overlap in (False, True):
            metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
            detail = metric(reference, hypothesis, uem=uem, detailed=True)
            suffix = f"collar{collar:g}_{'skip_overlap' if skip_overlap else 'overlap'}"
            values[f"der_{suffix}_pct"] = 100.0 * detail["diarization error rate"]
            values[f"ref_{suffix}_s"] = detail["total"]
            values[f"miss_{suffix}_s"] = detail["missed detection"]
            values[f"fa_{suffix}_s"] = detail["false alarm"]
            values[f"conf_{suffix}_s"] = detail["confusion"]
    return values


def clip_atoms(atoms: list[Atom], start: float, end: float):
    for atom in atoms:
        overlap_start = max(start, atom.start)
        overlap_end = min(end, atom.end)
        if overlap_end > overlap_start:
            yield atom, overlap_start, overlap_end


def analyze_pair(
    session: str,
    track: str,
    wav: Path,
    ref_path: Path,
    hyp_path: Path,
    window_seconds: float,
) -> dict[str, list[dict] | dict]:
    info = sf.info(str(wav))
    duration = info.frames / info.samplerate
    ref_segments = read_rttm(ref_path, session, duration)
    hyp_segments = read_rttm(hyp_path, session, duration)
    raw = raw_atoms(ref_segments, hyp_segments, duration)
    mapping, cooccurrence, ref_duration, hyp_duration = global_mapping(raw)
    atoms = mapped_atoms(raw, mapping)
    overall = finish_components(aggregate_atoms(atoms, 0.0, duration))

    session_row = {
        "session": session,
        "track": track,
        "audio_duration_s": duration,
        "sample_rate": info.samplerate,
        "ref_speakers": len(ref_duration),
        "hyp_speakers": len(hyp_duration),
        "mapped_hyp_speakers": len(mapping),
        "unmapped_hyp_speakers": len(hyp_duration) - len(mapping),
        "ref_turns": len(ref_segments),
        "hyp_turns": len(hyp_segments),
        "ref_end_s": max((row.end for row in ref_segments), default=0.0),
        "hyp_end_s": max((row.end for row in hyp_segments), default=0.0),
        **overall,
        **pyannote_scores(ref_path, hyp_path, session, duration),
    }
    # The manual global-mapping decomposition must reproduce the baseline score.
    difference = abs(session_row["der_pct"] - session_row["der_collar0_overlap_pct"])
    if difference > 1e-6:
        raise RuntimeError(
            f"DER mismatch for {session}/{track}: manual={session_row['der_pct']} "
            f"pyannote={session_row['der_collar0_overlap_pct']}"
        )

    windows = []
    window_count = math.ceil(duration / window_seconds)
    for window_index in range(window_count):
        start = window_index * window_seconds
        end = min(duration, start + window_seconds)
        values = finish_components(aggregate_atoms(atoms, start, end))
        windows.append(
            {
                "session": session,
                "track": track,
                "window_index": window_index,
                "start_s": start,
                "end_s": end,
                **values,
            }
        )

    context_rows = []
    contexts = {
        "silence": lambda atom: atom.ref_count == 0,
        "single_speaker": lambda atom: atom.ref_count == 1,
        "overlap": lambda atom: atom.ref_count >= 2,
    }
    total_error = overall["error_s"]
    for name, predicate in contexts.items():
        values = blank_components()
        for atom in atoms:
            if predicate(atom):
                add_atom(values, atom, atom.duration)
        values = finish_components(values)
        context_rows.append(
            {
                "session": session,
                "track": track,
                "context": name,
                "error_contribution_pct": (
                    100.0 * values["error_s"] / total_error if total_error else None
                ),
                **values,
            }
        )

    mapping_rows = []
    for hyp_label in sorted(hyp_duration):
        mapped_ref = mapping.get(hyp_label, "")
        matched = cooccurrence.get((hyp_label, mapped_ref), 0.0) if mapped_ref else 0.0
        dominant_ref = ""
        dominant_overlap = 0.0
        for ref_label in ref_duration:
            overlap = cooccurrence.get((hyp_label, ref_label), 0.0)
            if overlap > dominant_overlap:
                dominant_ref, dominant_overlap = ref_label, overlap
        mapping_rows.append(
            {
                "session": session,
                "track": track,
                "hyp_speaker": hyp_label,
                "mapped_ref_speaker": mapped_ref,
                "dominant_ref_speaker": dominant_ref,
                "hyp_duration_s": hyp_duration[hyp_label],
                "mapped_cooccurrence_s": matched,
                "dominant_cooccurrence_s": dominant_overlap,
                "hyp_mapped_purity_pct": 100.0 * matched / hyp_duration[hyp_label],
                "hyp_dominant_purity_pct": 100.0 * dominant_overlap / hyp_duration[hyp_label],
                "silence_or_other_s": max(0.0, hyp_duration[hyp_label] - dominant_overlap),
            }
        )

    cooccurrence_rows = []
    for hyp_label in sorted(hyp_duration):
        for ref_label in sorted(ref_duration):
            overlap = cooccurrence.get((hyp_label, ref_label), 0.0)
            if overlap <= 0.0:
                continue
            cooccurrence_rows.append(
                {
                    "session": session,
                    "track": track,
                    "hyp_speaker": hyp_label,
                    "ref_speaker": ref_label,
                    "cooccurrence_s": overlap,
                    "pct_of_hyp_duration": 100.0 * overlap / hyp_duration[hyp_label],
                    "pct_of_ref_duration": 100.0 * overlap / ref_duration[ref_label],
                    "is_global_mapping": mapping.get(hyp_label) == ref_label,
                }
            )

    speaker_rows = []
    turn_counts = Counter(row.speaker for row in ref_segments)
    for ref_label in sorted(ref_duration):
        covered = 0.0
        overlap_time = 0.0
        partners = []
        mapped_clusters = []
        for atom in atoms:
            if atom.ref.get(ref_label, 0) <= 0:
                continue
            covered += atom.duration * min(
                atom.ref.get(ref_label, 0), atom.hyp_mapped.get(ref_label, 0)
            )
            overlap_time += atom.duration * (atom.ref_count > 1)
        for hyp_label in sorted(hyp_duration):
            amount = cooccurrence.get((hyp_label, ref_label), 0.0)
            if amount > 0:
                partners.append((amount, hyp_label))
            if mapping.get(hyp_label) == ref_label:
                mapped_clusters.append(hyp_label)
        partners.sort(reverse=True)
        top_amount, top_cluster = partners[0] if partners else (0.0, "")
        speaker_rows.append(
            {
                "session": session,
                "track": track,
                "ref_speaker": ref_label,
                "ref_turns": turn_counts[ref_label],
                "ref_duration_s": ref_duration[ref_label],
                "correctly_covered_s": covered,
                "target_coverage_pct": 100.0 * covered / ref_duration[ref_label],
                "overlap_wall_s": overlap_time,
                "overlap_share_pct": 100.0 * overlap_time / ref_duration[ref_label],
                "globally_mapped_hyp_clusters": "+".join(mapped_clusters),
                "top_cooccurring_hyp_cluster": top_cluster,
                "top_cooccurrence_s": top_amount,
                "top_cluster_coverage_pct": 100.0 * top_amount / ref_duration[ref_label],
                "cooccurring_hyp_clusters": len(partners),
                "cooccurring_hyp_clusters_over_1s": sum(x[0] >= 1.0 for x in partners),
            }
        )

    turn_rows = []
    for turn_index, segment in enumerate(
        sorted(ref_segments, key=lambda row: (row.start, row.end, row.speaker)), 1
    ):
        target_covered = 0.0
        target_uncovered_no_hyp = 0.0
        target_uncovered_hyp_active = 0.0
        target_overlap = 0.0
        values = blank_components()
        for atom, clipped_start, clipped_end in clip_atoms(atoms, segment.start, segment.end):
            length = clipped_end - clipped_start
            add_atom(values, atom, length)
            if atom.hyp_mapped.get(segment.speaker, 0) > 0:
                target_covered += length
            elif atom.hyp_count == 0:
                target_uncovered_no_hyp += length
            else:
                target_uncovered_hyp_active += length
            if atom.ref_count > 1:
                target_overlap += length
        values = finish_components(values)
        turn_rows.append(
            {
                "session": session,
                "track": track,
                "turn_index": turn_index,
                "rttm_line_no": segment.line_no,
                "ref_speaker": segment.speaker,
                "start_s": segment.start,
                "end_s": segment.end,
                "turn_duration_s": segment.end - segment.start,
                "target_covered_s": target_covered,
                "target_coverage_pct": 100.0 * target_covered / (segment.end - segment.start),
                "target_uncovered_no_hyp_s": target_uncovered_no_hyp,
                "target_uncovered_hyp_active_s": target_uncovered_hyp_active,
                "target_overlap_s": target_overlap,
                "target_overlap_pct": 100.0 * target_overlap / (segment.end - segment.start),
                "interval_ref_s": values["ref_s"],
                "interval_miss_s": values["miss_s"],
                "interval_fa_s": values["fa_s"],
                "interval_conf_s": values["conf_s"],
                "interval_der_pct": values["der_pct"],
            }
        )

    duration_buckets = [
        ("lt_0.5s", 0.0, 0.5),
        ("0.5_to_1s", 0.5, 1.0),
        ("1_to_2s", 1.0, 2.0),
        ("2_to_5s", 2.0, 5.0),
        ("ge_5s", 5.0, math.inf),
    ]
    bucket_rows = []
    for name, lower, upper in duration_buckets:
        selected = [
            row
            for row in turn_rows
            if lower <= row["turn_duration_s"] < upper
        ]
        total_duration = sum(row["turn_duration_s"] for row in selected)
        covered = sum(row["target_covered_s"] for row in selected)
        bucket_rows.append(
            {
                "session": session,
                "track": track,
                "duration_bucket": name,
                "turns": len(selected),
                "turn_duration_s": total_duration,
                "target_covered_s": covered,
                "target_coverage_pct": 100.0 * covered / total_duration if total_duration else None,
                "fully_missed_turns": sum(row["target_covered_s"] <= 1e-9 for row in selected),
                "low_coverage_turns_lt50pct": sum(row["target_coverage_pct"] < 50.0 for row in selected),
            }
        )

    error_regions = []
    previous = None
    for atom in atoms:
        if atom.miss_count + atom.fa_count + atom.conf_count == 0:
            previous = None
            continue
        identity = (
            tuple(sorted(atom.ref.elements())),
            tuple(sorted(atom.hyp_raw.elements())),
            tuple(sorted(atom.hyp_mapped.elements())),
            atom.miss_count,
            atom.fa_count,
            atom.conf_count,
        )
        if previous and previous["identity"] == identity and abs(previous["end_s"] - atom.start) < 1e-9:
            previous["end_s"] = atom.end
            previous["duration_s"] += atom.duration
            previous["ref_s"] += atom.duration * atom.ref_count
            previous["miss_s"] += atom.duration * atom.miss_count
            previous["fa_s"] += atom.duration * atom.fa_count
            previous["conf_s"] += atom.duration * atom.conf_count
            previous["error_s"] += atom.duration * (
                atom.miss_count + atom.fa_count + atom.conf_count
            )
            previous["local_der_pct"] = (
                100.0 * previous["error_s"] / previous["ref_s"]
                if previous["ref_s"]
                else None
            )
            continue
        row = {
            "session": session,
            "track": track,
            "start_s": atom.start,
            "end_s": atom.end,
            "duration_s": atom.duration,
            "ref_speakers_active": format_labels(atom.ref),
            "hyp_clusters_raw_active": format_labels(atom.hyp_raw),
            "hyp_speakers_mapped_active": format_labels(atom.hyp_mapped),
            "ref_s": atom.duration * atom.ref_count,
            "miss_s": atom.duration * atom.miss_count,
            "fa_s": atom.duration * atom.fa_count,
            "conf_s": atom.duration * atom.conf_count,
            "error_s": atom.duration * (
                atom.miss_count + atom.fa_count + atom.conf_count
            ),
            "local_der_pct": (
                100.0
                * (atom.miss_count + atom.fa_count + atom.conf_count)
                / atom.ref_count
                if atom.ref_count
                else None
            ),
            "identity": identity,
        }
        error_regions.append(row)
        previous = row
    for row in error_regions:
        del row["identity"]

    return {
        "session": session_row,
        "windows": windows,
        "contexts": context_rows,
        "mappings": mapping_rows,
        "cooccurrences": cooccurrence_rows,
        "speakers": speaker_rows,
        "turns": turn_rows,
        "duration_buckets": bucket_rows,
        "error_regions": error_regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/der_analysis"))
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--tracks", nargs="+", choices=["D1", "R1"], default=["D1", "R1"])
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_root = results_root / "prepare" / "dev"
    sessions = sorted(path.stem for path in ref_root.glob("*.rttm"))
    if not sessions:
        raise FileNotFoundError(f"No reference RTTMs under {ref_root}")

    tables: dict[str, list[dict]] = {
        "sessions": [],
        "time_windows": [],
        "contexts": [],
        "speaker_mappings": [],
        "speaker_cooccurrence": [],
        "reference_speakers": [],
        "reference_turns": [],
        "turn_duration_buckets": [],
        "error_regions": [],
    }
    for track in args.tracks:
        for session in sessions:
            print(f"Analyzing {session}/{track}", flush=True)
            result = analyze_pair(
                session=session,
                track=track,
                wav=select_wav(data_root / session, track),
                ref_path=ref_root / f"{session}.rttm",
                hyp_path=results_root / "infer" / "dev" / track / f"{session}.rttm",
                window_seconds=args.window_seconds,
            )
            tables["sessions"].append(result["session"])
            for source, destination in (
                ("windows", "time_windows"),
                ("contexts", "contexts"),
                ("mappings", "speaker_mappings"),
                ("cooccurrences", "speaker_cooccurrence"),
                ("speakers", "reference_speakers"),
                ("turns", "reference_turns"),
                ("duration_buckets", "turn_duration_buckets"),
                ("error_regions", "error_regions"),
            ):
                tables[destination].extend(result[source])

    for name, rows in tables.items():
        write_csv(output_dir / f"{name}.csv", rows)

    metadata = {
        "data_root": str(data_root),
        "results_root": str(results_root),
        "tracks": args.tracks,
        "sessions": sessions,
        "window_seconds": args.window_seconds,
        "mapping_policy": "one global Hungarian mapping per session/track, fixed for all drill-down tables",
        "scoring_policy": "zero collar, overlap included, full-recording UEM unless diagnostic column says otherwise",
        "files": {name: f"{name}.csv" for name in tables},
        "row_counts": {name: len(rows) for name, rows in tables.items()},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["row_counts"], ensure_ascii=False, indent=2))
    print(f"Detailed DER analysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
