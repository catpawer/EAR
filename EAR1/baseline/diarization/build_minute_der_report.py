#!/usr/bin/env python3
"""Export five additive DER sources for every dev session's D1/R1 minute."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from analyze_der import global_mapping, mapped_atoms, raw_atoms, read_rttm, select_wav
from build_der_report_html import build_page


SOURCES = (
    "single_speaker_miss",
    "overlap_miss",
    "silence_false_alarm",
    "speech_extra_speaker",
    "speaker_confusion",
)
LABELS = (
    "单人讲话漏检",
    "多人重叠讲话漏检",
    "静音时误报有人讲话",
    "有语音时多报说话人数",
    "说话人身份混淆",
)


def minute_rows(atoms, mapping: dict[str, str], duration: float, session: str, track: str) -> list[dict]:
    rows = []
    for index in range(math.ceil(duration / 60)):
        start, end = index * 60.0, min((index + 1) * 60.0, duration)
        source_seconds = {name: 0.0 for name in SOURCES}
        ref_seconds = 0.0
        ref_speakers: set[str] = set()
        hyp_speakers: set[str] = set()
        for atom in atoms:
            clipped = min(end, atom.end) - max(start, atom.start)
            if clipped <= 0:
                continue
            ref_speakers.update(atom.ref)
            hyp_speakers.update(atom.hyp_raw)
            ref_seconds += clipped * atom.ref_count
            if atom.ref_count == 1:
                source_seconds["single_speaker_miss"] += clipped * atom.miss_count
            elif atom.ref_count > 1:
                source_seconds["overlap_miss"] += clipped * atom.miss_count
            if atom.ref_count == 0:
                source_seconds["silence_false_alarm"] += clipped * atom.fa_count
            else:
                source_seconds["speech_extra_speaker"] += clipped * atom.fa_count
            source_seconds["speaker_confusion"] += clipped * atom.conf_count
        error_seconds = sum(source_seconds.values())
        row = {
            "session": session,
            "track": track,
            "minute_index": index + 1,
            "start_s": start,
            "end_s": end,
            "reference_speaker_s": ref_seconds,
            "reference_speaker_ids": ", ".join(sorted(ref_speakers)),
            "hyp_speaker_ids": ", ".join(sorted(hyp_speakers)),
            "hyp_speaker_global_mapping": ", ".join(
                f"{speaker}→{mapping.get(speaker, '未匹配')}"
                for speaker in sorted(hyp_speakers)
            ),
            **{f"{name}_s": value for name, value in source_seconds.items()},
            "error_s": error_seconds,
            **{
                f"{name}_pp": 100 * source_seconds[name] / ref_seconds
                if ref_seconds > 0 else None
                for name in SOURCES
            },
            "overall_der_pct": 100 * error_seconds / ref_seconds
            if ref_seconds > 0 else None,
        }
        rows.append(row)
    return rows


def validate(rows: list[dict], old_windows: dict, baseline: dict) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["session"], row["track"], row["minute_index"] - 1)
        old = old_windows[key]
        for field, expected in (
            ("reference_speaker_s", float(old["ref_s"])),
            ("error_s", float(old["error_s"])),
            (
                "miss_s",
                float(old["miss_s"]),
            ),
            (
                "fa_s",
                float(old["fa_s"]),
            ),
            ("speaker_confusion_s", float(old["conf_s"])),
        ):
            actual = row[field] if field in row else (
                row["single_speaker_miss_s"] + row["overlap_miss_s"]
                if field == "miss_s" else
                row["silence_false_alarm_s"] + row["speech_extra_speaker_s"]
            )
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Minute mismatch at {key}/{field}: {actual} != {expected}")
        if row["reference_speaker_s"] > 0:
            total_pp = sum(row[f"{name}_pp"] for name in SOURCES)
            if not math.isclose(total_pp, row["overall_der_pct"], abs_tol=1e-8):
                raise ValueError(f"Contributions do not sum to DER at {key}")
        grouped[row["session"], row["track"]].append(row)

    for key, pair_rows in grouped.items():
        reference = sum(row["reference_speaker_s"] for row in pair_rows)
        error = sum(row["error_s"] for row in pair_rows)
        expected = baseline[key]
        if not math.isclose(100 * error / reference, expected, abs_tol=1e-6):
            raise ValueError(f"Session mismatch at {key}: {100 * error / reference} != {expected}")


def aggregate_pair(pair_rows: list[dict]) -> dict:
    ref = sum(row["reference_speaker_s"] for row in pair_rows)
    summed = {name: sum(row[f"{name}_s"] for row in pair_rows) for name in SOURCES}
    return {
        "reference_speaker_s": ref,
        **{f"{name}_pp": 100 * summed[name] / ref for name in SOURCES},
        "overall_der_pct": 100 * sum(summed.values()) / ref,
    }


def fmt_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}%"


def fmt_contribution(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def fmt_time(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{remainder:05.2f}"


def markdown_report(rows: list[dict], sessions: list[str]) -> str:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["session"]][row["minute_index"], row["track"]] = row

    lines = [
        "# EAR-1 开发集逐分钟 DER 错误来源统计",
        "",
        "每个一分钟窗口分别列出 D1 和 R1 两行。每项数字都是对该分钟整体 DER 的贡献，"
        "因此五项相加等于最后一列的整体 DER。",
        "",
        "计算口径：每项错误秒数 ÷ **该分钟参考说话人总时长** × 100%。"
        "多人同时讲话时，参考时长按说话人数累加，可能超过一分钟。"
        "窗口采用各自音频的 `[0,60)、[60,120)…`，最后不足一分钟的窗口只计实际音频。"
        "全程使用该 session/轨道的**同一套全局说话人映射**，与基线的零容忍边界、"
        "保留重叠、整段录音评分设置一致。",
        "",
        "**R1 时轴提示：** 20260923-1 的 R1 音频相对 D1 晚约 2.03 秒，"
        "当前 TextGrid 未为 R1 补偿该时间差。本表保留原始、未经平移的基线分数；"
        "R1 的部分漏检和误报可能是时轴不匹配造成。"
        "详见[通道时间轴诊断 CSV](channel_alignment_diagnostics.csv)和"
        "[含时间段的合并报告](MINUTE_DER_REPORT_ALL.html)。",
        "",
        "如果某分钟参考说话人时长为 0，则该分钟 DER 没有定义，表内显示“—”；"
        "其中的误报秒数仍记录在 CSV。表格数字独立四舍五入到小数点后两位，"
        "显示值相加可能相差 0.01 个百分点。",
        "",
        "“参考说话人编号”列列出该分钟与音频有交集的全部参考说话人；预测编号列以"
        "`原始聚类编号→全局匹配的参考编号`展示该分钟出现的预测说话人。"
        "“未匹配”表示该预测聚类没有一一匹配到参考说话人。"
        "原始聚类编号只在当前 session/轨道内有意义。编号仅表示该分钟出现过，"
        "不代表该说话人在整分钟持续讲话。",
        "",
        "完整精度及五类错误秒数见 [逐分钟明细 CSV](minute_error_sources.csv)。",
        "",
    ]
    header = "| 分钟（起止） | 轨道 | 参考说话人编号 | 预测编号→参考编号 | 参考说话人时长 | " + " | ".join(
        f"{name}（百分点）" for name in LABELS
    ) + " | 整体 DER |"
    separator = "|---|:---:|---|---|---:|" + "---:|" * 6
    for session in sessions:
        lines.extend([f"## {session}", "", header, separator])
        by_minute = grouped[session]
        minutes = max(index for index, _ in by_minute)
        for index in range(1, minutes + 1):
            for track in ("D1", "R1"):
                row = by_minute.get((index, track))
                if row is None:
                    continue
                time = f"第 {index} 分钟 {fmt_time(row['start_s'])}–{fmt_time(row['end_s'])}"
                contributions = " | ".join(fmt_contribution(row[f"{name}_pp"]) for name in SOURCES)
                lines.append(
                    f"| {time} | {track} | {row['reference_speaker_ids'] or '—'} | "
                    f"{row['hyp_speaker_global_mapping'] or '—'} | {row['reference_speaker_s']:.2f} s | "
                    f"{contributions} | **{fmt_percent(row['overall_der_pct'])}** |"
                )
        for track in ("D1", "R1"):
            pair_rows = [r for r in by_minute.values() if r["track"] == track]
            total = aggregate_pair(pair_rows)
            contributions = " | ".join(fmt_contribution(total[f"{name}_pp"]) for name in SOURCES)
            ref_ids = sorted({speaker.strip() for r in pair_rows for speaker in r["reference_speaker_ids"].split(",") if speaker.strip()})
            hyp_mapping = sorted({item.strip() for r in pair_rows for item in r["hyp_speaker_global_mapping"].split(",") if item.strip()})
            lines.append(
                f"| **本 session 合计** | **{track}** | {', '.join(ref_ids)} | "
                f"{', '.join(hyp_mapping)} | **{total['reference_speaker_s']:.2f} s** | "
                f"{contributions} | **{fmt_percent(total['overall_der_pct'])}** |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/der_analysis"))
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    ref_root = results_root / "prepare" / "dev"
    sessions = sorted(path.stem for path in ref_root.glob("*.rttm"))
    if len(sessions) != 7:
        raise ValueError(f"Expected 7 development sessions, found {len(sessions)}")

    old_windows = {}
    with (output_dir / "time_windows.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            old_windows[row["session"], row["track"], int(row["window_index"])] = row
    baseline = {}
    for track in ("D1", "R1"):
        with (results_root / "infer" / "dev" / track / "result_collar0.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                if row["session"] != "OVERALL":
                    baseline[row["session"], track] = float(row["der_pct"])

    rows = []
    for session in sessions:
        for track in ("D1", "R1"):
            wav = select_wav(data_root / session, track)
            info = sf.info(str(wav))
            duration = info.frames / info.samplerate
            reference = read_rttm(ref_root / f"{session}.rttm", session, duration)
            hypothesis = read_rttm(
                results_root / "infer" / "dev" / track / f"{session}.rttm",
                session, duration,
            )
            raw = raw_atoms(reference, hypothesis, duration)
            mapping, _, _, _ = global_mapping(raw)
            rows.extend(minute_rows(mapped_atoms(raw, mapping), mapping, duration, session, track))
    validate(rows, old_windows, baseline)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "minute_error_sources.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_dir / "MINUTE_DER_REPORT.md"
    markdown_path.write_text(markdown_report(rows, sessions), encoding="utf-8")
    html_path = output_dir / "MINUTE_DER_REPORT.html"
    build_page(
        markdown_path, html_path,
        title="EAR-1 逐分钟 DER 错误来源统计",
        subtitle="7 个 session，每分钟同时对照 D1 和 R1 的五类错误贡献。",
        metrics=[
            ("开发集 session", str(len(sessions))),
            ("一分钟轨道窗口", str(len(rows))),
            ("D1 总体 DER", "31.22%"),
            ("R1 总体 DER", "33.75%"),
        ],
        downloads=[
            ("下载逐分钟 CSV", csv_path.name),
            ("下载 Markdown", markdown_path.name),
            ("总体分析报告", "DER_ANALYSIS_REPORT.html"),
        ],
        extra_css="""
        .report table { min-width: 1640px; }
        .report th:first-child, .report td:first-child { white-space: nowrap; }
        .report th:nth-child(3), .report td:nth-child(3) { min-width: 145px; }
        .report th:nth-child(4), .report td:nth-child(4) { min-width: 220px; }
        .report tbody tr:has(td:first-child strong) td { background: #dff4f2; font-weight: 700; }
        """,
    )
    print(f"Validated {len(rows)} windows across {len(sessions)} sessions")
    print(f"CSV: {csv_path}\nMarkdown: {markdown_path}\nHTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
