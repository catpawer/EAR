#!/usr/bin/env python3
"""Build a minute DER report with complete error intervals for one session."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from analyze_der import global_mapping, mapped_atoms, raw_atoms, read_rttm, select_wav
from build_der_report_html import build_page
from build_minute_der_report import (
    LABELS,
    SOURCES,
    aggregate_pair,
    fmt_contribution,
    fmt_percent,
    minute_rows,
    validate,
)


def clock_time(seconds: float) -> str:
    hours, remaining = divmod(seconds, 3600)
    minutes, remaining = divmod(remaining, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{remaining:06.3f}"


def source_counts(atom) -> dict[str, int]:
    return {
        "single_speaker_miss": atom.miss_count if atom.ref_count == 1 else 0,
        "overlap_miss": atom.miss_count if atom.ref_count > 1 else 0,
        "silence_false_alarm": atom.fa_count if atom.ref_count == 0 else 0,
        "speech_extra_speaker": atom.fa_count if atom.ref_count > 0 else 0,
        "speaker_confusion": atom.conf_count,
    }


def enrich_rows(rows: list[dict], atoms, session: str, track: str) -> list[dict]:
    """Measure predicted speaker time and merge adjacent equal-count error spans."""
    intervals: list[dict] = []
    for row in rows:
        start, end = row["start_s"], row["end_s"]
        row["hypothesis_speaker_s"] = 0.0
        by_source: dict[str, list[dict]] = {name: [] for name in SOURCES}
        for atom in atoms:
            left, right = max(start, atom.start), min(end, atom.end)
            if right <= left:
                continue
            row["hypothesis_speaker_s"] += (right - left) * atom.hyp_count
            for source, count in source_counts(atom).items():
                if count == 0:
                    continue
                prior = by_source[source][-1] if by_source[source] else None
                if prior and prior["speaker_error_count"] == count and math.isclose(
                    prior["end_s"], left, rel_tol=0, abs_tol=1e-8
                ):
                    prior["end_s"] = right
                    prior["error_s"] += (right - left) * count
                else:
                    by_source[source].append({
                        "session": session,
                        "track": track,
                        "minute_index": row["minute_index"],
                        "source": source,
                        "start_s": left,
                        "end_s": right,
                        "speaker_error_count": count,
                        "error_s": (right - left) * count,
                    })
        for source in SOURCES:
            spans = by_source[source]
            actual = sum(span["error_s"] for span in spans)
            expected = row[f"{source}_s"]
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Interval total mismatch: {session}/{track}/{row['minute_index']}/{source}")
            row[f"{source}_interval_count"] = len(spans)
            row[f"{source}_intervals"] = "; ".join(
                f"{clock_time(span['start_s'])}–{clock_time(span['end_s'])}"
                + (f" ×{span['speaker_error_count']}" if span["speaker_error_count"] > 1 else "")
                for span in spans
            )
            for span in spans:
                span["source_zh"] = LABELS[SOURCES.index(source)]
                span["start_time"] = clock_time(span["start_s"])
                span["end_time"] = clock_time(span["end_s"])
                span["wall_s"] = span["end_s"] - span["start_s"]
            intervals.extend(spans)
    return intervals


def error_cell(row: dict, source: str, spans: list[dict]) -> str:
    value = fmt_contribution(row[f"{source}_pp"])
    count = len(spans)
    if not count:
        return value
    rendered = []
    for span in spans:
        wall_seconds = span["wall_s"]
        level = " interval-over-2s" if wall_seconds > 2 else (
            " interval-over-1s" if wall_seconds > 1 else ""
        )
        label = (
            f"{span['start_time']}–{span['end_time']}"
            + (f" ×{span['speaker_error_count']}" if span["speaker_error_count"] > 1 else "")
        )
        rendered.append(f'<span class="error-interval{level}">{html.escape(label)}</span>')
    list_html = "".join(rendered)
    total = row[f"{source}_s"]
    return (
        '<details class="error-spans" open>'
        f'<summary><strong>{value}</strong><small>{count} 段 · {total:.3f} s</small></summary>'
        f'<div class="span-list">{list_html}</div></details>'
    )


def make_markdown(rows: list[dict], intervals: list[dict], session: str) -> str:
    by_pair = {(row["minute_index"], row["track"]): row for row in rows}
    by_source: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for span in intervals:
        by_source[span["minute_index"], span["track"], span["source"]].append(span)
    minute_count = max(row["minute_index"] for row in rows)
    lines = [
        f"# EAR-1 {session} 逐分钟 DER 与错误时间段",
        "",
        "每个一分钟窗口分别列出 D1、R1。五类错误单元格默认展开该分钟所有错误时间段；"
        "时间采用录音绝对时间 `时:分:秒.毫秒`，区间左闭右开，并在分钟边界截断。",
        "",
        "时间段按**实际录音跨度**着色：超过 1 秒标为浅黄色，超过 2 秒标为浅红色。"
        "阈值不按重叠说话人的错误秒数计算。",
        "",
        "参考、预测说话人时长均为**说话人秒数**：同一秒有两人同时讲话时计 2 秒。"
        "五类错误的数字为该分钟 DER 的百分点贡献，即错误说话人秒数 ÷ 参考说话人秒数 × 100。"
        "展开标题中的秒数同样按出错人数累加；时间段后的 `×2` 表示该区间每秒计 2 个说话人错误秒。"
        "区间本身则表示连续的实际录音时间。",
        "",
        "本报告沿用原始、未经时间平移的基线评分。"
        "R1 可能与参考标注存在时轴偏移，见"
        "[通道时间轴诊断 CSV](channel_alignment_diagnostics.csv)。",
        "",
        "说话人编号列显示该分钟出现过的参考编号及预测聚类到参考编号的全局映射。"
        "同一套映射用于整个 session 和轨道，编号出现不代表持续讲话。"
        "这里的“静音”只表示当前参考 TextGrid 没有标注说话人，"
        "不等于人工回听已确认无人讲话。"
        "没有参考说话人时 DER 无定义，显示“—”；误报仍保留在数据文件中。"
        "表格数值四舍五入到两位小数。",
        "",
        f"完整精度与时间列表见 [逐分钟 CSV](minute_error_sources_{session}.csv)，"
        f"逐条连续时间段及人数见 [错误时间段 CSV](minute_error_intervals_{session}.csv)。",
        "",
        f"## {session}",
        "",
    ]
    if session == "20260923-1":
        lines.extend([
            "**R1 时间轴对齐警示：** 当前数据副本的 R1 音频相对 D1 晚约 2.02–2.04 秒，"
            "但两份 TextGrid 使用相同语音时间戳。"
            "因此 R1 的 00:21:28.450–00:21:30.493 虽被当前参考标注计为"
            "“静音时误报”，其音频内容很可能对应参考中"
            "约 00:21:26.420–00:21:28.463 的语音，与 P029 的"
            "00:21:24.470–00:21:28.450 语音段几乎重合。"
            "仅将 R1 参考时间整体后移 2.03 秒作诊断，"
            "该 session 的 R1 DER 从 46.92% 降至 20.93%。"
            "本报告仍按未经平移的参考标注显示原始基线分数；"
            "这类 R1 错误不应直接归因于模型。"
            "详细数值见[通道时间轴诊断 CSV](channel_alignment_diagnostics.csv)。",
            "",
        ])
    lines.extend([
        "| 分钟（起止） | 轨道 | 参考说话人编号 | 预测编号→参考编号 | 参考说话人时长 | 模型预测说话人时长 | "
        + " | ".join(f"{label}（百分点＋时间段）" for label in LABELS)
        + " | 整体 DER |",
        "|---|:---:|---|---|---:|---:|" + "---:|" * 6,
    ])
    for track in ("D1", "R1"):
        for index in range(1, minute_count + 1):
            row = by_pair.get((index, track))
            if row is None:
                continue
            window = f"第 {index} 分钟 {clock_time(row['start_s'])}–{clock_time(row['end_s'])}"
            cells = " | ".join(
                error_cell(row, source, by_source[index, track, source]) for source in SOURCES
            )
            lines.append(
                f"| {window} | {track} | {row['reference_speaker_ids'] or '—'} | "
                f"{row['hyp_speaker_global_mapping'] or '—'} | "
                f"{row['reference_speaker_s']:.3f} s | {row['hypothesis_speaker_s']:.3f} s | "
                f"{cells} | **{fmt_percent(row['overall_der_pct'])}** |"
            )
        pair_rows = [row for row in rows if row["track"] == track]
        total = aggregate_pair(pair_rows)
        ref_ids = sorted({name.strip() for row in pair_rows for name in row["reference_speaker_ids"].split(",") if name.strip()})
        mappings = sorted({name.strip() for row in pair_rows for name in row["hyp_speaker_global_mapping"].split(",") if name.strip()})
        contributions = " | ".join(fmt_contribution(total[f"{source}_pp"]) for source in SOURCES)
        lines.append(
            f"| **本 session 合计** | **{track}** | {', '.join(ref_ids)} | "
            f"{', '.join(mappings)} | **{total['reference_speaker_s']:.3f} s** | "
            f"**{sum(row['hypothesis_speaker_s'] for row in pair_rows):.3f} s** | "
            f"{contributions} | **{fmt_percent(total['overall_der_pct'])}** |"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="20260923-1")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/der_analysis"))
    args = parser.parse_args()
    session = args.session
    results_root, output_dir = args.results_root.resolve(), args.output_dir.resolve()
    reference_path = results_root / "prepare" / "dev" / f"{session}.rttm"
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    old_windows = {}
    with (output_dir / "time_windows.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["session"] == session:
                old_windows[session, row["track"], int(row["window_index"])] = row
    baseline = {}
    for track in ("D1", "R1"):
        with (results_root / "infer" / "dev" / track / "result_collar0.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            baseline[session, track] = next(
                float(row["der_pct"]) for row in csv.DictReader(stream) if row["session"] == session
            )

    rows, intervals = [], []
    for track in ("D1", "R1"):
        wav = select_wav(args.data_root.resolve() / session, track)
        info = sf.info(str(wav))
        duration = info.frames / info.samplerate
        reference = read_rttm(reference_path, session, duration)
        prediction = read_rttm(results_root / "infer" / "dev" / track / f"{session}.rttm", session, duration)
        raw = raw_atoms(reference, prediction, duration)
        mapping, _, _, _ = global_mapping(raw)
        atoms = mapped_atoms(raw, mapping)
        pair_rows = minute_rows(atoms, mapping, duration, session, track)
        intervals.extend(enrich_rows(pair_rows, atoms, session, track))
        rows.extend(pair_rows)
    rows.sort(key=lambda row: (row["track"], row["minute_index"]))
    intervals.sort(key=lambda item: (item["track"], item["minute_index"], SOURCES.index(item["source"]), item["start_s"]))
    validate(rows, old_windows, baseline)
    for row in rows:
        old = old_windows[session, row["track"], row["minute_index"] - 1]
        if not math.isclose(row["hypothesis_speaker_s"], float(old["hyp_s"]), rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Predicted speaker duration mismatch: {session}/{row['track']}/{row['minute_index']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"minute_error_sources_{session}.csv"
    interval_path = output_dir / f"minute_error_intervals_{session}.csv"
    markdown_path = output_dir / f"MINUTE_DER_REPORT_{session}.md"
    html_path = output_dir / f"MINUTE_DER_REPORT_{session}.html"
    write_csv(csv_path, rows)
    write_csv(interval_path, intervals)
    markdown_path.write_text(make_markdown(rows, intervals, session), encoding="utf-8")
    build_page(
        markdown_path, html_path,
        title=f"EAR-1 {session} 逐分钟 DER 与错误时间段",
        subtitle="D1 / R1 对照：参考与预测说话人时长，以及五类错误发生的完整时间段。",
        metrics=[
            ("Session", session),
            ("分钟轨道窗口", str(len(rows))),
            ("D1 DER", f"{baseline[session, 'D1']:.2f}%"),
            ("R1 DER", f"{baseline[session, 'R1']:.2f}%"),
        ],
        downloads=[
            ("逐分钟 CSV", csv_path.name),
            ("错误时间段 CSV", interval_path.name),
            ("七个 session 报告", "MINUTE_DER_REPORT.html"),
        ],
        allow_html=True,
        extra_css="""
        .report table { min-width: 2200px; }
        .report th:first-child, .report td:first-child { min-width: 260px; white-space: nowrap; }
        .report th:nth-child(3), .report td:nth-child(3) { min-width: 180px; }
        .report th:nth-child(4), .report td:nth-child(4) { min-width: 260px; }
        .report th:nth-child(n+7):nth-child(-n+11),
        .report td:nth-child(n+7):nth-child(-n+11) { min-width: 190px; }
        .error-spans summary { cursor: pointer; color: #006d77; }
        .error-spans summary small { display: block; color: #5c6a71; font-size: 11px; }
        .span-list { margin-top: 8px; padding: 8px;
                     border: 1px solid #dce5e8; border-radius: 8px; background: #f7fafb;
                     white-space: nowrap; line-height: 1.8; font-variant-numeric: tabular-nums; }
        .error-interval { display: block; padding: 1px 5px; border-left: 3px solid transparent; border-radius: 4px; }
        .interval-over-1s { color: #6d4a00; background: #fff0bd; border-left-color: #c48600; font-weight: 600; }
        .interval-over-2s { color: #8f2118; background: #ffd9d5; border-left-color: #b42318; font-weight: 800; }
        .report tbody tr:has(td:first-child strong) td { background: #dff4f2; font-weight: 700; }
        """,
    )
    print(f"Validated {len(rows)} minute-track windows and {len(intervals)} intervals for {session}")
    print(f"HTML: {html_path}\nMinute CSV: {csv_path}\nInterval CSV: {interval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
