#!/usr/bin/env python3
"""Build detailed minute DER reports for all dev sessions and one combined HTML."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import soundfile as sf

from analyze_der import global_mapping, mapped_atoms, raw_atoms, read_rttm, select_wav
from build_der_report_html import build_page
from build_minute_der_report import SOURCES, minute_rows, validate
from build_session_minute_der_report import enrich_rows, make_markdown, write_csv


REPORT_CSS = """
.report table { min-width: 2200px; }
.report th:first-child, .report td:first-child { min-width: 260px; white-space: nowrap; }
.report th:nth-child(3), .report td:nth-child(3) { min-width: 180px; }
.report th:nth-child(4), .report td:nth-child(4) { min-width: 260px; }
.report th:nth-child(n+7):nth-child(-n+11), .report td:nth-child(n+7):nth-child(-n+11) { min-width: 190px; }
.error-spans summary { cursor: pointer; color: #006d77; }
.error-spans summary small { display: block; color: #5c6a71; font-size: 11px; }
.span-list { margin-top: 8px; padding: 8px; border: 1px solid #dce5e8; border-radius: 8px;
             background: #f7fafb; white-space: nowrap; line-height: 1.8;
             font-variant-numeric: tabular-nums; }
.error-interval { display: block; padding: 1px 5px; border-left: 3px solid transparent; border-radius: 4px; }
.interval-over-1s { color: #6d4a00; background: #fff0bd; border-left-color: #c48600; font-weight: 600; }
.interval-over-2s { color: #8f2118; background: #ffd9d5; border-left-color: #b42318; font-weight: 800; }
.report tbody tr:has(td:first-child strong) td { background: #dff4f2; font-weight: 700; }
"""


def combined_markdown(by_session: dict[str, tuple[list[dict], list[dict]]]) -> str:
    lines = [
        "# EAR-1 开发集逐分钟 DER 与错误时间段",
        "",
        "开发集每个 session 单独成表；表内先列 D1 的全部分钟及合计，再列 R1。"
        "五类错误时间段默认展开，使用录音绝对时间（时:分:秒.毫秒）；"
        "区间左闭右开，并在分钟边界截断。",
        "",
        "时间段按**实际录音跨度**着色：超过 1 秒为浅黄色，超过 2 秒为浅红色。"
        "这两个阈值不按重叠说话人的错误秒数计算。",
        "",
        "参考与模型预测的时长都是**说话人秒数**；两人同时讲话一秒计 2 秒。"
        "每类错误的数字是其对该分钟 DER 的百分点贡献，计算方式为"
        "错误说话人秒数 ÷ 参考说话人秒数 × 100。"
        "时间段后的 ×2 表示每秒计 2 个说话人错误秒。",
        "",
        "预测编号后展示当前 session/轨道的全局说话人映射。"
        "编号出现只表示该分钟内出现过，不代表持续讲话。"
        "这里的“静音”只表示当前参考 TextGrid 没有标注说话人，"
        "不等于人工回听已确认无人讲话。"
        "某分钟参考说话人时长为 0 时 DER 无定义，显示“—”；误报秒数仍记录在 CSV。"
        "表格数字显示到两位小数，CSV 保留完整精度。",
        "",
        "**时间轴警示：** 20260923-1 的 R1 音频相对 D1 晚约 2.03 秒，"
        "而当前参考 TextGrid 与 D1 共用同一时间轴。"
        "下表及 CSV 保留未经平移的基线评分；"
        "这一 session 的 R1 46.92% DER 包含显著的时间轴不匹配影响。"
        "平移参考时间 +2.03 秒的诊断 DER 为 20.93%，不作为官方分数。"
        "七个 session 的估计偏移和诊断分数见"
        "[通道时间轴诊断 CSV](channel_alignment_diagnostics.csv)。",
        "",
        "[下载全部逐分钟 CSV](minute_error_sources_all.csv) · "
        "[下载全部错误时间段 CSV](minute_error_intervals_all.csv)",
        "",
    ]
    for session, (rows, intervals) in by_session.items():
        section_header = f"## {session}" + chr(10)
        single = make_markdown(rows, intervals, session)
        before, separator, section = single.partition(section_header)
        if not separator or not before:
            raise ValueError(f"Missing session section in generated Markdown: {session}")
        lines.extend([
            f"## {session}",
            "",
            f"[单独打开 {session} 报告](MINUTE_DER_REPORT_{session}.html)",
            "",
            section.rstrip(),
            "",
        ])
    return chr(10).join(lines)


def write_session_report(
    output_dir: Path,
    session: str,
    rows: list[dict],
    intervals: list[dict],
    baseline: dict[tuple[str, str], float],
) -> None:
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
            ("开发集合并报告", "MINUTE_DER_REPORT_ALL.html"),
        ],
        allow_html=True,
        extra_css=REPORT_CSS,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/der_analysis"))
    args = parser.parse_args()
    data_root, results_root, output_dir = (
        args.data_root.resolve(), args.results_root.resolve(), args.output_dir.resolve()
    )
    reference_root = results_root / "prepare" / "dev"
    sessions = sorted(path.stem for path in reference_root.glob("*.rttm"))
    if len(sessions) != 7:
        raise ValueError(f"Expected 7 development sessions, found {len(sessions)}")

    old_windows = {}
    with (output_dir / "time_windows.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            old_windows[row["session"], row["track"], int(row["window_index"])] = row
    baseline: dict[tuple[str, str], float] = {}
    overall_baseline: dict[str, float] = {}
    for track in ("D1", "R1"):
        with (results_root / "infer" / "dev" / track / "result_collar0.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                if row["session"] == "OVERALL":
                    overall_baseline[track] = float(row["der_pct"])
                else:
                    baseline[row["session"], track] = float(row["der_pct"])

    by_session: dict[str, tuple[list[dict], list[dict]]] = {}
    all_rows: list[dict] = []
    all_intervals: list[dict] = []
    for session in sessions:
        rows: list[dict] = []
        intervals: list[dict] = []
        for track in ("D1", "R1"):
            wav = select_wav(data_root / session, track)
            info = sf.info(str(wav))
            duration = info.frames / info.samplerate
            reference = read_rttm(reference_root / f"{session}.rttm", session, duration)
            prediction = read_rttm(
                results_root / "infer" / "dev" / track / f"{session}.rttm", session, duration
            )
            raw = raw_atoms(reference, prediction, duration)
            mapping, _, _, _ = global_mapping(raw)
            atoms = mapped_atoms(raw, mapping)
            pair_rows = minute_rows(atoms, mapping, duration, session, track)
            intervals.extend(enrich_rows(pair_rows, atoms, session, track))
            rows.extend(pair_rows)
        rows.sort(key=lambda row: (row["track"], row["minute_index"]))
        intervals.sort(key=lambda row: (
            row["track"], row["minute_index"], SOURCES.index(row["source"]), row["start_s"]
        ))
        by_session[session] = (rows, intervals)
        all_rows.extend(rows)
        all_intervals.extend(intervals)

    validate(all_rows, old_windows, baseline)
    for row in all_rows:
        key = row["session"], row["track"], row["minute_index"] - 1
        old = old_windows[key]
        if not math.isclose(row["hypothesis_speaker_s"], float(old["hyp_s"]), rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Predicted speaker duration mismatch at {key}")
    for track in ("D1", "R1"):
        pair_rows = [row for row in all_rows if row["track"] == track]
        reference = sum(row["reference_speaker_s"] for row in pair_rows)
        error = sum(row["error_s"] for row in pair_rows)
        actual = 100 * error / reference
        if not math.isclose(actual, overall_baseline[track], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Overall DER mismatch for {track}: {actual} != {overall_baseline[track]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for session, (rows, intervals) in by_session.items():
        write_session_report(output_dir, session, rows, intervals, baseline)

    csv_path = output_dir / "minute_error_sources_all.csv"
    interval_path = output_dir / "minute_error_intervals_all.csv"
    markdown_path = output_dir / "MINUTE_DER_REPORT_ALL.md"
    html_path = output_dir / "MINUTE_DER_REPORT_ALL.html"
    write_csv(csv_path, all_rows)
    write_csv(interval_path, all_intervals)
    markdown_path.write_text(combined_markdown(by_session), encoding="utf-8")
    build_page(
        markdown_path, html_path,
        title="EAR-1 开发集逐分钟 DER 与错误时间段",
        subtitle="7 个 session 合并展示；每个 session 先 D1 后 R1，包含五类错误的完整时间段。",
        metrics=[
            ("开发集 session", str(len(sessions))),
            ("分钟轨道窗口", str(len(all_rows))),
            ("D1 总体 DER", f"{overall_baseline['D1']:.2f}%"),
            ("R1 总体 DER", f"{overall_baseline['R1']:.2f}%"),
        ],
        downloads=[
            ("逐分钟 CSV", csv_path.name),
            ("错误时间段 CSV", interval_path.name),
            ("通道对齐诊断", "channel_alignment_diagnostics.csv"),
            ("总体分析报告", "DER_ANALYSIS_REPORT.html"),
        ],
        allow_html=True,
        extra_css=REPORT_CSS,
    )
    print(f"Validated {len(all_rows)} minute-track windows and {len(all_intervals)} intervals across {len(sessions)} sessions")
    print(f"Combined HTML: {html_path}")
    print(f"Minute CSV: {csv_path}")
    print(f"Interval CSV: {interval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
