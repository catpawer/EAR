#!/usr/bin/env python3
"""Build a standalone HTML version of the Chinese DER analysis report."""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path

from markdown_it import MarkdownIt


STYLE = r"""
:root {
  color-scheme: light;
  --bg: #f3f6f8;
  --surface: #ffffff;
  --surface-alt: #f7fafb;
  --text: #172126;
  --muted: #5c6a71;
  --line: #dce5e8;
  --brand: #006d77;
  --brand-dark: #004f57;
  --brand-soft: #dff4f2;
  --warn: #b54708;
  --danger: #b42318;
  --shadow: 0 16px 44px rgba(27, 44, 52, .09);
  --radius: 18px;
  --sidebar: 292px;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--text);
  background:
    radial-gradient(circle at 88% 0%, rgba(0, 109, 119, .10), transparent 27rem),
    var(--bg);
  font-family: Inter, "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  font-size: 16px;
  line-height: 1.78;
}

a { color: var(--brand); text-decoration-thickness: 1px; text-underline-offset: 3px; }
a:hover { color: var(--brand-dark); }
code {
  padding: .12rem .38rem;
  border-radius: 6px;
  color: #9b2c2c;
  background: #fff3f2;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: .88em;
}
pre {
  overflow: auto;
  padding: 1rem 1.1rem;
  border: 1px solid #25343a;
  border-radius: 12px;
  color: #e7f4f5;
  background: #142328;
  box-shadow: inset 0 1px rgba(255,255,255,.04);
}
pre code { padding: 0; color: inherit; background: none; }

.sidebar {
  position: fixed;
  inset: 0 auto 0 0;
  z-index: 20;
  width: var(--sidebar);
  padding: 28px 20px 24px;
  overflow-y: auto;
  color: #dceff0;
  background: linear-gradient(180deg, #0e3036 0%, #10262b 100%);
  border-right: 1px solid rgba(255,255,255,.08);
}
.brand {
  display: flex;
  gap: 12px;
  align-items: center;
  margin: 0 4px 24px;
}
.brand-mark {
  display: grid;
  width: 42px;
  height: 42px;
  flex: 0 0 42px;
  place-items: center;
  border-radius: 12px;
  color: #073a40;
  background: #82ddd6;
  font-weight: 900;
  letter-spacing: -.04em;
}
.brand strong { display: block; color: white; font-size: 15px; }
.brand span { display: block; color: #9ebbc0; font-size: 12px; }
.toc-title {
  margin: 0 8px 10px;
  color: #7fa4aa;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .14em;
  text-transform: uppercase;
}
.toc { margin: 0; padding: 0; list-style: none; }
.toc a {
  display: block;
  margin: 2px 0;
  padding: 8px 10px;
  border-left: 2px solid transparent;
  border-radius: 0 9px 9px 0;
  color: #bbd0d3;
  text-decoration: none;
  font-size: 13px;
  line-height: 1.35;
  transition: .18s ease;
}
.toc a:hover, .toc a.active {
  border-left-color: #82ddd6;
  color: white;
  background: rgba(130, 221, 214, .10);
}
.toc .level-3 a { padding-left: 22px; color: #94b0b4; font-size: 12px; }
.sidebar-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 22px 4px 0;
}
.sidebar button, .sidebar .download {
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 9px;
  padding: 8px 9px;
  color: #dceff0;
  background: rgba(255,255,255,.06);
  cursor: pointer;
  text-align: center;
  text-decoration: none;
  font: inherit;
  font-size: 12px;
}
.sidebar button:hover, .sidebar .download:hover { background: rgba(255,255,255,.11); color: white; }

.page { margin-left: var(--sidebar); min-width: 0; }
.hero {
  position: relative;
  overflow: hidden;
  padding: 64px clamp(24px, 5vw, 78px) 48px;
  color: white;
  background: linear-gradient(120deg, #005d66 0%, #087e84 58%, #289899 100%);
}
.hero::after {
  content: "";
  position: absolute;
  width: 420px;
  height: 420px;
  right: -130px;
  top: -235px;
  border: 72px solid rgba(255,255,255,.08);
  border-radius: 50%;
}
.hero-inner { position: relative; z-index: 1; max-width: 1120px; margin: auto; }
.eyebrow {
  margin: 0 0 12px;
  color: #bcece8;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: .18em;
  text-transform: uppercase;
}
.hero h1 {
  max-width: 860px;
  margin: 0;
  font-size: clamp(32px, 4.6vw, 60px);
  line-height: 1.12;
  letter-spacing: -.04em;
}
.hero-subtitle { max-width: 780px; margin: 18px 0 0; color: #d8f3f1; font-size: 17px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  max-width: 1120px;
  margin: 34px auto 0;
}
.metric {
  padding: 17px 18px;
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 14px;
  background: rgba(255,255,255,.10);
  backdrop-filter: blur(5px);
}
.metric span { display: block; color: #c6e8e6; font-size: 12px; }
.metric strong { display: block; margin-top: 2px; font-size: 27px; line-height: 1.15; }

.content-shell { max-width: 1180px; margin: auto; padding: 34px clamp(18px, 4vw, 56px) 80px; }
.quick-links {
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
  margin: 0 0 26px;
}
.quick-links a {
  padding: 7px 11px;
  border: 1px solid var(--line);
  border-radius: 999px;
  color: var(--brand-dark);
  background: var(--surface);
  box-shadow: 0 4px 14px rgba(26, 56, 62, .04);
  text-decoration: none;
  font-size: 12px;
  font-weight: 700;
}
.quick-links a:hover { border-color: #8fcac6; background: var(--brand-soft); }
.report {
  padding: clamp(24px, 4vw, 54px);
  border: 1px solid rgba(202, 216, 220, .9);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow);
}
.report > h1:first-child { display: none; }
.report h2 {
  margin: 3.3rem 0 1.15rem;
  padding-top: .4rem;
  color: #123c43;
  font-size: clamp(24px, 3vw, 32px);
  line-height: 1.3;
  letter-spacing: -.025em;
}
.report h2:first-of-type { margin-top: 0; }
.report h2::after {
  content: "";
  display: block;
  width: 48px;
  height: 4px;
  margin-top: 10px;
  border-radius: 2px;
  background: #57bdb7;
}
.report h3 { margin: 2.1rem 0 .8rem; color: #28545a; font-size: 21px; }
.report h4 { margin: 1.65rem 0 .55rem; color: #315c62; font-size: 17px; }
.report p { margin: .7rem 0; }
.report ul, .report ol { padding-left: 1.55rem; }
.report li { margin: .35rem 0; }
.report strong { color: #073f46; }
.report blockquote {
  margin: 1.2rem 0;
  padding: .8rem 1rem;
  border-left: 4px solid #62c9c2;
  color: var(--muted);
  background: var(--brand-soft);
}
.table-wrap {
  margin: 1.25rem 0 1.65rem;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: 0 4px 18px rgba(25, 50, 58, .04);
}
table { width: 100%; min-width: 650px; border-collapse: separate; border-spacing: 0; font-size: 13px; line-height: 1.45; }
thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 11px 12px;
  border-bottom: 1px solid #cddadd;
  color: #16474e;
  background: #e9f5f4;
  text-align: left;
  white-space: nowrap;
}
tbody td { padding: 9px 12px; border-bottom: 1px solid #e7edef; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f9fbfb; }
tbody tr:hover td { background: #eef8f7; }
tbody tr:last-child td { border-bottom: 0; }

.footer {
  max-width: 1120px;
  margin: 28px auto 0;
  color: #6b787d;
  text-align: center;
  font-size: 12px;
}
.back-top {
  position: fixed;
  right: 22px;
  bottom: 20px;
  z-index: 15;
  display: none;
  width: 42px;
  height: 42px;
  border: 0;
  border-radius: 50%;
  color: white;
  background: var(--brand);
  box-shadow: 0 8px 24px rgba(0, 80, 88, .25);
  cursor: pointer;
}
.back-top.visible { display: block; }
.mobile-nav { display: none; }

@media (max-width: 940px) {
  .sidebar { transform: translateX(-100%); transition: transform .22s ease; }
  .sidebar.open { transform: translateX(0); box-shadow: 14px 0 36px rgba(0,0,0,.22); }
  .page { margin-left: 0; }
  .mobile-nav {
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 30;
    display: block;
    width: 42px;
    height: 42px;
    border: 1px solid rgba(255,255,255,.32);
    border-radius: 11px;
    color: white;
    background: rgba(8, 50, 57, .78);
    backdrop-filter: blur(6px);
    cursor: pointer;
  }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 560px) {
  body { font-size: 15px; }
  .hero { padding-top: 78px; }
  .metrics { grid-template-columns: 1fr 1fr; }
  .metric { padding: 13px; }
  .metric strong { font-size: 22px; }
  .report { padding: 22px 17px; border-radius: 13px; }
}

@media print {
  @page { margin: 14mm; }
  body { background: white; font-size: 10.5pt; }
  .sidebar, .mobile-nav, .back-top, .quick-links, .sidebar-actions { display: none !important; }
  .page { margin: 0; }
  .hero { padding: 20px 0 24px; color: black; background: white; }
  .hero h1 { font-size: 28pt; }
  .hero-subtitle, .eyebrow { color: #455; }
  .metrics { margin-top: 16px; }
  .metric { border-color: #aaa; color: black; background: white; break-inside: avoid; }
  .metric span { color: #555; }
  .content-shell { max-width: none; padding: 0; }
  .report { padding: 0; border: 0; box-shadow: none; }
  .report h2 { break-after: avoid; }
  .table-wrap { overflow: visible; box-shadow: none; break-inside: avoid; }
  table { min-width: 0; font-size: 8pt; }
  thead th { position: static; }
  pre { white-space: pre-wrap; }
}
"""


SCRIPT = r"""
const sidebar = document.querySelector('.sidebar');
const menu = document.querySelector('.mobile-nav');
const backTop = document.querySelector('.back-top');
const navLinks = [...document.querySelectorAll('.toc a')];
const sections = navLinks.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);

menu?.addEventListener('click', () => sidebar.classList.toggle('open'));
navLinks.forEach(link => link.addEventListener('click', () => sidebar.classList.remove('open')));
document.querySelector('[data-print]')?.addEventListener('click', () => window.print());
backTop?.addEventListener('click', () => window.scrollTo({top: 0, behavior: 'smooth'}));
window.addEventListener('scroll', () => backTop.classList.toggle('visible', window.scrollY > 700), {passive: true});

const observer = new IntersectionObserver(entries => {
  const visible = entries.filter(e => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
  if (!visible) return;
  navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + visible.target.id));
}, {rootMargin: '-15% 0px -72% 0px', threshold: 0});
sections.forEach(section => observer.observe(section));
"""


def render_markdown(source: str, *, allow_html: bool = False) -> tuple[str, list[tuple[int, str, str]]]:
    md = MarkdownIt("commonmark", {"html": allow_html, "linkify": True, "typographer": True})
    md.enable("table")
    tokens = md.parse(source)
    headings: list[tuple[int, str, str]] = []
    number = 0
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        level = int(token.tag[1])
        title = tokens[index + 1].content
        anchor = "report-title" if level == 1 and not headings else f"section-{number}"
        if anchor != "report-title":
            number += 1
        token.attrSet("id", anchor)
        headings.append((level, title, anchor))
    body = md.renderer.render(tokens, md.options, {})
    body = body.replace("<table>", '<div class="table-wrap"><table>')
    body = body.replace("</table>", "</table></div>")
    return body, headings


def build_toc(headings: list[tuple[int, str, str]]) -> str:
    items = []
    for level, title, anchor in headings:
        if level not in (2, 3):
            continue
        items.append(
            f'<li class="level-{level}"><a href="#{html.escape(anchor)}">'
            f"{html.escape(title)}</a></li>"
        )
    return "\n".join(items)


def build_page(
    markdown_path: Path,
    output_path: Path,
    *,
    title: str = "EAR-1 DiariZen 基线 DER 详细分析报告",
    subtitle: str = "从总体 DER 下钻到 session、分钟窗口、参考说话人、逐条语音和连续错误区间。",
    metrics: list[tuple[str, str]] | None = None,
    downloads: list[tuple[str, str]] | None = None,
    extra_css: str = "",
    allow_html: bool = False,
) -> None:
    source = markdown_path.read_text(encoding="utf-8")
    body, headings = render_markdown(source, allow_html=allow_html)
    toc = build_toc(headings)
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    if metrics is None:
        metrics = [
            ("D1 DER", "31.22%"),
            ("R1 DER", "33.75%"),
            ("重叠区错误贡献", "42–46%"),
            ("参考语音段", "8,813"),
        ]
    if downloads is None:
        downloads = [
            ("Session 汇总", "sessions.csv"),
            ("分钟窗口", "time_windows.csv"),
            ("逐分钟详细报告", "MINUTE_DER_REPORT_ALL.html"),
            ("通道对齐诊断", "channel_alignment_diagnostics.csv"),
            ("逐语音段", "reference_turns.csv"),
            ("错误区间", "error_regions.csv"),
            ("说话人覆盖", "reference_speakers.csv"),
            ("音频诊断", "audio_diagnostics.csv"),
            ("原始 Markdown", markdown_path.name),
        ]
    quick_links = "\n".join(
        f'<a href="{html.escape(path)}" download>{html.escape(label)}</a>'
        for label, path in downloads
    )
    metric_cards = "\n".join(
        f'<div class="metric"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong></div>'
        for label, value in metrics
    )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="EAR-1 DiariZen 基线 DER 逐 session、时间窗口、说话人与错误区间分析报告">
  <title>{html.escape(title)}</title>
  <style>{STYLE}\n{extra_css}</style>
</head>
<body>
  <button class="mobile-nav" type="button" aria-label="打开目录">☰</button>
  <aside class="sidebar" aria-label="报告目录">
    <div class="brand">
      <div class="brand-mark">DER</div>
      <div><strong>EAR-1 分析报告</strong><span>DiariZen baseline</span></div>
    </div>
    <p class="toc-title">目录</p>
    <ol class="toc">{toc}</ol>
    <div class="sidebar-actions">
      <button type="button" data-print>打印 / PDF</button>
      <a class="download" href="{html.escape(markdown_path.name)}" download>下载 MD</a>
    </div>
  </aside>

  <main class="page">
    <header class="hero">
      <div class="hero-inner">
        <p class="eyebrow">EAR-1 Challenge · Detailed Error Analysis</p>
        <h1>{html.escape(title)}</h1>
        <p class="hero-subtitle">{html.escape(subtitle)}</p>
        <section class="metrics" aria-label="关键指标">
          {metric_cards}
        </section>
      </div>
    </header>

    <div class="content-shell">
      <nav class="quick-links" aria-label="分析数据下载">{quick_links}</nav>
      <article class="report">{body}</article>
      <footer class="footer">由 DER 分析产物生成 · {html.escape(generated)} · 单文件离线 HTML</footer>
    </div>
  </main>
  <button class="back-top" type="button" aria-label="返回顶部">↑</button>
  <script>{SCRIPT}</script>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_page(args.markdown.resolve(), args.output.resolve())
    print(f"HTML report written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
