#!/usr/bin/env python3
"""Export the Chinese Markdown comparison report to a formatted editable DOCX."""
import re
import tempfile
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'EAR1_TASK1_MODEL_COMPARISON.md'
DEST = ROOT / 'EAR1_TASK1_MODEL_COMPARISON.docx'


def set_font(style, size, bold=False, color=None):
    style.font.name = 'Noto Sans CJK SC'
    style.font.size = Pt(size)
    style.font.bold = bold
    if color:
        style.font.color.rgb = RGBColor(*color)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.insert(0, rfonts)
    rfonts.set(qn('w:eastAsia'), 'Noto Sans CJK SC')


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = tcpr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tcpr.append(shd)
    shd.set(qn('w:fill'), fill)


def repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    tag = OxmlElement('w:tblHeader')
    tag.set(qn('w:val'), 'true')
    trpr.append(tag)


with tempfile.TemporaryDirectory() as tempdir:
    raw = Path(tempdir) / 'raw.docx'
    pypandoc.convert_file(str(SOURCE), to='docx', format='gfm', outputfile=str(raw),
                          extra_args=['--standalone', '--resource-path', str(ROOT)])
    doc = Document(raw)

section = doc.sections[0]
section.page_width, section.page_height = Mm(297), Mm(210)  # A4 landscape
section.left_margin = section.right_margin = Mm(12)
section.top_margin = Mm(12)
section.bottom_margin = Mm(10)
section.header_distance = Mm(6)
section.footer_distance = Mm(6)

normal = doc.styles['Normal']
set_font(normal, 9.5)
normal.paragraph_format.line_spacing = 1.15
normal.paragraph_format.space_after = Pt(4)
for name, size in [('Heading 1', 17), ('Heading 2', 13), ('Heading 3', 10.5)]:
    style = doc.styles[name]
    set_font(style, size, bold=True, color=(27, 64, 110))
    style.paragraph_format.space_before = Pt(12 if name != 'Heading 3' else 8)
    style.paragraph_format.space_after = Pt(5)
    style.paragraph_format.keep_with_next = True

if 'Table Grid' not in doc.styles:
    doc.styles.add_style('Table Grid', WD_STYLE_TYPE.TABLE)
for index, table in enumerate(doc.tables):
    table.style = 'Table Grid'
    table.autofit = False
    borders = OxmlElement('w:tblBorders')
    for edge in ('top','left','bottom','right','insideH','insideV'):
        tag = OxmlElement(f'w:{edge}')
        tag.set(qn('w:val'), 'single')
        tag.set(qn('w:sz'), '4')
        tag.set(qn('w:color'), 'D5DDE7')
        borders.append(tag)
    table._tbl.tblPr.append(borders)
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    n = len(headers)
    if n == 10 and '参考人数' in ''.join(headers):
        widths = [31, 23, 12, 37, 37, 36, 17, 25, 27, 27]
    elif n == 10:
        widths = [31, 23, 12, 20, 24, 27, 32, 32, 32, 29]
    elif n == 9:
        widths = [46, 14, 18, 30, 28, 28, 28, 27, 48]
    elif n == 7:
        widths = [16, 46, 24, 34, 34, 34, 47]
    elif n == 4:
        widths = [43, 63, 32, 135]
    else:
        widths = [273 / n] * n
    assert len(widths) == n and sum(widths) <= 273, (index, headers, widths)
    for column, width in zip(table.columns, widths):
        column.width = Mm(width)
    repeat_header(table.rows[0])
    for ri, row in enumerate(table.rows):
        for ci, (cell, width) in enumerate(zip(row.cells, widths)):
            cell.width = Mm(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ri == 0:
                shade(cell, 'E7EEF7')
            elif ri % 2 == 0:
                shade(cell, 'F7F9FC')
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                if ri == 0:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif re.fullmatch(r'[+−\-]?\d+(?:\.\d+)?(?:%| pp)?|—', cell.text.strip()):
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                for run in paragraph.runs:
                    run.font.name = 'Noto Sans CJK SC'
                    run.font.size = Pt(7 if n >= 9 else 8)
                    if ri == 0:
                        run.bold = True
                    rpr = run._r.get_or_add_rPr()
                    rfonts = rpr.rFonts
                    if rfonts is None:
                        rfonts = OxmlElement('w:rFonts')
                        rpr.insert(0, rfonts)
                    rfonts.set(qn('w:eastAsia'), 'Noto Sans CJK SC')

footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('EAR-1 任务一模型对比  ·  ')
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)

doc.core_properties.title = 'EAR-1 任务一模型对比实验报告'
doc.core_properties.subject = '说话人日志开发集音频推理与 DER 对比'
doc.save(DEST)
print(f'{DEST}  tables={len(doc.tables)}  paragraphs={len(doc.paragraphs)}')
