#!/usr/bin/env python3
"""Collect verified per-model CSVs into two convenient download files."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'scores'
models = ('diarizen_baseline','nemotron3','pyannote_community1','funasr')
for kind, expected in (('per_audio',56),('error_breakdown',224)):
    rows = []
    for model in models:
        with (root/f'{model}_{kind}.csv').open() as f:
            rows.extend(csv.DictReader(f))
    assert len(rows) == expected, (kind,len(rows))
    with (root/f'all_models_{kind}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(kind,len(rows))
