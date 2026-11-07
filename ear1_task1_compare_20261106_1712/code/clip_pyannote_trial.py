#!/usr/bin/env python3
"""Clip the two raw Community-1 trial RTTMs to their original audio duration."""
from pathlib import Path
import shutil

import soundfile as sf
from evaluate import ROOT, wav, write_rttm

for sid in ('20260924-5', '20260923-5'):
    source = ROOT / 'predictions' / 'pyannote_community1' / 'D1' / f'{sid}.rttm'
    assert source.exists(), source
    backup = ROOT / 'logs' / 'pyannote_trial_unclipped' / f'{sid}.rttm'
    backup.parent.mkdir(parents=True, exist_ok=True)
    assert not backup.exists(), backup
    shutil.copy2(source, backup)
    duration = sf.info(wav(sid, 'D1')).duration
    rows = []
    max_raw_end = 0.
    for line in source.read_text().splitlines():
        cols = line.split()
        start, end = float(cols[3]), float(cols[3]) + float(cols[4])
        max_raw_end = max(max_raw_end, end)
        end = min(end, duration)
        if end > start:
            rows.append((start, end, cols[7]))
    write_rttm(source, sid, rows)
    print(sid, 'audio_s', duration, 'raw_end_s', max_raw_end, 'kept_segments', len(rows))
