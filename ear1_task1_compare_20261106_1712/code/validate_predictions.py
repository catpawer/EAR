#!/usr/bin/env python3
"""Check exactly 14 RTTMs per model and timestamps against original audio."""
import csv
import re
from evaluate import ROOT, sessions

manifest = {(r['session'],r['track']):r for r in csv.DictReader((ROOT/'config/audio_reference_manifest.csv').open())}
assert len(manifest) == 14
models = ('diarizen_baseline','nemotron3','pyannote_community1','funasr')
rows = []
for model in models:
    found = sorted((ROOT/'predictions'/model).glob('*/*.rttm'))
    assert len(found) == 14, (model,len(found))
    for sid in sessions():
        for track in ('D1','R1'):
            path = ROOT/'predictions'/model/track/f'{sid}.rttm'
            assert path.exists(), path
            duration = float(manifest[(sid,track)]['audio_duration_s'])
            count, min_start, max_end, labels = 0, float('inf'), 0., set()
            for line in path.read_text().splitlines():
                cols = line.split()
                assert len(cols) >= 8 and cols[0] == 'SPEAKER' and cols[1] == sid
                start, dur = float(cols[3]),float(cols[4])
                assert start >= 0 and dur > 0
                count += 1; min_start = min(min_start,start); max_end = max(max_end,start+dur); labels.add(cols[7])
            # Existing baseline RTTMs are left untouched; full-recording UEM clips them at score time.
            tolerance = 0.5 if model == 'diarizen_baseline' else 0.1
            assert count and max_end <= duration + tolerance, (model,sid,track,max_end,duration)
            rows.append(dict(model=model,session=sid,track=track,segments=count,speakers=len(labels),
                             earliest_prediction_s=min_start,latest_prediction_s=max_end,audio_duration_s=duration,
                             end_minus_audio_s=max_end-duration))
out = ROOT/'config/prediction_integrity.csv'
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
print(f'Validated {len(rows)} RTTMs from {len(models)} model groups: {out}')

coverage = {}
pattern = re.compile(r'/predictions/nemotron3/(D1|R1)/([^/]+)\.rttm.*\bcoverage\s+([\d.]+)')
for name in ('nemotron_trial.log','nemotron_full_shard0.log','nemotron_full_shard1.log'):
    for line in (ROOT/'logs'/name).read_text().splitlines():
        match = pattern.search(line)
        if match:
            track,sid,seconds = match.groups()
            key = (sid,track)
            assert key not in coverage, (key,name)
            duration = float(manifest[key]['audio_duration_s'])
            assert abs(float(seconds)-duration) < 0.01, (key,seconds,duration)
            coverage[key] = dict(session=sid,track=track,logits_coverage_s=seconds,audio_duration_s=duration,
                                 difference_s=float(seconds)-duration)
assert len(coverage) == 14, len(coverage)
out = ROOT/'config/nemotron_full_audio_coverage.csv'
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,next(iter(coverage.values())).keys());w.writeheader();w.writerows(coverage.values())
print(f'Validated Nemotron logits cover all 14 original audio durations within 0.01 s: {out}')
