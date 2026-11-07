#!/usr/bin/env python3
"""EAR-1 Task 1 reference conversion and fixed-mapping DER decomposition."""
import argparse
import csv
import json
import re
from pathlib import Path

import soundfile as sf

DATA = Path('/data2/qiuenduo/EAR1-Challenge/Train&Dev')
GRIDS = Path('/data2/xujunxian/EAR1-Challenge/Dev')
SPLIT = Path('/data2/qiuenduo/EAR1-Challenge/split_tr56_dev7.lst')
ROOT = Path(__file__).resolve().parents[1]
TRACKS = ('D1', 'R1')
TIER = re.compile(r'^\s*item\s*\[\d+\]\s*:\s*$', re.M)
NAME = re.compile(r'^\s*name\s*=\s*"([^"]+)"\s*$', re.M)
INTERVAL = re.compile(r'intervals\s*\[\d+\]\s*:\s*\n\s*xmin\s*=\s*([\d.eE+-]+)\s*\n\s*xmax\s*=\s*([\d.eE+-]+)\s*\n\s*text\s*=\s*"((?:[^"]|"")*)"', re.M)


def sessions():
    content = SPLIT.read_text(encoding='utf-8-sig').replace('\r', '')
    ids = content.split('Dev Set：', 1)[1].splitlines()
    ids = [line.strip() for line in ids if line.strip()]
    assert len(ids) == 7 and len(set(ids)) == 7, ids
    return ids


def wav(sid, track):
    files = [p for p in (DATA / sid).iterdir() if p.stem.upper() == track and p.suffix.lower() == '.wav']
    assert len(files) == 1, (sid, track, files)
    return files[0]


def speech(grid):
    content = grid.read_text(encoding='utf-8-sig')
    chunks = TIER.split(content)[1:]
    assert chunks, grid
    rows = []
    for chunk in chunks:
        m = NAME.search(chunk)
        assert m, grid
        label = m.group(1)
        matches = list(INTERVAL.finditer(chunk))
        declared = re.search(r'intervals:\s*size\s*=\s*(\d+)', chunk)
        assert declared and len(matches) == int(declared.group(1)), grid
        for x in matches:
            start, end = float(x.group(1)), float(x.group(2))
            if x.group(3).replace('""', '"').strip() and end > start:
                rows.append((start, end, label))
    return sorted(rows)


def write_rttm(path, sid, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as out:
        for start, end, label in sorted(rows):
            out.write(f'SPEAKER {sid} 1 {start:.6f} {end-start:.6f} <NA> <NA> {label} <NA> <NA>\n')


def read_rttm(path, sid):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=sid)
    for index, line in enumerate(path.read_text().splitlines()):
        cols = line.split()
        if not cols:
            continue
        assert cols[0] == 'SPEAKER' and cols[1] == sid and len(cols) >= 8, (path, line)
        start, duration = float(cols[3]), float(cols[4])
        assert start >= 0 and duration >= 0, (path, line)
        if duration:
            ann[Segment(start, start + duration), index] = cols[7]
    return ann


def prepare():
    manifest = []
    for sid in sessions():
        assert (DATA / sid).is_dir() and (GRIDS / sid).is_dir(), sid
        for track in TRACKS:
            a = wav(sid, track)
            gridfiles = list((GRIDS / sid).glob(f'{track}.TextGrid'))
            assert len(gridfiles) == 1, (sid, track, gridfiles)
            info = sf.info(a)
            assert info.channels == 1
            duration = info.frames / info.samplerate
            rows = speech(gridfiles[0])
            grid_text = gridfiles[0].read_text(encoding='utf-8-sig')
            grid_end = float(re.search(r'^xmax\s*=\s*([\d.eE+-]+)', grid_text, re.M).group(1))
            assert rows and min(s for s, _, _ in rows) >= -1e-6
            # Preserve TextGrid timestamps. Full-recording UEM clips only at score time.
            out = ROOT / 'reference' / track / f'{sid}.rttm'
            write_rttm(out, sid, rows)
            manifest.append(dict(session=sid, track=track, audio=str(a), reference_textgrid=str(gridfiles[0]),
                                 reference_rttm=str(out), samplerate=info.samplerate, frames=info.frames,
                                 audio_duration_s=duration, textgrid_xmax_s=grid_end,
                                 last_speech_end_s=max(e for _, e, _ in rows),
                                 reference_labels=json.dumps(sorted({label for _, _, label in rows}), ensure_ascii=False)))
    dest = ROOT / 'config' / 'audio_reference_manifest.csv'
    with dest.open('w', newline='') as out:
        writer = csv.DictWriter(out, manifest[0].keys()); writer.writeheader(); writer.writerows(manifest)
    print(f'Prepared {len(manifest)} unique audio/TextGrid pairs: {dest}', flush=True)


def count_regions(ref, duration):
    from pyannote.core import Segment, Timeline
    events = [(0., 0), (duration, 0)]
    for segment in ref.itersegments():
        start, end = max(0., segment.start), min(duration, segment.end)
        if end > start:
            events.extend([(start, 1), (end, -1)])
    events.sort()
    regions = {0: [], 1: [], 2: []}
    count, prev = 0, 0.
    for time, delta in events:
        if time > prev:
            regions[min(count, 2)].append(Segment(prev, time))
        count += delta
        prev = time
    assert count == 0
    return {n: Timeline(segments, uri=ref.uri) for n, segments in regions.items()}


def components(ref, mapped, region):
    from pyannote.metrics.identification import IdentificationErrorRate
    m = IdentificationErrorRate(collar=0., skip_overlap=False)
    d = m(ref, mapped, uem=region, detailed=True)
    return {k: float(d[k]) for k in ('total', 'false alarm', 'missed detection', 'confusion')}


def score(model):
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    summary, detail = [], []
    base = ROOT / 'predictions' / model
    for sid in sessions():
        for track in TRACKS:
            pred = base / track / f'{sid}.rttm'
            if not pred.exists():
                continue
            ref = read_rttm(ROOT / 'reference' / track / f'{sid}.rttm', sid)
            hyp = read_rttm(pred, sid)
            audio_duration = sf.info(wav(sid, track)).duration
            uem = Timeline([Segment(0., audio_duration)], uri=sid)
            der = DiarizationErrorRate(collar=0., skip_overlap=False)
            overall = der(ref, hyp, uem=uem, detailed=True)
            mapping = der.optimal_mapping(ref, hyp, uem=uem)
            mapped = hyp.rename_labels(mapping=mapping)
            regions = count_regions(ref, audio_duration)
            areas = {name: components(ref, mapped, regions[n]) for name, n in [('silence', 0), ('single', 1), ('overlap', 2)]}
            for key in ('total', 'false alarm', 'missed detection', 'confusion'):
                found = sum(v[key] for v in areas.values())
                expected = float(overall[key])
                if abs(found - expected) > 0.005:
                    raise AssertionError((model, sid, track, key, found, expected, mapping))
            ref_labels = sorted({label for segment, _, label in ref.itertracks(yield_label=True)
                                 if segment.end > 0 and segment.start < audio_duration})
            pred_labels = sorted({label for _, _, label in hyp.itertracks(yield_label=True)})
            speech_s = sum(sum(seg.duration for seg in regions[n]) for n in (1, 2))
            overlap_s = sum(seg.duration for seg in regions[2])
            row = dict(model=model, session=sid, track=track, reference_speakers=len(ref_labels),
                       reference_labels=';'.join(ref_labels), predicted_speakers=len(pred_labels),
                       predicted_labels=';'.join(pred_labels), overlap_audio_s=overlap_s,
                       speech_audio_s=speech_s, overlap_rate=overlap_s/speech_s if speech_s else 0.,
                       overall_der_pct=100*overall['diarization error rate'],
                       single_der_pct=100*sum(areas['single'][k] for k in ('false alarm','missed detection','confusion'))/areas['single']['total'] if areas['single']['total'] else None,
                       overlap_der_pct=100*sum(areas['overlap'][k] for k in ('false alarm','missed detection','confusion'))/areas['overlap']['total'] if areas['overlap']['total'] else None,
                       audio_duration_s=audio_duration, mapping=json.dumps(mapping, ensure_ascii=False, sort_keys=True))
            summary.append(row)
            for name, stats in [('overall', overall), *areas.items()]:
                denom = float(stats['total']) if stats['total'] else 0.
                detail.append(dict(model=model, session=sid, track=track, region=name,
                                   audio_time_s=audio_duration if name == 'overall' else sum(s.duration for s in regions[{'silence':0,'single':1,'overlap':2}[name]]),
                                   reference_speaker_s=denom,
                                   false_alarm_s=float(stats['false alarm']), missed_detection_s=float(stats['missed detection']),
                                   confusion_s=float(stats['confusion']),
                                   false_alarm_contribution_pct=100*stats['false alarm']/denom if denom else None,
                                   missed_detection_contribution_pct=100*stats['missed detection']/denom if denom else None,
                                   confusion_contribution_pct=100*stats['confusion']/denom if denom else None,
                                   der_pct=100*sum(stats[k] for k in ('false alarm','missed detection','confusion'))/denom if denom else None))
    assert summary, f'No predictions for {model}'
    scores = ROOT / 'scores'; scores.mkdir(exist_ok=True)
    for name, rows in [('per_audio', summary), ('error_breakdown', detail)]:
        with (scores / f'{model}_{name}.csv').open('w', newline='') as out:
            writer = csv.DictWriter(out, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f'{model}: {len(summary)} files scored; outputs in {scores}', flush=True)
    for track in TRACKS:
        rows = [r for r in detail if r['track'] == track and r['region'] == 'overall']
        if not rows:
            continue
        total = sum(r['reference_speaker_s'] for r in rows)
        error = sum(r[k] for r in rows for k in ('false_alarm_s','missed_detection_s','confusion_s'))
        print(f'{track}: {len(rows)} files, micro DER {100*error/total:.4f}%, ref {total:.3f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','score'])
    parser.add_argument('--model')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else:
        assert args.model
        score(args.model)
