"""Use corrected-reference RTTM only to select labeled, non-overlapped enrollment audio."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from common import ROOT, read_audio_8k, reference_segments, rows, save_wav


def subtract(segments, blockers):
    pieces = []
    for start, end in segments:
        active = [(start, end)]
        for bstart, bend in blockers:
            if bend <= start or bstart >= end:
                continue
            nxt = []
            for a, z in active:
                if bstart >= z or bend <= a:
                    nxt.append((a, z))
                else:
                    if a < bstart: nxt.append((a, bstart))
                    if bend < z: nxt.append((bend, z))
            active = nxt
        pieces.extend((a + .12, z - .12) for a, z in active if z - a > .75)
    return [(a, z) for a, z in pieces if z - a >= .5]


def candidates(audio, isolated):
    out = []
    for start, end in isolated:
        pos = start
        while pos + 1.0 <= end:
            stop = min(pos + 3., end)
            a = max(0, round(pos * 8000)); z = min(len(audio), round(stop * 8000))
            x = audio[a:z]
            if len(x) >= 8000:
                rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
                out.append((rms, pos, stop, x))
            pos = stop
    return sorted(out, reverse=True, key=lambda q: q[0])


def main():
    manifest = []
    for row in rows():
        audio = read_audio_8k(row['audio'])
        refs = reference_segments(row['reference_rttm'])
        tg_sha = hashlib.sha256(Path(row['reference_textgrid']).read_bytes()).hexdigest()
        for speaker, intervals in sorted(refs.items()):
            blockers = sorted((a, z) for other, spans in refs.items() if other != speaker for a, z in spans)
            choices = candidates(audio, subtract(intervals, blockers))
            if not choices:
                # A short speaker may only occur in overlaps. Mark this weaker
                # enrollment explicitly rather than silently omitting the track.
                choices = candidates(audio, [(a + .12, z - .12) for a, z in intervals if z-a > .75])
                enrollment_status = 'ok_mixed_oracle'
            else:
                enrollment_status = 'ok_exclusive_oracle'
            if not choices:
                manifest.append(dict(session=row['session'], track=row['track'], speaker=speaker,
                                     status='no_usable_enrollment', textgrid_sha256=tg_sha))
                continue
            picked = []
            total = 0
            for rms, a, z, x in choices:
                if total >= 64000: break
                x = x[:64000-total]
                if rms < 0.0003: continue
                picked.append((a, a + len(x)/8000, rms, x))
                total += len(x)
            if not picked:
                manifest.append(dict(session=row['session'], track=row['track'], speaker=speaker,
                                     status='all_candidates_quiet', textgrid_sha256=tg_sha))
                continue
            path = ROOT / 'config' / 'enrollments' / row['session'] / row['track'] / f'{speaker}.wav'
            save_wav(path, np.concatenate([x for _, _, _, x in picked]))
            manifest.append(dict(session=row['session'], track=row['track'], speaker=speaker,
                                 status=enrollment_status, path=str(path), duration_s=total/8000,
                                 source_intervals=json.dumps([[round(a, 3), round(z, 3)] for a, z, _, _ in picked]),
                                 mean_rms=float(np.mean([v for _, _, v, _ in picked])),
                                 textgrid_sha256=tg_sha))
        print(row['session'], row['track'], len(refs), flush=True)
    for item in manifest:
        if item['status'] == 'ok_mixed_oracle':
            alternatives = [x for x in manifest if x['speaker'] == item['speaker']
                            and x['track'] == item['track'] and x['status'] == 'ok_exclusive_oracle']
            if alternatives:
                best = max(alternatives, key=lambda x: float(x['duration_s']))
                item['status'] = 'ok_cross_session_oracle'
                item['path'] = best['path']
                item['duration_s'] = best['duration_s']
                item['source_intervals'] = best['source_intervals']
                item['mean_rms'] = best['mean_rms']
    keys = ['session','track','speaker','status','path','duration_s','source_intervals','mean_rms','textgrid_sha256']
    with (ROOT/'config'/'enrollment_manifest.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(manifest)
    print('enrollments', len(manifest), 'ok', sum(x['status'].startswith('ok') for x in manifest), flush=True)


if __name__ == '__main__': main()
