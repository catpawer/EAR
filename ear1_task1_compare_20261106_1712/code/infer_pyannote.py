#!/usr/bin/env python3
"""Community-1 full-file audio-only inference; uses ordinary overlap-capable output."""
import argparse
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline

from evaluate import ROOT, sessions, wav, write_rttm

parser = argparse.ArgumentParser()
parser.add_argument('--trial', action='store_true')
parser.add_argument('--shard-index', type=int, default=0)
parser.add_argument('--shards', type=int, default=1)
parser.add_argument('--only-session')
parser.add_argument('--only-track', choices=['D1','R1'])
args = parser.parse_args()
assert 0 <= args.shard_index < args.shards
assert torch.cuda.is_available(), 'CUDA unavailable'
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1')
pipeline.to(torch.device('cuda:0'))
items = [(sid, 'D1') for sid in ['20260924-5', '20260923-5']] if args.trial else [(sid, track) for sid in sessions() for track in ('D1', 'R1')]
for i, (sid, track) in enumerate(items):
    if i % args.shards != args.shard_index: continue
    if args.only_session and sid != args.only_session: continue
    if args.only_track and track != args.only_track: continue
    dest = ROOT / 'predictions' / 'pyannote_community1' / track / f'{sid}.rttm'
    if dest.exists():
        print('SKIP', dest, flush=True); continue
    source = wav(sid, track)
    duration = sf.info(source).duration
    print('INFER', sid, track, source, flush=True)
    result = pipeline(str(source))
    ann = result.speaker_diarization
    rows = [(max(0., s.start), min(duration, s.end), str(label)) for s, _, label in ann.itertracks(yield_label=True)
            if min(duration, s.end) > max(0., s.start)]
    write_rttm(dest, sid, rows)
    print('SAVED', dest, len(rows), max((e for _, e, _ in rows), default=0), flush=True)
