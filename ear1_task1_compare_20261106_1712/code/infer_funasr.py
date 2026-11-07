#!/usr/bin/env python3
"""Official FunASR audio-only VAD + ASR + CAM++ embedding/clustering pipeline."""
import argparse
import json

import torch
from funasr import AutoModel

from evaluate import ROOT, sessions, wav, write_rttm

parser = argparse.ArgumentParser()
parser.add_argument('--trial', action='store_true')
parser.add_argument('--shard-index', type=int, default=0)
parser.add_argument('--shards', type=int, default=1)
args = parser.parse_args()
assert 0 <= args.shard_index < args.shards
assert torch.cuda.is_available(), 'CUDA unavailable'
pipeline = AutoModel(model='iic/SenseVoiceSmall', vad_model='fsmn-vad', spk_model='cam++',
                     spk_mode='vad_segment', vad_kwargs={'max_single_segment_time': 30000},
                     device='cuda:0', disable_update=True)
items = [(sid, 'D1') for sid in ['20260924-5', '20260923-5']] if args.trial else [(sid, track) for sid in sessions() for track in ('D1', 'R1')]
for i, (sid, track) in enumerate(items):
    if i % args.shards != args.shard_index: continue
    dest = ROOT / 'predictions' / 'funasr' / track / f'{sid}.rttm'
    if dest.exists():
        print('SKIP', dest, flush=True); continue
    source = wav(sid, track)
    print('INFER', sid, track, source, flush=True)
    result = pipeline.generate(input=str(source), batch_size_s=30, sentence_timestamp=True,
                               return_spk_res=True)
    assert len(result) == 1, result
    data = result[0]
    sentences = data.get('sentence_info') or []
    assert sentences, f'No sentence_info: {sorted(data)}'
    rows = []
    for item in sentences:
        if item.get('spk') is None:
            continue
        start, end = float(item['start'])/1000, float(item['end'])/1000
        if end > start:
            rows.append((start, end, f'SPK_{item["spk"]}'))
    assert rows, f'No labeled segments: {sorted(data)}'
    write_rttm(dest, sid, rows)
    print('SAVED', dest, len(rows), max(e for _, e, _ in rows), flush=True)
