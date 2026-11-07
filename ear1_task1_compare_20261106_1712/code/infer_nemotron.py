#!/usr/bin/env python3
"""Nemotron 3 streaming inference with persistent speaker cache across whole recording."""
import argparse

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

from evaluate import ROOT, sessions, wav, write_rttm

parser = argparse.ArgumentParser()
parser.add_argument('--trial', action='store_true')
parser.add_argument('--shard-index', type=int, default=0)
parser.add_argument('--shards', type=int, default=1)
args = parser.parse_args()
assert 0 <= args.shard_index < args.shards
assert torch.cuda.is_available(), 'CUDA unavailable'
model_id = 'nvidia/Nemotron-3-Diarization'
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id).to('cuda:0').eval()
processor.set_streaming_mode('low_latency')
sr = processor.feature_extractor.sampling_rate
items = [(sid, 'D1') for sid in ['20260924-5', '20260923-5']] if args.trial else [(sid, track) for sid in sessions() for track in ('D1', 'R1')]
for i, (sid, track) in enumerate(items):
    if i % args.shards != args.shard_index: continue
    dest = ROOT / 'predictions' / 'nemotron3' / track / f'{sid}.rttm'
    if dest.exists():
        print('SKIP', dest, flush=True); continue
    source = wav(sid, track)
    audio, source_sr = sf.read(source, dtype='float32')
    assert audio.ndim == 1
    if source_sr != sr:
        from math import gcd
        g = gcd(source_sr, sr)
        audio = resample_poly(audio, sr//g, source_sr//g).astype(np.float32)
    duration = sf.info(source).duration
    assert abs(len(audio)/sr-duration) < 0.01
    print('INFER', sid, track, source, source_sr, sr, duration, flush=True)
    def chunks():
        yield processor(audio[:processor.num_samples_first_audio_chunk], sampling_rate=sr,
                        is_streaming=True, is_first_audio_chunk=True)
        mel_frame_idx = processor.num_mel_frames_per_step
        start_idx = processor.audio_chunk_start(mel_frame_idx)
        while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= len(audio):
            yield processor(audio[start_idx:end_idx], sampling_rate=sr,
                            is_streaming=True, is_first_audio_chunk=False)
            mel_frame_idx += processor.num_mel_frames_per_step
            start_idx = processor.audio_chunk_start(mel_frame_idx)
        yield processor(audio[start_idx:], sampling_rate=sr, is_streaming=True,
                        is_first_audio_chunk=False, is_last_audio_chunk=True)
    cache, logits, n = None, [], 0
    with torch.inference_mode():
        for inputs in chunks():
            inputs = inputs.to(model.device, dtype=model.dtype)
            outputs = model(**inputs, speaker_cache=cache)
            logits.append(outputs.logits.cpu())
            cache = outputs.speaker_cache
            n += 1
            if n % 500 == 0: print('PROGRESS', sid, track, n, flush=True)
    logits = torch.cat(logits, dim=1)
    coverage = logits.shape[1]*0.01
    assert abs(coverage-duration) < 0.1, (coverage, duration)
    segments = processor.extract_speaker_dict(logits)[0]
    rows = [(float(s['Start']), float(s['End']), f'SPK_{s["Speaker"]}') for s in segments]
    assert all(0 <= a < b <= duration + 0.1 for a,b,_ in rows)
    write_rttm(dest, sid, rows)
    print('SAVED', dest, len(rows), 'chunks', n, 'coverage', coverage, flush=True)
