import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml

from common import ROOT, overlap_add, read_audio_8k, rows, save_wav, write_json

CODE = ROOT / 'code' / 'USEF-TSE'
sys.path.insert(0, str(CODE))
CONFIG = ROOT / 'weights' / 'USEF-TSE' / 'chkpt' / 'USEF-TFGridNet' / 'config.yaml'
CHECKPOINT = ROOT / 'weights' / 'USEF-TSE' / 'chkpt' / 'USEF-TFGridNet' / 'whamr!' / 'temp_best.pth.tar'


def load_model(device):
    with CONFIG.open() as f:
        conf = load_hyperpyyaml(f)
    model = conf['modules']['masknet']
    raw = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['model_state_dict']
    state = {k.replace('module.', '').replace('convolution_', 'convolution_module.'): v
             for k, v in raw.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def extract(model, audio, enrollment, device, chunk_s=6, overlap_s=1):
    window = int(chunk_s * 8000)
    step = int((chunk_s - overlap_s) * 8000)
    starts = list(range(0, len(audio), step))
    chunks = []
    aux = torch.from_numpy(enrollment[None]).to(device)
    skipped = 0
    with torch.inference_mode():
        for start in starts:
            x = audio[start:start + window]
            if len(x) < window:
                x = np.pad(x, (0, window - len(x)))
            if float(np.max(np.abs(x))) < 0.0002:
                y = np.zeros(window, dtype=np.float32)
                skipped += 1
            else:
                y = model(torch.from_numpy(x[None]).to(device), aux)[0].float().cpu().numpy()
            chunks.append(y[None])
    return overlap_add(chunks, starts, len(audio))[0], len(starts), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    model = load_model(device)
    with (ROOT / 'config' / 'enrollment_manifest.csv').open(newline='') as f:
        enrollments = list(csv.DictReader(f))
    lookup = {}
    for e in enrollments:
        if e['status'].startswith('ok'):
            lookup.setdefault((e['session'], e['track']), []).append(e)
    for index, row in enumerate(rows()):
        if index % args.shards != args.shard:
            continue
        start_time = time.time()
        audio = read_audio_8k(row['audio'])
        details = dict(model='USEF-TFGridNet-WHAMR', session=row['session'], track=row['track'],
                       source_audio=row['audio'], sample_rate=8000, original_rate=48000,
                       input_samples=len(audio), enrollment_source='corrected_textgrid_oracle',
                       status='complete', tracks=[])
        for e in lookup.get((row['session'], row['track']), []):
            enrollment, sr = sf.read(e['path'], dtype='float32')
            assert sr == 8000 and enrollment.ndim == 1 and len(enrollment) > 0
            t0 = time.time()
            separated, n_chunks, skipped = extract(model, audio, enrollment, device)
            target = ROOT / 'outputs' / 'usef_tse_whamr' / row['session'] / row['track'] / f"{e['speaker']}.wav"
            save_wav(target, separated)
            details['tracks'].append(dict(speaker=e['speaker'], output=str(target), samples=len(separated),
                                          chunks=n_chunks, silent_chunks_skipped=skipped,
                                          enrollment=e['path'], enrollment_status=e['status'],
                                          elapsed_s=round(time.time()-t0, 2)))
            print(row['session'], row['track'], e['speaker'], 'chunks', n_chunks,
                  'seconds', round(time.time()-t0, 1), flush=True)
        details['elapsed_s'] = round(time.time()-start_time, 2)
        if len(details['tracks']) != len(lookup.get((row['session'], row['track']), [])):
            details['status']='incomplete'
        write_json(ROOT / 'results' / 'usef_tse_whamr' / row['session'] / f"{row['track']}.json", details)


if __name__ == '__main__': main()
