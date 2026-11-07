import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = Path('/home/xujunxian/ear1_task1_compare_20261106_1712/config/audio_reference_manifest.csv')


def rows():
    with SOURCE_MANIFEST.open(newline='') as f:
        data = list(csv.DictReader(f))
    assert len(data) == 14
    for r in data:
        assert Path(r['audio']).is_file()
        assert Path(r['reference_textgrid']).is_file()
        assert '/data2/xujunxian/EAR1-Challenge/Dev/' in r['reference_textgrid']
    return data


def read_audio_8k(path):
    audio, sr = sf.read(path, dtype='float32', always_2d=False)
    assert audio.ndim == 1 and sr == 48000, (path, audio.shape, sr)
    audio = resample_poly(audio, 1, 6).astype(np.float32)
    return audio


def save_wav(path, audio):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), 8000, subtype='PCM_16')


def reference_segments(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        x = line.split()
        if len(x) >= 8 and x[0] == 'SPEAKER':
            start = float(x[3]); end = start + float(x[4]); speaker = x[7]
            if end > start:
                out.setdefault(speaker, []).append((start, end))
    return out


def overlap_add(chunks, starts, length):
    n_sources = chunks[0].shape[0]
    output = np.zeros((n_sources, length), dtype=np.float32)
    weight = np.zeros(length, dtype=np.float32)
    for i, (chunk, start) in enumerate(zip(chunks, starts)):
        n = min(chunk.shape[1], length - start)
        window = np.ones(n, dtype=np.float32)
        if i > 0:
            fade = min(n, starts[i - 1] + chunks[i - 1].shape[1] - start)
            if fade > 0:
                window[:fade] = np.linspace(0, 1, fade, endpoint=False)
        if i + 1 < len(chunks):
            fade = min(n, start + n - starts[i + 1])
            if fade > 0:
                window[-fade:] = np.minimum(window[-fade:], np.linspace(1, 0, fade, endpoint=False))
        output[:, start:start+n] += chunk[:, :n] * window
        weight[start:start+n] += window
    output /= np.maximum(weight, 1e-6)[None, :]
    return output


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
