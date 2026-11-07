"""Create aligned examples so the mixtures and outputs can be auditioned."""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from common import ROOT, rows, save_wav


EXAMPLES = [
    ('20260924-5', 'D1', 1429., 12., ['P037', 'P038', 'P039', 'P042']),
    ('20260924-5', 'R1', 1429., 12., ['P037', 'P038', 'P039', 'P042']),
    ('20260908-2', 'D1', 18., 12., ['P007', 'P008']),
]


def segment(path, start, seconds):
    with sf.SoundFile(path) as f:
        f.seek(min(round(start*f.samplerate),f.frames))
        data=f.read(round(seconds*f.samplerate),dtype='float32')
        if f.samplerate==48000: data=resample_poly(data,1,6).astype(np.float32)
        elif f.samplerate!=8000: raise ValueError((path,f.samplerate))
    return data


def main():
    bykey={(r['session'],r['track']):r for r in rows()}
    for session,track,start,seconds,speakers in EXAMPLES:
        r=bykey[(session,track)]
        out=ROOT/'review_clips'/session/track
        save_wav(out/'mixture.wav',segment(r['audio'],start,seconds))
        for source in [1,2]:
            p=ROOT/'outputs'/'tfgridnet'/session/track/f'source{source}.wav'
            if p.is_file(): save_wav(out/f'tfgridnet_source{source}.wav',segment(p,start,seconds))
        for speaker in speakers:
            p=ROOT/'outputs'/'usef_tse_whamr'/session/track/f'{speaker}.wav'
            if p.is_file(): save_wav(out/f'usef_{speaker}.wav',segment(p,start,seconds))
        print(out)


if __name__=='__main__': main()
