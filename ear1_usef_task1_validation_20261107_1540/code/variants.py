#!/usr/bin/env python3
"""Generate three fixed USEF fusion operating points from audio-only diagnostics."""
import csv
from pathlib import Path
import shutil

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/home/xujunxian/ear1_task1_compare_20261106_1712/predictions/diarizen_baseline')
MANIFEST={(r['session'],r['track']):r for r in csv.DictReader(open('/home/xujunxian/ear1_task1_compare_20261106_1712/config/audio_reference_manifest.csv'))}
STEP=.1
MODES={'activity':(-1.,-2.,.12),'voiceprint':(.35,.04,.18),'strict':(.45,.10,.25)}

for diag in sorted((ROOT/'diagnostics').glob('*/*.csv')):
    sid,track=diag.stem,diag.parent.name
    rows=list(csv.DictReader(diag.open()))
    n=int(np.ceil(float(MANIFEST[(sid,track)]['audio_duration_s'])/STEP))
    for mode,(min_sim,min_margin,min_energy) in MODES.items():
        owner=np.full(n,-1,dtype=np.int32)
        best=np.full(n,-np.inf,dtype=np.float32)
        for i,r in enumerate(rows):
            own=float(r['own_cosine']);margin=float(r['margin']);energy=float(r['median_energy_ratio'])
            if own<min_sim or margin<min_margin or energy<min_energy:continue
            a=max(0,int(round(float(r['start_s'])/STEP)))
            b=min(n,int(round(float(r['end_s'])/STEP)))
            if b<=a:continue
            confidence=own+margin+.05*np.log1p(energy)
            sel=confidence>best[a:b]
            owner[a:b][sel]=i;best[a:b][sel]=confidence
        out=ROOT/'predictions'/f'usef_{mode}'/track/f'{sid}.rttm'
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/track/f'{sid}.rttm',out)
        added=0;seconds=0.
        with out.open('a') as f:
            # Extract continuous selected target labels; require >= 0.5s.
            labels=sorted({r['target'] for r in rows})
            for label in labels:
                m=np.array([j>=0 and rows[j]['target']==label for j in owner])
                edges=np.flatnonzero(np.diff(np.r_[False,m,False].astype(np.int8)))
                for x,y in edges.reshape(-1,2):
                    if y-x<5:continue
                    a=x*STEP;b=min(y*STEP,float(MANIFEST[(sid,track)]['audio_duration_s']))
                    if b<=a:continue
                    f.write(f'SPEAKER {sid} 1 {a:.6f} {b-a:.6f} <NA> <NA> {label} <NA> <NA>\n')
                    added+=1;seconds+=b-a
        print(sid,track,mode,'added',added,'speaker_s',round(seconds,2),flush=True)
