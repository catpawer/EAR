#!/usr/bin/env python3
"""Select USEF enrollment from audio-only DiariZen predictions, never reference."""
import csv
import json
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/home/xujunxian/ear1_task1_compare_20261106_1712')

def read_rttm(path):
    out={}
    for line in path.read_text().splitlines():
        c=line.split()
        if c:
            a=float(c[3]);b=a+float(c[4])
            if b>a:out.setdefault(c[7],[]).append((a,b))
    return out

def subtract(intervals,blockers):
    pieces=[]
    for a,b in intervals:
        active=[(a,b)]
        for x,y in blockers:
            nxt=[]
            for p,q in active:
                if y<=p or x>=q:nxt.append((p,q))
                else:
                    if p<x:nxt.append((p,x))
                    if y<q:nxt.append((y,q))
            active=nxt
        pieces.extend((p+.2,q-.2) for p,q in active if q-p>.8)
    return [(p,q) for p,q in pieces if q-p>=.75]

def candidates(audio,intervals):
    out=[]
    for a,b in intervals:
        p=a
        while p+1.0<=b:
            q=min(p+3.,b)
            x=audio[max(0,round(p*8000)):min(len(audio),round(q*8000))]
            if len(x)>=8000:
                rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)))
                out.append((rms,p,q,x))
            p=q
    return sorted(out,key=lambda z:z[0],reverse=True)

def main():
    manifest=list(csv.DictReader((BASE/'config'/'audio_reference_manifest.csv').open()))
    assert len(manifest)==14
    report=[]
    for row in manifest:
        sid,track=row['session'],row['track']
        x,sr=sf.read(row['audio'],dtype='float32')
        assert sr==48000 and x.ndim==1
        x=scipy.signal.resample_poly(x,1,6).astype(np.float32)
        labels=read_rttm(BASE/'predictions'/'diarizen_baseline'/track/f'{sid}.rttm')
        for speaker,intervals in sorted(labels.items()):
            blockers=[p for other,spans in labels.items() if other!=speaker for p in spans]
            exclusive=subtract(intervals,blockers)
            selected=candidates(x,exclusive)
            status='predicted_solo'
            if not selected:
                selected=candidates(x,[(a+.2,b-.2) for a,b in intervals if b-a>.8])
                status='predicted_mixed_fallback'
            clips=[];chosen=[];total=0
            for rms,a,b,chunk in selected:
                if total>=64000:break
                if rms<.0003:continue
                chunk=chunk[:64000-total]
                clips.append(chunk);chosen.append([round(a,3),round(a+len(chunk)/8000,3),round(rms,5)])
                total+=len(chunk)
            if not clips:
                report.append(dict(session=sid,track=track,speaker=speaker,status='no_enrollment',path='',duration_s=0,source_intervals='[]'))
                continue
            dest=ROOT/'enrollments'/sid/track/f'{speaker}.wav'
            dest.parent.mkdir(parents=True,exist_ok=True)
            sf.write(dest,np.concatenate(clips),8000,subtype='PCM_16')
            report.append(dict(session=sid,track=track,speaker=speaker,status=status,path=str(dest),duration_s=total/8000,source_intervals=json.dumps(chosen)))
        print(sid,track,'predicted',len(labels),'enrolled',sum(r['session']==sid and r['track']==track and r['status']!='no_enrollment' for r in report),flush=True)
    dest=ROOT/'enrollment_manifest.csv'
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,report[0]);w.writeheader();w.writerows(report)
    print('TOTAL',len(report),'usable',sum(r['status']!='no_enrollment' for r in report),dest,flush=True)

if __name__=='__main__':main()
