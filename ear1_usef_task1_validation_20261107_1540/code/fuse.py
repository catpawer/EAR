#!/usr/bin/env python3
"""Audio-only USEF target activity and voiceprint diagnostics for RTTM fusion."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf
from funasr import AutoModel

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/home/xujunxian/ear1_task1_compare_20261106_1712')
STEP=.1

def intervals(path):
    rows=[]
    for line in path.read_text().splitlines():
        c=line.split()
        if c:
            a=float(c[3]);b=a+float(c[4]);rows.append((a,b,c[7]))
    return rows

def mask(spans,n):
    m=np.zeros(n,dtype=bool)
    for a,b in spans:
        x=max(0,int(np.floor(a/STEP)));y=min(n,int(np.ceil(b/STEP)))
        m[x:y]=True
    return m

def runs(m):
    ix=np.flatnonzero(np.diff(np.r_[False,m,False].astype(np.int8)))
    return [(a*STEP,b*STEP) for a,b in ix.reshape(-1,2)]

def embedding(spk,x):
    result=spk.generate(input=x.astype(np.float32),fs=8000,show_progress=False)
    v=result[0]['spk_embedding'].detach().cpu().numpy().ravel()
    return v/np.linalg.norm(v)

def vad_spans(vad,path,cache):
    if cache.exists():return json.loads(cache.read_text())
    result=vad.generate(input=str(path),show_progress=False)
    assert len(result)==1
    spans=[(a/1000.,b/1000.) for a,b in result[0]['value'] if b>a]
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps(spans))
    return spans

def frame_rms(x,n):
    pad=n*800-len(x)
    if pad>0:x=np.pad(x,(0,pad))
    return np.sqrt(np.mean(x[:n*800].reshape(n,800).astype(np.float64)**2,axis=1))

def process(row,enroll,vad,spk):
    sid,track=row['session'],row['track']
    mix,sr=sf.read(row['audio'],dtype='float32')
    assert sr==48000 and mix.ndim==1
    mix=scipy.signal.resample_poly(mix,1,6).astype(np.float32)
    n=int(np.ceil(len(mix)/800))
    mix_rms=frame_rms(mix,n)
    base=intervals(BASE/'predictions'/'diarizen_baseline'/track/f'{sid}.rttm')
    labels=sorted({l for _,_,l in base})
    base_masks={l:mask([(a,b) for a,b,ll in base if ll==l],n) for l in labels}
    count=np.sum(np.stack(list(base_masks.values())),axis=0)
    en={r['speaker']:r for r in enroll if r['session']==sid and r['track']==track and r['status']!='no_enrollment'}
    evec={}
    for l,r in en.items():
        x,fs=sf.read(r['path'],dtype='float32')
        assert fs==8000
        evec[l]=embedding(spk,x)
    candidates=[]
    for label,r in en.items():
        path=ROOT/'outputs'/sid/track/f'{label}.wav'
        x,fs=sf.read(path,dtype='float32')
        assert fs==8000 and abs(len(x)-len(mix))<=1,(path,len(x),len(mix))
        vadmask=mask(vad_spans(vad,path,ROOT/'cache'/'vad'/sid/track/f'{label}.json'),n)
        out_rms=frame_rms(x,n)
        ratio=out_rms/np.maximum(mix_rms,1e-5)
        eligible=vadmask&(count==1)&~base_masks[label]&(ratio>=.12)&(out_rms>=.0005)
        eligible=scipy.ndimage.binary_closing(eligible,structure=np.ones(3))&(count==1)&~base_masks[label]
        for a,b in runs(eligible):
            if b-a<1.2:continue
            for start in np.arange(a,b,6.):
                end=min(start+6.,b)
                if end-start<1.2:continue
                active=[l for l,m in base_masks.items() if m[min(n-1,int(((start+end)/2)/STEP))]]
                if len(active)!=1:continue
                active=active[0]
                if active not in evec:continue
                # Discard candidates spanning a change in the baseline speaker.
                if any(ll!=active and max(start,s)<min(end,t) for s,t,ll in base):continue
                clip=x[int(start*8000):min(len(x),int(end*8000))]
                if len(clip)<8000:continue
                v=embedding(spk,clip)
                own=float(v@evec[label]);other=float(v@evec[active])
                ra=ratio[max(0,int(start/STEP)):min(n,int(np.ceil(end/STEP)))]
                candidates.append(dict(session=sid,track=track,target=label,active=active,
                                       start_s=round(start,3),end_s=round(end,3),
                                       own_cosine=own,active_cosine=other,margin=own-other,
                                       median_energy_ratio=float(np.median(ra)),
                                       enrollment_status=r['status']))
        print('TARGET',sid,track,label,'vad_intervals',int(vadmask.sum()),'candidates',sum(c['target']==label for c in candidates),flush=True)
    out=ROOT/'diagnostics'/track/f'{sid}.csv';out.parent.mkdir(parents=True,exist_ok=True)
    fields=['session','track','target','active','start_s','end_s','own_cosine','active_cosine','margin','median_energy_ratio','enrollment_status']
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(candidates)
    print('SAVED',sid,track,len(candidates),'candidates',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trial',action='store_true');args=ap.parse_args()
    manifest=list(csv.DictReader((BASE/'config'/'audio_reference_manifest.csv').open()))
    enroll=list(csv.DictReader((ROOT/'enrollment_manifest.csv').open()))
    if args.trial:manifest=[r for r in manifest if r['session']=='20260924-5']
    vad=AutoModel(model='fsmn-vad',device='cuda:0',disable_update=True)
    spk=AutoModel(model='cam++',device='cuda:0',disable_update=True)
    for row in manifest:
        out=ROOT/'diagnostics'/row['track']/f"{row['session']}.csv"
        if out.exists():print('SKIP',row['session'],row['track'],flush=True);continue
        metap=ROOT/'metadata'/row['session']/f"{row['track']}.json"
        if not metap.exists():print('WAIT',row['session'],row['track'],flush=True);continue
        process(row,enroll,vad,spk)

if __name__=='__main__':main()
