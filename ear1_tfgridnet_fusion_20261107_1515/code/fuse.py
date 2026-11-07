#!/usr/bin/env python3
"""Audio-only exploratory TF-GridNet overlap recovery for EAR-1 Task 1.

Reference labels and RTTM are never read here. Enrollment comes from DiariZen
single-speaker predictions on the original mix. Separated-source FSMN-VAD and
CAM++ embeddings must support both speakers before a new overlap label is added.
"""
import argparse
import csv
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from funasr import AutoModel

BASE = Path('/home/xujunxian/ear1_task1_compare_20261106_1712')
STEMS = Path('/home/xujunxian/ear1_separation_20261106_195147/outputs/tfgridnet')
ROOT = Path(__file__).resolve().parents[1]
STEP = .1
MIN_CAND = 1.2
MIN_SIM = .48
MIN_MARGIN = .08
MAX_STEM_SAME = .78

def rttm(path):
    rows=[]
    for line in path.read_text().splitlines():
        c=line.split()
        if c:
            s=float(c[3]); e=s+float(c[4])
            if e>s: rows.append((s,e,c[7]))
    return rows

def mask_intervals(intervals,n,step=STEP):
    m=np.zeros(n,dtype=bool)
    for s,e in intervals:
        a=max(0,int(np.floor(s/step))); b=min(n,int(np.ceil(e/step)))
        m[a:b]=True
    return m

def runs(mask,step=STEP):
    ix=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8)))
    return [(a*step,b*step) for a,b in ix.reshape(-1,2)]

def emb(model,x,fs):
    result=model.generate(input=x.astype(np.float32),fs=fs,show_progress=False)
    v=result[0]['spk_embedding'].detach().cpu().numpy().ravel()
    return v/np.linalg.norm(v)

def enrollment(spk,orig,base,fs):
    labels=sorted({l for _,_,l in base})
    templates={}
    # Select long DiariZen solo intervals; ranking depends on prediction only.
    for label in labels:
        mine=[]
        for s,e,l in base:
            if l!=label or e-s<2.5: continue
            a=s+.2; b=min(e-.2,a+5.)
            if b-a<2.0: continue
            if any(ol!=label and max(a,os)<min(b,oe) for os,oe,ol in base): continue
            mine.append((b-a,a,b))
        mine=sorted(mine,reverse=True)[:8]
        vectors=[]
        for _,a,b in mine:
            x=orig[int(a*fs):int(b*fs)]
            if len(x)<2*fs: continue
            x=scipy.signal.resample_poly(x,16000,fs)
            try: vectors.append(emb(spk,x,16000))
            except Exception as ex: logging.warning('enrollment %s %.1f: %s',label,a,ex)
        if vectors:
            v=np.median(np.stack(vectors),axis=0)
            templates[label]=v/np.linalg.norm(v)
    return templates,{l:sum(1 for _,_,ll in base if ll==l) for l in labels}

def vad_intervals(vad,path,cache):
    if cache.exists(): return json.loads(cache.read_text())
    result=vad.generate(input=str(path),show_progress=False)
    assert len(result)==1,result
    intervals=[(a/1000.,b/1000.) for a,b in result[0]['value'] if b>a]
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps(intervals))
    return intervals

def process(sid,track,vad,spk,manifest):
    key=(sid,track)
    audio=Path(manifest[key]['audio'])
    baseline=BASE/'predictions'/'diarizen_baseline'/track/f'{sid}.rttm'
    out=ROOT/'predictions'/'tfgridnet_fusion'/track/f'{sid}.rttm'
    if out.exists():
        print('SKIP',sid,track,flush=True); return
    assert audio.exists() and baseline.exists()
    orig,fs=sf.read(audio,dtype='float32')
    assert fs==48000 and orig.ndim==1
    duration=len(orig)/fs
    stem_paths=[STEMS/sid/track/f'source{i}.wav' for i in (1,2)]
    stem=[]; vads=[]
    for i,path in enumerate(stem_paths,1):
        info=sf.info(path)
        assert info.samplerate==8000 and abs(info.duration-duration)<1/8000+1e-4,(path,info.duration,duration)
        stem.append(sf.read(path,dtype='float32')[0])
        intervals=vad_intervals(vad,path,ROOT/'cache'/'vad'/sid/track/f'source{i}.json')
        vads.append(intervals)
    base=rttm(baseline)
    n=int(np.ceil(duration/STEP))
    source_masks=[mask_intervals(v,n) for v in vads]
    labels=sorted({l for _,_,l in base})
    base_masks={l:mask_intervals([(s,e) for s,e,ll in base if ll==l],n) for l in labels}
    count=np.sum(np.stack(list(base_masks.values())),axis=0)
    both=source_masks[0]&source_masks[1]
    eligible=both&(count==1)
    # Merge tiny VAD gaps, but never bridge a non-single-speaker baseline area.
    eligible=scipy.ndimage.binary_closing(eligible,structure=np.ones(3))&(count==1)
    segments=[]
    for a,b in runs(eligible):
        if b-a<MIN_CAND: continue
        for x in np.arange(a,b,6.):
            y=min(x+6.,b)
            if y-x>=MIN_CAND: segments.append((x,y))
    templates,counts=enrollment(spk,orig,base,fs)
    print('FILE',sid,track,'duration',round(duration,1),'base',counts,'enroll',list(templates),
          'vad',list(map(len,vads)),'both_s',round(float(both.sum()*STEP),1),'candidates',len(segments),flush=True)
    candidates=[]; added=[]
    for a,b in segments:
        mid=int(((a+b)/2)/STEP)
        active=[l for l,m in base_masks.items() if m[mid]]
        if len(active)!=1 or len(templates)<2: continue
        active=active[0]
        if active not in templates: continue
        vectors=[]
        for x in stem:
            clip=x[int(a*8000):int(b*8000)]
            if len(clip)<8000: break
            vectors.append(emb(spk,clip,8000))
        if len(vectors)!=2: continue
        sims=np.array([[float(v@templates[l]) for l in templates] for v in vectors])
        labs=list(templates)
        ia=labs.index(active)
        choices=[]
        for active_lane,other_lane in ((0,1),(1,0)):
            j=int(np.argmax(np.where(np.arange(len(labs))==ia,-2,sims[other_lane])))
            choices.append((min(sims[active_lane,ia],sims[other_lane,j]),active_lane,other_lane,j))
        confidence,al,ol,j=max(choices)
        other=labs[j]
        active_sim=float(sims[al,ia]); other_sim=float(sims[ol,j])
        margin=float(other_sim-sims[ol,ia]); stem_same=float(vectors[0]@vectors[1])
        accepted=active_sim>=MIN_SIM and other_sim>=MIN_SIM and margin>=MIN_MARGIN and stem_same<=MAX_STEM_SAME
        if accepted: added.append((a,b,other))
        candidates.append(dict(session=sid,track=track,start_s=a,end_s=b,active=active,
                               proposed=other,active_sim=active_sim,other_sim=other_sim,
                               other_vs_active_margin=margin,stem_cosine=stem_same,
                               accepted=int(accepted),all_similarities=json.dumps(sims.tolist())))
    out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(baseline,out)
    with out.open('a') as f:
        for a,b,l in added:
            f.write(f'SPEAKER {sid} 1 {a:.6f} {b-a:.6f} <NA> <NA> {l} <NA> <NA>\n')
    diag=ROOT/'diagnostics'/track/f'{sid}.csv';diag.parent.mkdir(parents=True,exist_ok=True)
    if candidates:
        with diag.open('w',newline='') as f:
            w=csv.DictWriter(f,candidates[0]);w.writeheader();w.writerows(candidates)
    else:diag.write_text('session,track,start_s,end_s,accepted\n')
    print('SAVED',sid,track,'added',len(added),'speaker_s',round(sum(b-a for a,b,_ in added),2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trial',action='store_true');args=ap.parse_args()
    with (BASE/'config'/'audio_reference_manifest.csv').open() as f:
        manifest={(r['session'],r['track']):r for r in csv.DictReader(f)}
    tasks=[('20260924-5',t) for t in ('D1','R1')] if args.trial else list(manifest)
    vad=AutoModel(model='fsmn-vad',device='cuda:0',disable_update=True)
    spk=AutoModel(model='cam++',device='cuda:0',disable_update=True)
    for sid,track in tasks:process(sid,track,vad,spk,manifest)

if __name__=='__main__':main()
