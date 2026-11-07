#!/usr/bin/env python3
"""USEF-TSE extraction conditioned only on DiariZen-derived audio enrollment."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/xujunxian/ear1_separation_20261106_195147')
sys.path.insert(0,str(OLD/'scripts'))
from common import read_audio_8k
from run_usef import load_model,extract

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True)
    ap.add_argument('--trial',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.gpu)
    device=torch.device(f'cuda:{args.gpu}')
    with (ROOT/'enrollment_manifest.csv').open() as f:enroll=list(csv.DictReader(f))
    source=list(csv.DictReader(open('/home/xujunxian/ear1_task1_compare_20261106_1712/config/audio_reference_manifest.csv')))
    tasks=[r for r in source if not args.trial or r['session']=='20260924-5']
    selected=[r for i,r in enumerate(tasks) if i%args.shards==args.shard]
    model=load_model(device)
    for row in selected:
        sid,track=row['session'],row['track']
        target_dir=ROOT/'outputs'/sid/track
        labels=[r for r in enroll if r['session']==sid and r['track']==track and r['status']!='no_enrollment']
        metap=ROOT/'metadata'/sid/f'{track}.json'
        if metap.exists() and all((target_dir/f"{r['speaker']}.wav").exists() for r in labels):
            print('SKIP',sid,track,flush=True);continue
        start=time.time()
        audio=read_audio_8k(row['audio'])
        outputs={};details=[]
        for r in labels:
            ref,sr=sf.read(r['path'],dtype='float32')
            assert sr==8000 and len(ref)>0
            t0=time.time()
            y,nchunks,skipped=extract(model,audio,ref,device)
            assert len(y)==len(audio)
            outputs[r['speaker']]=y
            details.append(dict(speaker=r['speaker'],enrollment_status=r['status'],enrollment_path=r['path'],chunks=nchunks,silent_chunks_skipped=skipped,raw_peak=float(np.max(np.abs(y))),elapsed_s=round(time.time()-t0,2)))
            print('EXTRACT',sid,track,r['speaker'],'chunks',nchunks,'sec',round(time.time()-t0,1),flush=True)
        peak=max((v['raw_peak'] for v in details),default=0.)
        gain=min(1.,.95/max(peak,1e-12))
        target_dir.mkdir(parents=True,exist_ok=True)
        for d in details:
            speaker=d['speaker'];dest=target_dir/f'{speaker}.wav'
            temp=target_dir/f'{speaker}.tmp.wav'
            sf.write(temp,outputs[speaker]*gain,8000,subtype='PCM_16')
            info=sf.info(temp)
            assert info.frames==len(audio) and info.samplerate==8000
            temp.replace(dest)
            d['output_path']=str(dest)
        metap.parent.mkdir(parents=True,exist_ok=True)
        metap.write_text(json.dumps(dict(session=sid,track=track,audio=row['audio'],sample_rate=8000,
                                         samples=len(audio),enrollment_source='diarizen_predictions_only',
                                         shared_gain=gain,targets=details,elapsed_s=round(time.time()-start,2)),indent=2))
        print('SAVED',sid,track,len(details),'targets','gain',round(gain,5),flush=True)

if __name__=='__main__':main()
