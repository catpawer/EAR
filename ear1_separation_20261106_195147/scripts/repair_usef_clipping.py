"""Regenerate the few USEF outputs whose raw peaks exceed PCM16 range."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from common import ROOT, read_audio_8k, rows, save_wav, write_json
from run_usef import extract, load_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=4)
    args=p.parse_args()
    with (ROOT/'results'/'output_inventory.csv').open(newline='') as f:
        inventory=list(csv.DictReader(f))
    affected={}
    for x in inventory:
        if x['model']=='usef_tse_whamr' and float(x['peak'])>=.99:
            affected.setdefault((x['session'],x['track']),set()).add(x['label'])
    keys=sorted(affected)
    torch.set_num_threads(2)
    device=torch.device(f'cuda:{args.gpu}'); torch.cuda.set_device(device)
    model=load_model(device)
    source={(x['session'],x['track']):x for x in rows()}
    with (ROOT/'config'/'enrollment_manifest.csv').open(newline='') as f:
        enrollment={(x['session'],x['track'],x['speaker']):x for x in csv.DictReader(f)}
    for index,key in enumerate(keys):
        if index%args.shards!=args.shard: continue
        session,track=key
        audio=read_audio_8k(source[key]['audio'])
        meta_path=ROOT/'results'/'usef_tse_whamr'/session/f'{track}.json'
        meta=json.loads(meta_path.read_text())
        for speaker in sorted(affected[key]):
            t0=time.time()
            info=enrollment[(session,track,speaker)]
            ref,sr=sf.read(info['path'],dtype='float32')
            assert sr==8000
            output,_,_=extract(model,audio,ref,device)
            raw_peak=float(np.max(np.abs(output)))
            gain=min(1.0,.95/max(raw_peak,1e-12))
            output*=gain
            target=ROOT/'outputs'/'usef_tse_whamr'/session/track/f'{speaker}.wav'
            tmp=target.with_suffix('.repaired.wav')
            save_wav(tmp,output)
            check=sf.info(tmp)
            assert check.frames==len(audio) and check.samplerate==8000
            tmp.replace(target)
            for item in meta['tracks']:
                if item['speaker']==speaker:
                    item['peak_before_gain']=raw_peak
                    item['output_gain']=gain
                    item['clipping_repaired']=True
            print(session,track,speaker,'raw_peak',round(raw_peak,4),'gain',round(gain,4),
                  'seconds',round(time.time()-t0,1),flush=True)
        write_json(meta_path,meta)


if __name__=='__main__': main()
