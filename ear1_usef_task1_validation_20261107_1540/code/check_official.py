#!/usr/bin/env python3
"""Run EAR-1 baseline scorer separately for D1/R1 and compare components."""
import csv
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PY='/home/xujunxian/miniconda3/envs/ear1-pyannote/bin/python'
SCRIPT='/home/xujunxian/EAR1/baseline/diarization/ear1_baseline.py'
DATA='/data2/qiuenduo/EAR1-Challenge/Train&Dev'
SPLIT='/data2/qiuenduo/EAR1-Challenge/split_tr56_dev7.lst'

for model in ('usef_activity','usef_voiceprint','usef_strict'):
    for track in ('D1','R1'):
        out=ROOT/'official_check'/model/track
        for p in (ROOT/'reference'/track).glob('*.rttm'):
            d=out/'prepare'/'dev'/p.name;d.parent.mkdir(parents=True,exist_ok=True)
            if not d.exists():d.symlink_to(p)
        for p in (ROOT/'predictions'/model/track).glob('*.rttm'):
            d=out/'infer'/'dev'/track/p.name;d.parent.mkdir(parents=True,exist_ok=True)
            if not d.exists():d.symlink_to(p)
        log=ROOT/'official_check'/f'{model}_{track}.log'
        with log.open('w') as f:
            subprocess.run([PY,SCRIPT,'score','--data',DATA,'--split',SPLIT,
                            '--tracks',track,'--out',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
        result=list(csv.DictReader((out/'infer'/'dev'/track/'result_collar0.csv').open()))
        details=[r for r in csv.DictReader((ROOT/'scores'/f'{model}_error_breakdown.csv').open())
                 if r['track']==track and r['region']=='overall']
        assert len(result)==8 and len(details)==7
        official=result[-1]
        for key,field in [('ref_s','reference_speaker_s'),('fa_s','false_alarm_s'),
                          ('miss_s','missed_detection_s'),('conf_s','confusion_s')]:
            diff=abs(float(official[key])-sum(float(r[field]) for r in details))
            assert diff<.005,(model,track,key,diff)
        print('MATCH',model,track,'DER',official['der_pct'],flush=True)
