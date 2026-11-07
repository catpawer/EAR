#!/usr/bin/env python3
"""Fixed, exploratory confidence operating points (set before scoring)."""
import csv
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'predictions'/'diarizen_baseline'
POINTS={
    'conservative':(.48,.48,.08,.78),
    'moderate':(.35,.30,.04,.75),
    'permissive':(.28,.26,.02,.80),
}

for diag in sorted((ROOT/'diagnostics').glob('*/*.csv')):
    track=diag.parent.name; sid=diag.stem
    rows=list(csv.DictReader(diag.open()))
    baseline_lines=(BASE/track/f'{sid}.rttm').read_text().splitlines()
    baseline=[]
    for line in baseline_lines:
        c=line.split();start=float(c[3]);baseline.append((start,start+float(c[4]),c[7]))
    for name,(active_min,other_min,margin_min,same_max) in POINTS.items():
        out=ROOT/'predictions'/f'tfgridnet_{name}'/track/f'{sid}.rttm'
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/track/f'{sid}.rttm',out)
        accepted=0;seconds=0.
        with out.open('a') as f:
            for r in rows:
                if not r.get('active_sim'):continue
                if (float(r['active_sim'])>=active_min and
                    float(r['other_sim'])>=other_min and
                    float(r['other_vs_active_margin'])>=margin_min and
                    float(r['stem_cosine'])<=same_max):
                    a=float(r['start_s']);b=float(r['end_s'])
                    if b<=a:continue
                    # A candidate embedding must not span a DiariZen speaker switch.
                    if any(label!=r['active'] and max(a,start)<min(b,end)
                           for start,end,label in baseline):continue
                    f.write(f"SPEAKER {sid} 1 {a:.6f} {b-a:.6f} <NA> <NA> {r['proposed']} <NA> <NA>\n")
                    accepted+=1;seconds+=b-a
        print(sid,track,name,accepted,round(seconds,2),flush=True)
