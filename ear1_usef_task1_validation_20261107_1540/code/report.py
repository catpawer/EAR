#!/usr/bin/env python3
"""Chinese report of audio-only USEF-TSE Task 1 validation."""
import csv
import json
from pathlib import Path
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
MODELS=['diarizen_baseline','usef_activity','usef_voiceprint','usef_strict']
NAMES={'diarizen_baseline':'原始 DiariZen','usef_activity':'USEF 活动检测','usef_voiceprint':'USEF 声纹门槛','usef_strict':'USEF 严格门槛'}

def load(p):return list(csv.DictReader(p.open()))
per={m:{(r['session'],r['track']):r for r in load(ROOT/'scores'/f'{m}_per_audio.csv')} for m in MODELS}
err={m:{(r['session'],r['track'],r['region']):r for r in load(ROOT/'scores'/f'{m}_error_breakdown.csv')} for m in MODELS}
base=per['diarizen_baseline'];assert len(base)==14
for m in MODELS[1:]:assert set(per[m])==set(base),(m,len(per[m]))

inventory=[]
for sid,track in sorted(base):
    meta=json.loads((ROOT/'metadata'/sid/f'{track}.json').read_text())
    duration=meta['samples']/8000
    for target in meta['targets']:
        path=Path(target['output_path']);info=sf.info(path)
        assert info.samplerate==8000 and info.frames==meta['samples']
        spans=json.loads((ROOT/'cache'/'vad'/sid/track/f"{target['speaker']}.json").read_text())
        vad_s=sum(b-a for a,b in spans)
        inventory.append(dict(session=sid,track=track,speaker=target['speaker'],
                              enrollment_status=target['enrollment_status'],path=str(path),
                              samples=info.frames,sample_rate=info.samplerate,
                              duration_s=duration,vad_s=vad_s,vad_audio_fraction=vad_s/duration,
                              shared_gain=meta['shared_gain'],raw_peak=target['raw_peak']))
with (ROOT/'output_inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,inventory[0]);w.writeheader();w.writerows(inventory)

def micro(rows):
    den=sum(float(r['reference_speaker_s']) for r in rows)
    comp={k:sum(float(r[k]) for r in rows) for k in ('false_alarm_s','missed_detection_s','confusion_s')}
    return den,comp,100*sum(comp.values())/den if den else None

def fmt(x):return f'{x:.2f}'
def delta(x):return f'{x:+.2f}'

table=[]
for sid,track in sorted(base):
    b=float(base[(sid,track)]['overall_der_pct'])
    row=dict(session=sid,track=track,reference_speakers=base[(sid,track)]['reference_speakers'],
             predicted_speakers=base[(sid,track)]['predicted_speakers'],
             overlap_rate=base[(sid,track)]['overlap_rate'],baseline_der_pct=b)
    for m in MODELS[1:]:
        v=float(per[m][(sid,track)]['overall_der_pct'])
        row[m+'_der_pct']=v;row[m+'_delta_pp']=v-b
    table.append(row)
with (ROOT/'per_audio_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,table[0]);w.writeheader();w.writerows(table)

lines=['# USEF-TSE 对 EAR-1 任务一的音频独立验证', '',
       '## 实验设计', '',
       '用原始 DiariZen RTTM 中预测为独讲的原混合音，按预测说话人选取至多 8 秒目标声音；不读取参考 TextGrid 来选声音。USEF-TSE 对原始全长音频逐人目标提取，FSMN-VAD 在各目标轨道上找语音，CAM++ 可选地核验目标身份；只在原始 DiariZen 恰好报另一人的区域补一个说话人。参考 TextGrid 仅用于预测结束后的评分。', '',
       '- 7 场开发集、D1/R1 共 14 条，原音频 48 kHz；目标提取输出 8 kHz，全长时间轴不变。',
       '- USEF-TSE：WHAMR! 预训练的 USEF-TFGridNet，代码 revision `bdaeb3609735b3a9dbedd1b0625343826f2cb5fe`，权重 SHA-256 `2ef45e026de5248f43ef5c0585c5b94a724ab1b67307e693805d1e35ce7bf99f`；6 秒分块、1 秒重叠、交叉淡化。',
       '- 目标声音：原始 DiariZen 每个预测标签的独讲片段，按 RMS 选取至多 8 秒；没有独讲片段时用该标签的预测片段作较弱回退，并在 `enrollment_manifest.csv` 标出。没有把真实说话人数或参考身份传给 USEF。',
       '- 候选需要 USEF 目标轨经 FSMN-VAD 检出、目标/原混合音 RMS 比至少 0.12、持续至少 1.2 秒，且不跨越原始 DiariZen 说话人切换。每 0.1 秒最多增加一名说话人。',
       '- 三组探索性门槛（目标声纹余弦 / 目标相对当前说话人的余弦差 / 目标到混合音 RMS 比）：活动检测 `不筛声纹/不筛差值/0.12`；声纹门槛 `0.35/0.04/0.18`；严格门槛 `0.45/0.10/0.25`。门槛在读取对应 DER 前固定。',
       '- 评分：修正后的 `/data2/xujunxian/EAR1-Challenge/Dev` TextGrid 转换为参考 RTTM；完整录音 UEM 含静音、collar=0、重叠计分，7 场按参考说话人秒微平均。分区域错误沿用完整录音全局说话人映射，逐条验证拆解与总体一致。三组方案也用 EAR-1 官方脚本逐轨独立复核；因官方脚本在同一输出目录共用参考 RTTM，D1 与 R1 分别建目录。',
       '- 原始 DiariZen 在这套修正参考上的重评分为 D1 31.22%、R1 28.53%。历史 R1 33.75% 使用不同的参考时间轴，不能直接拿来与本次 USEF 结果比较。',
       '- `20260923-1/R1.TextGrid` 与本地修正文件 `20260923-1_R1_aligned_plus2p03.TextGrid` 的 SHA-256 均为 `9cdaed386fcad5695983a79ada3797510d3fac46446b8ddeb9b21baa28cfcfc2`。',
       '- 现有 `ear1_separation_20261106_195147/outputs/usef_tse_whamr/` 使用参考 TextGrid 选目标声音，属于参考辅助输出；本次重新推理并写入独立目录，以保证音频独立比较。',
       '', '## 7 场微平均', '',
       '| 轨道 | 方案 | 参考说话人秒 | 误报秒 | 漏检秒 | 混淆秒 | DER | 相对原始变化 |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for track in ('D1','R1'):
    b=None
    for m in MODELS:
        rows=[r for (sid,t,region),r in err[m].items() if t==track and region=='overall']
        den,c,d=micro(rows)
        if b is None:b=d
        lines.append(f"| {track} | {NAMES[m]} | {fmt(den)} | {fmt(c['false_alarm_s'])} | {fmt(c['missed_detection_s'])} | {fmt(c['confusion_s'])} | {fmt(d)}% | {delta(d-b)} pp |")
lines+=['', '| 轨道 | 方案 | 单人段 DER | 重叠段 DER | 静音误报秒 |',
        '|---|---|---:|---:|---:|']
for track in ('D1','R1'):
    for m in MODELS:
        single=[r for (sid,t,region),r in err[m].items() if t==track and region=='single']
        overlap=[r for (sid,t,region),r in err[m].items() if t==track and region=='overlap']
        silence=[r for (sid,t,region),r in err[m].items() if t==track and region=='silence']
        lines.append(f"| {track} | {NAMES[m]} | {fmt(micro(single)[2])}% | {fmt(micro(overlap)[2])}% | {fmt(sum(float(r['false_alarm_s']) for r in silence))} |")
lines+=['', '非重叠与重叠区域各自用参考说话人秒作 DER 分母，不直接相加；总体还包含静音误报。错误秒数是说话人时间，可以超过对应音频时间。',
        '', '## 逐条总体 DER', '',
        '| 场次 | 轨道 | 参考/预测人数 | 重叠率 | 原始 DiariZen | 活动检测 Δ | 声纹门槛 Δ | 严格门槛 Δ |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
for r in table:
    lines.append(f"| {r['session']} | {r['track']} | {r['reference_speakers']}/{r['predicted_speakers']} | {100*float(r['overlap_rate']):.2f}% | {r['baseline_der_pct']:.2f}% | {delta(r['usef_activity_delta_pp'])} pp | {delta(r['usef_voiceprint_delta_pp'])} pp | {delta(r['usef_strict_delta_pp'])} pp |")
lines+=['', '## 重点场次 20260924-5', '',
        '参考标注 D1/R1 各有 8 个非空发言标签；原始 DiariZen 只预测 3/4 人。这里的“参考人数”是标注标签数，并不保证等于现场实际人数。参考重叠率约 37%。', '',
        '| 轨道 | 方案 | 总体 DER | 单人段 DER | 重叠段 DER | 误报秒 | 漏检秒 | 混淆秒 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
for track in ('D1','R1'):
    for m in MODELS:
        o=err[m][('20260924-5',track,'overall')]
        s=err[m][('20260924-5',track,'single')]
        v=err[m][('20260924-5',track,'overlap')]
        lines.append(f"| {track} | {NAMES[m]} | {fmt(float(o['der_pct']))}% | {fmt(float(s['der_pct']))}% | {fmt(float(v['der_pct']))}% | {fmt(float(o['false_alarm_s']))} | {fmt(float(o['missed_detection_s']))} | {fmt(float(o['confusion_s']))} |")
lines.append('')
for track in ('D1','R1'):
    ratios=[100*r['vad_audio_fraction'] for r in inventory if r['session']=='20260924-5' and r['track']==track]
    lines.append(f"{track} 的 {len(ratios)} 条目标轨经 VAD 判为有语音的时长占各自全录音的 {min(ratios):.1f}%–{max(ratios):.1f}%；这只是目标轨活动率，不证明目标身份正确。")
lines+=['', '## 候选与输出完整性', '',
        '| 场次 | 轨道 | 已预测目标数 | 候选数 | 活动检测新增秒 | 声纹门槛新增秒 | 严格门槛新增秒 |',
        '|---|---|---:|---:|---:|---:|---:|']
for sid,track in sorted(base):
    meta=json.loads((ROOT/'metadata'/sid/f'{track}.json').read_text())
    candidates=load(ROOT/'diagnostics'/track/f'{sid}.csv')
    added=[]
    for m in MODELS[1:]:
        pred=(ROOT/'predictions'/m/track/f'{sid}.rttm').read_text().splitlines()
        baseline=(ROOT/'predictions'/'diarizen_baseline'/track/f'{sid}.rttm').read_text().splitlines()
        assert pred[:len(baseline)]==baseline
        added.append(sum(float(l.split()[4]) for l in pred[len(baseline):]))
    lines.append(f"| {sid} | {track} | {len(meta['targets'])} | {len(candidates)} | {fmt(added[0])} | {fmt(added[1])} | {fmt(added[2])} |")
lines+=['', '## 解释与限制', '',
        '目标提取依赖原始 DiariZen 已找到的说话人；没有目标声音的漏检说话人仍无法恢复。USEF 权重训练于 WHAMR! 双人混合，与 EAR-1 的多人长录音有域差异。目标音轨上的 VAD 触发和声纹匹配只能作自动代理，不能证明音轨已被干净分离。评分结果只适用于本次音频独立注册和融合规则，不能代表 USEF-TSE 的理论上限。', '',
        '## 文件', '',
        '- `enrollment_manifest.csv`：每个目标的来源、状态和片段；`enrollments/`：音频独立目标声音。',
        '- `outputs/`、`metadata/`、`output_inventory.csv`：新提取的全长 USEF 目标音轨及增益、时长、VAD 占比记录。',
        '- `diagnostics/` 与 `cache/vad/`：候选声纹分数和目标轨 VAD 时间段。',
        '- `predictions/usef_{activity,voiceprint,strict}/`：三组预测 RTTM。',
        '- `scores/`、`per_audio_comparison.csv`：逐条及分区域 DER、误报/漏检/混淆。',
        '- `official_check/`：EAR-1 官方脚本的逐轨独立评分复核。',
        '- `code/`：目标声音选取、模型推理、融合和评分代码。',
        '']

outcomes=[]
for m in MODELS[1:]:
    ds=[]
    for track in ('D1','R1'):
        a=[r for (sid,t,region),r in err[m].items() if t==track and region=='overall']
        b=[r for (sid,t,region),r in err['diarizen_baseline'].items() if t==track and region=='overall']
        ds.append(micro(a)[2]-micro(b)[2])
    outcomes.append((m,ds))
if all(all(x>=-1e-6 for x in ds) for _,ds in outcomes):
    decision='三组方案在 D1、R1 的 7 场微平均上均未改善原始 DiariZen；当前音频独立的 USEF 融合规则不适合直接用于任务一。'
else:
    decision='至少一组方案在某条轨道的 7 场微平均上改善了 DER；需结合分区域误报和逐条结果判断能否采用。'
lines[2:2]=['## 结论','',decision,'']
dest=ROOT/'EAR1_USEF_TASK1_VALIDATION.md';dest.write_text('\n'.join(lines),encoding='utf-8')
print(decision, dest)
