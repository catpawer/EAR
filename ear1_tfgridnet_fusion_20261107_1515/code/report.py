#!/usr/bin/env python3
"""Summarize fixed-operating-point DER scores; no inference or threshold tuning."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCORES=ROOT/'scores'
MODELS=['diarizen_baseline','tfgridnet_conservative','tfgridnet_moderate','tfgridnet_permissive']
NAMES={'diarizen_baseline':'原始 DiariZen','tfgridnet_conservative':'TF-GridNet 保守','tfgridnet_moderate':'TF-GridNet 中等','tfgridnet_permissive':'TF-GridNet 宽松'}

def read(name):
    with (SCORES/name).open() as f:return list(csv.DictReader(f))

per={m:{(r['session'],r['track']):r for r in read(m+'_per_audio.csv')} for m in MODELS}
err={m:{(r['session'],r['track'],r['region']):r for r in read(m+'_error_breakdown.csv')} for m in MODELS}
base=per[MODELS[0]]
assert len(base)==14
for m in MODELS[1:]:assert set(per[m])==set(base),(m,len(per[m]))

def der(r):return float(r['overall_der_pct'])
def micro(rows):
    total=sum(float(r['reference_speaker_s']) for r in rows)
    errors={k:sum(float(r[k]) for r in rows) for k in ('false_alarm_s','missed_detection_s','confusion_s')}
    return total,errors,100*sum(errors.values())/total if total else None

def fmt(x,n=2):return f'{x:.{n}f}' if x is not None else '—'
def change(x):return f'{x:+.2f}'

table=[]
for sid,track in sorted(base):
    row={'session':sid,'track':track,
         'reference_speakers':base[(sid,track)]['reference_speakers'],
         'predicted_speakers':base[(sid,track)]['predicted_speakers'],
         'overlap_rate':base[(sid,track)]['overlap_rate']}
    b=der(base[(sid,track)])
    row['baseline_der_pct']=b
    for m in MODELS[1:]:
        row[m+'_der_pct']=der(per[m][(sid,track)])
        row[m+'_delta_pp']=der(per[m][(sid,track)])-b
    table.append(row)
with (ROOT/'per_audio_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,table[0]);w.writeheader();w.writerows(table)

lines=['# TF-GridNet 分离音轨对 EAR-1 任务一日志的验证', '',
       '## 结论', '',
       '本实验把 TF-GridNet 双音轨中的候选重叠发言加到原始 DiariZen RTTM，检验任务一 DER 是否改善。只用音频和原始 DiariZen 预测进行推理；参考 TextGrid 仅在生成预测后评分。先以保守门槛试跑；它没有新增发言。随后在读取中等、宽松方案评分前固定另外两组门槛。该过程属于探索性验证，不是预注册测试。', '',
       '## 方法与可复核文件', '',
       '- 开发集 7 场，D1/R1 共 14 条。原音频 48 kHz，TF-GridNet 输出 8 kHz；全部音轨时长与原音频相差不超过一个 8 kHz 采样点。声纹输入按模型需要重采样到 16 kHz，RTTM 时间戳均按秒保留在原音频时间轴。',
       '- TF-GridNet 权重：作者 WSJ0-2mix 单声道双声源模型，SHA-256 `f8ef60995e82d7c0ba3273eef7d837ab2d9be93eea5ab9d75a6eae42f0e51949`；分离输出来自 `ear1_separation_20261106_195147/outputs/tfgridnet/`。',
       '- 双音轨各自使用 FunASR 1.4.22 的 FSMN-VAD 检测语音；使用同版 FunASR 的 CAM++ 声纹，将候选片段对应到原始 DiariZen 的说话人。声纹模板从原始 DiariZen 预测为独讲的原混合音片段提取；没有使用真实人数、参考身份或参考区间。',
       '- 只在两条分离音轨均被 VAD 判为语音、原 DiariZen 恰好判定 1 人时形成候选。候选至少 1.2 秒，跨越 DiariZen 说话人切换的候选被剔除；双音轨需要分别匹配到当前说话人和另一个已知 DiariZen 说话人，才添加第二个 RTTM 标签。保留原 RTTM 的全部行。TF-GridNet 两路不是稳定身份轨，且本方法只能恢复原 DiariZen 已发现的说话人，无法补出完全未发现的人。',
       '- 三组预定门槛依次为 `当前声纹相似度 / 新声纹相似度 / 新身份相对当前身份的最小相似度差 / 两音轨声纹最大相似度`：保守 `0.48/0.48/0.08/0.78`，中等 `0.35/0.30/0.04/0.75`，宽松 `0.28/0.26/0.02/0.80`。相似度均为 CAM++ 向量余弦。',
       '- 评分采用 EAR-1 官方口径：逐轨完整音频 UEM（含静音），collar=0，重叠计分，7 场按参考说话人时长微平均。参考来自 `/data2/xujunxian/EAR1-Challenge/Dev` 的修正 TextGrid；`20260923-1/R1.TextGrid` 与本地 `20260923-1_R1_aligned_plus2p03.TextGrid` SHA-256 相同。各区域沿用完整录音全局说话人映射；评分代码逐条核验分区域误报、漏检、混淆加总与整体一致。',
       '- 原始 DiariZen RTTM 按相同参考重评 D1 31.22%、R1 28.53%。历史 R1 33.75% 使用不同参考时间轴，不能作为本实验的直接对照；前次实验已使用 EAR-1 官方脚本逐条核验重评数值。本次中等、宽松方案也用官方脚本逐轨独立复核，微平均与拆解评分脚本一致，日志在 `official_check/`。官方脚本在同一输出目录内共用一份参考 RTTM，因此 D1 与 R1 必须各建一套评分目录。',
       '', '## 7 场微平均', '',
       '| 轨道 | 方案 | 参考说话人秒 | 误报秒 | 漏检秒 | 混淆秒 | DER | 相对原始变化 |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for track in ('D1','R1'):
    baseline_der=None
    for m in MODELS:
        rows=[r for (sid,t,region),r in err[m].items() if t==track and region=='overall']
        total,e,d=micro(rows)
        if m==MODELS[0]:baseline_der=d
        lines.append(f"| {track} | {NAMES[m]} | {fmt(total)} | {fmt(e['false_alarm_s'])} | {fmt(e['missed_detection_s'])} | {fmt(e['confusion_s'])} | {fmt(d)}% | {change(d-baseline_der)} pp |")
lines+=['', '| 轨道 | 方案 | 单人段 DER | 重叠段 DER | 静音误报秒 |',
        '|---|---|---:|---:|---:|']
for track in ('D1','R1'):
    for m in MODELS:
        singles=[r for (sid,t,region),r in err[m].items() if t==track and region=='single']
        overlaps=[r for (sid,t,region),r in err[m].items() if t==track and region=='overlap']
        silence=[r for (sid,t,region),r in err[m].items() if t==track and region=='silence']
        lines.append(f"| {track} | {NAMES[m]} | {fmt(micro(singles)[2])}% | {fmt(micro(overlaps)[2])}% | {fmt(sum(float(r['false_alarm_s']) for r in silence))} |")
lines+=['', '## 逐条音频总体 DER', '',
       '| 场次 | 轨道 | 参考/预测人数 | 重叠率 | 原始 DiariZen | 保守 Δ | 中等 Δ | 宽松 Δ |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for r in table:
    lines.append(f"| {r['session']} | {r['track']} | {r['reference_speakers']}/{r['predicted_speakers']} | {100*float(r['overlap_rate']):.2f}% | {r['baseline_der_pct']:.2f}% | {change(r['tfgridnet_conservative_delta_pp'])} pp | {change(r['tfgridnet_moderate_delta_pp'])} pp | {change(r['tfgridnet_permissive_delta_pp'])} pp |")
lines+=['', '## 重点场次：20260924-5', '',
       '参考标注人数 D1/R1 均为 8；原始 DiariZen 只预测到 3/4 人。参考重叠率约 37%，这也是双声源分离前端最可能发挥作用的一场。标注人数是非空发言标签数，不保证等于现场实际人数。', '',
       '| 轨道 | 方案 | 总体 DER | 单人段 DER | 重叠段 DER | 总误报秒 | 总漏检秒 | 总混淆秒 |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for track in ('D1','R1'):
    for m in MODELS:
        o=err[m][('20260924-5',track,'overall')]
        s=err[m][('20260924-5',track,'single')]
        v=err[m][('20260924-5',track,'overlap')]
        lines.append(f"| {track} | {NAMES[m]} | {fmt(float(o['der_pct']))}% | {fmt(float(s['der_pct']))}% | {fmt(float(v['der_pct']))}% | {fmt(float(o['false_alarm_s']))} | {fmt(float(o['missed_detection_s']))} | {fmt(float(o['confusion_s']))} |")
lines+=['', '重叠段 DER 与单人段 DER 各自以该区域参考说话人秒为分母，不直接相加；总体还包含静音误报。所有错误时长单位是说话人秒，不是音频秒。', '',
       '## 候选片段与限制', '',
       '| 场次 | 轨道 | 候选数 | 保守新增秒 | 中等新增秒 | 宽松新增秒 |',
       '|---|---|---:|---:|---:|---:|']
for sid,track in sorted(base):
    p=ROOT/'diagnostics'/track/f'{sid}.csv'
    rows=list(csv.DictReader(p.open()))
    def add_seconds(m):
        pred=ROOT/'predictions'/m/track/f'{sid}.rttm'
        basep=ROOT/'predictions'/'diarizen_baseline'/track/f'{sid}.rttm'
        lines=pred.read_text().splitlines()[len(basep.read_text().splitlines()):]
        return sum(float(l.split()[4]) for l in lines)
    vals=[add_seconds(m) for m in MODELS[1:]]
    lines.append(f"| {sid} | {track} | {len(rows)} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} |")
lines+=['', '本验证只测试了从已有 DiariZen 说话人中补第二人，且 TF-GridNet 的 WSJ0-2mix 双人训练条件与本数据中的 4–8 人场次不一致。严苛门槛可能无新增；宽松门槛可能把串音当成第二人。结果只支持对这套具体融合规则作判断，不代表所有分离加日志方案的上限。现有 USEF-TSE 音轨使用修正 TextGrid 选取目标说话人声音，属于参考辅助实验，因此没有纳入这次音频独立的任务一 DER 对比。',
        '', '## 文件', '',
        '- `predictions/tfgridnet_{conservative,moderate,permissive}/`：新增候选后的逐条 RTTM。',
        '- `scores/*_per_audio.csv` 与 `scores/*_error_breakdown.csv`：逐条人数、重叠率、整体及分区域 DER、误报/漏检/混淆。',
        '- `per_audio_comparison.csv`：逐条相对原始 DiariZen 的 DER 变化。',
        '- `diagnostics/`：每个候选的声纹分数和决定；`cache/vad/`：两路 VAD 时间段。',
        '- `official_check/`：EAR-1 官方评分脚本对中等、宽松方案的独立逐轨复核。',
        '- `code/fuse.py`、`code/variants.py`、`code/evaluate.py`：推理、固定门槛和评分代码。',
        '']

# Decision sentence computed from micro scores, with no subjective threshold choice.
outcomes=[]
for m in MODELS[1:]:
    diffs=[]
    for track in ('D1','R1'):
        rows_m=[r for (sid,t,region),r in err[m].items() if t==track and region=='overall']
        rows_b=[r for (sid,t,region),r in err[MODELS[0]].items() if t==track and region=='overall']
        diffs.append(micro(rows_m)[2]-micro(rows_b)[2])
    outcomes.append((m,diffs))
if all(all(d>=-1e-6 for d in ds) for _,ds in outcomes):
    decision='三组固定门槛在 D1、R1 的 7 场微平均上都没有优于原始 DiariZen；当前融合方式不建议作为任务一正式系统。'
else:
    decision='至少一组门槛在某条轨道改善了 7 场微平均；需结合逐条及区域误报、漏检再决定是否采用，不能只看重点场次。'
lines.insert(5,decision)
(ROOT/'EAR1_TFGRIDNET_TASK1_VALIDATION.md').write_text('\n'.join(lines),encoding='utf-8')
print(decision)
print(ROOT/'EAR1_TFGRIDNET_TASK1_VALIDATION.md')
