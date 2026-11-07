#!/usr/bin/env python3
"""Build a Chinese experiment report from available verified scoring CSVs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / 'scores'


def rows(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def num(x, digits=2, suffix=''):
    return f'{float(x):.{digits}f}{suffix}' if x not in ('', None) else '—'


models = ['diarizen_baseline', 'nemotron3', 'pyannote_community1', 'funasr']
summaries = {m: rows(SCORES / f'{m}_per_audio.csv') for m in models}
details = {m: rows(SCORES / f'{m}_error_breakdown.csv') for m in models}
status_path = ROOT / 'config' / 'model_status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else {}
out = ['# EAR-1 任务一：音频说话人日志模型对比', '',
       '## 服务器与环境', '',
       '服务器有 4 张 NVIDIA GeForce RTX 4090（每张 24,564 MiB 显存），驱动 595.99.02。实验开始时用户目录所在磁盘约余 689 GiB，`/data2` 约余 8.4 TiB。'
       '三个新方案分别使用独立 Conda 环境 `ear1-nemotron`、`ear1-pyannote`、`ear1-funasr`；原有 `diarizen`、`asr` 未修改。'
       'Nemotron 与 FunASR 环境从 `asr` 环境克隆后在新路径安装各自所需包，原环境保持独立。', '',
       '## 数据与评分口径', '',
       '音频来自 `/data2/qiuenduo/EAR1-Challenge/Train&Dev`；参考标注仅来自 `/data2/xujunxian/EAR1-Challenge/Dev` 的同场次同轨道 TextGrid。'
       '14 对一一对应文件及采样率、时长见 [音频与参考清单](config/audio_reference_manifest.csv)。原始数据未改动。', '',
       '14 条音频均为单声道 48 kHz；对应 TextGrid 的 `xmax` 与各自音频时长相差不超过 0.005 秒。'
       'Nemotron 推理时从 48 kHz 重采样为 16 kHz，并核验 14 条输出帧均覆盖原始时长（误差小于 0.01 秒）；'
       'FunASR 直接读取原始文件，输出毫秒时间戳转换为原始音频秒。', '',
       '四组共 56 条 RTTM 的文件数和时间边界均通过核验，见 [完整性清单](config/prediction_integrity.csv) 与 '
       '[Nemotron 全音频覆盖检查](config/nemotron_full_audio_coverage.csv)。原始 DiariZen 预测在 `20260924-5/D1` 末尾越界 0.127 秒，'
       '仍保留原文件并由完整录音 UEM 裁切评分。Community-1 试跑原始输出曾越界 0.445 秒，已备份于 `logs/pyannote_trial_unclipped/`，'
       '正式 RTTM 裁到原音频末尾；分数不变。', '',
       '参考 RTTM 由有非空文字的 TextGrid 区间转换，标注人数不保证等于现场实际人数。标注区间边界包含一定静音。'
       '推理只读取音频，不提供真实人数、说话人身份或参考区间。', '',
       'DER 按官方 EAR-1 基线代码口径：D1/R1 分轨评分；UEM 为各自完整音频 `[0, 时长]`，包含静音；collar=0；重叠计分；'
       '7 场总体用参考说话人时长加权的微平均。以下所有错误单位是“说话人秒”，与音频时间不同；多人重叠时前者可以更大。'
       '重叠率分母是至少一人被标注的音频秒数，分子是至少两人同时被标注的音频秒数。', '',
       '区域 DER 各自用该区域参考说话人秒数作分母，并沿用每条完整录音所得的全局最佳说话人匹配。'
       '静音区误报单列；脚本逐条验证三区域误报、漏检、混淆和参考时长之和与总体一致。区域 DER 不直接相加。', '',
       '## 基线核验与参考差异', '',
       '原始 DiariZen 的已报告结果为 D1 **31.22%**、R1 **33.75%**。使用本次指定 TextGrid 和现存原始 DiariZen 预测重新评分，'
       'D1 为 **31.22%**，R1 为 **28.53%**。R1 的差异来自指定 TextGrid 与原先参考的时间轴不同；'
       '例如 `20260923-1/R1` 由 46.92% 变为 20.93%，与既有对齐诊断一致。没有为贴合旧数值改变 collar、UEM 或重叠口径。'
       '后文与新模型的逐条比较以 **同一指定参考下重算的原始 DiariZen 预测**为基准；原始 33.75% 仅作历史记录。', '',
       '上述重评分还使用原仓库的 `ear1_baseline.py score` 独立执行，14 条逐条 DER 与本实验拆解脚本数值一致；官方脚本输出位于 `scores/official_check/`。'
       '本次使用的基线代码 SHA-256 见 [校验文件](config/baseline_code_sha256.txt)。', '',
       '## 模型与运行状态', '',
       '| 模型 | 状态 | 环境 | 组件 / 说明 |', '|---|---|---|---|']
labels = {'diarizen_baseline':'原始 DiariZen 预测重评', 'nemotron3':'NVIDIA Nemotron 3 Diarization',
          'pyannote_community1':'pyannote Community-1', 'funasr':'FunASR 官方流程'}
envs = {'diarizen_baseline':'diarizen（保留）', 'nemotron3':'ear1-nemotron',
        'pyannote_community1':'ear1-pyannote', 'funasr':'ear1-funasr'}
descriptions = {'diarizen_baseline':'现有完整预测 RTTM',
                'nemotron3':'官方 Transformers 流式接口；跨 chunk 保留说话人缓存；8 路输出',
                'pyannote_community1':'speaker_diarization 普通输出，保留重叠',
                'funasr':'SenseVoiceSmall ASR + FSMN-VAD 分割 + CAM++ 声纹 + FunASR 聚类；vad_segment；不提供人数先验'}
for m in models:
    count = len(summaries[m])
    state = status.get(m, f'已评分 {count}/14 条' if count else '尚无有效预测')
    out.append(f'| {labels[m]} | {state} | {envs[m]} | {descriptions[m]} |')
out.extend(['', '模型与依赖的完整版本、可获得 revision 见 [模型元数据](config/model_revisions.json) 和 `config/requirements_*.txt`。'
            'FunASR 三个 ModelScope `master` 权重文件的 SHA-256 见 [权重校验表](config/funasr_weight_sha256.txt)。'
            '模型权重的 `master` 名称本身不是不可变 revision，SHA-256 用于锁定本次实际文件。'
            'Community-1 最初因 401 受阻，用户已在 Hugging Face 接受条件并在服务器本机认证，随后全部 14 条运行成功；报告未保存令牌。', ''])
out.extend(['', '## 总体结果', '', '| 模型 | 轨道 | 已评分条数 | 参考说话人秒 | 漏检秒 | 误报秒 | 混淆秒 | 微平均 DER | 相对同参考 DiariZen |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
baseline_der = {}
for track in ('D1','R1'):
    group = [r for r in details['diarizen_baseline'] if r['track']==track and r['region']=='overall']
    total = sum(float(r['reference_speaker_s']) for r in group)
    error = sum(float(r[k]) for r in group for k in ('missed_detection_s','false_alarm_s','confusion_s'))
    baseline_der[track] = 100*error/total
for m in models:
    for track in ('D1','R1'):
        group = [r for r in details[m] if r['track']==track and r['region']=='overall']
        if not group: continue
        sums = {key:sum(float(r[key]) for r in group) for key in ('reference_speaker_s','missed_detection_s','false_alarm_s','confusion_s')}
        der = 100*(sums['missed_detection_s']+sums['false_alarm_s']+sums['confusion_s'])/sums['reference_speaker_s']
        delta = f'{der-baseline_der[track]:+.2f} pp' if len(group)==7 else '—'
        out.append(f'| {labels[m]} | {track} | {len(group)} | {num(sums["reference_speaker_s"])} | {num(sums["missed_detection_s"])} | {num(sums["false_alarm_s"])} | {num(sums["confusion_s"])} | {num(der,2,"%")} | {delta} |')
out.extend(['', '仅 7/7 条时该轨道数值才是完整开发集结果；部分结果不与完整基线直接比较。', '',
            '在相同指定参考下，Nemotron 是三套新方案中 DER 最低的：D1 比 DiariZen 低 1.05 个百分点，R1 高 1.30 个百分点。'
            'Community-1 在两轨分别高 1.98、3.05 个百分点；FunASR 分别高 19.29、15.31 个百分点。'
            '因此没有一套新方案在 D1、R1 两轨同时超过原始 DiariZen 预测。', '',
            '## 每条音频结果', '',
            '| 模型 | 场次 | 轨道 | 参考人数（标签） | 预测人数（标签） | 重叠音频秒 / 语音音频秒 | 重叠率 | 总体 DER | 单人段 DER | 重叠段 DER |',
            '|---|---|---|---|---|---:|---:|---:|---:|---:|'])
for m in models:
    for r in summaries[m]:
        out.append(f'| {labels[m]} | {r["session"]} | {r["track"]} | {r["reference_speakers"]} ({r["reference_labels"]}) | '
                   f'{r["predicted_speakers"]} ({r["predicted_labels"]}) | {num(r["overlap_audio_s"])} / {num(r["speech_audio_s"])} | '
                   f'{num(100*float(r["overlap_rate"]),2,"%")} | {num(r["overall_der_pct"],2,"%")} | '
                   f'{num(r["single_der_pct"],2,"%")} | {num(r["overlap_der_pct"],2,"%")} |')
out.extend(['', '## 逐条区域错误拆解', '',
            '误报、漏检、混淆的百分比是各自对本行区域 DER 的贡献，分母为该区域参考说话人秒数。'
            '静音行分母为零，只列误报秒数；该误报仍计入总体 DER。', '',
            '| 模型 | 场次 | 轨道 | 区域 | 音频秒 | 参考说话人秒 | 误报秒（贡献） | 漏检秒（贡献） | 混淆秒（贡献） | DER |',
            '|---|---|---|---|---:|---:|---:|---:|---:|---:|'])
for m in models:
    for r in details[m]:
        def part(key):
            p = r[key+'_contribution_pct']
            return num(r[key+'_s'])+(f' ({num(p,2,"%")})' if p else '')
        out.append(f'| {labels[m]} | {r["session"]} | {r["track"]} | {r["region"]} | {num(r["audio_time_s"])} | '
                   f'{num(r["reference_speaker_s"])} | {part("false_alarm")} | {part("missed_detection")} | {part("confusion")} | {num(r["der_pct"],2,"%")} |')
out.extend(['', '## `20260924-5` 重点分析', ''])
out.extend(['本场参考标注 D1/R1 各为 8 人，约 37.1% 的被标注语音音频时间属于至少两人重叠。'
            'FunASR 的 `vad_segment` 流程在本场两轨均未输出重叠区间，其重叠段漏检需结合这一输出能力解释。', ''])
baseline = {(r['track'],r['region']):r for r in details['diarizen_baseline'] if r['session']=='20260924-5'}
for track in ('D1','R1'):
    ref = next((r for r in summaries['diarizen_baseline'] if r['session']=='20260924-5' and r['track']==track),None)
    if not ref: continue
    out.append(f'### {track}')
    out.append('')
    available = [(m, next((r for r in summaries[m] if r['session']=='20260924-5' and r['track']==track),None)) for m in models[1:]]
    available = [(m,r) for m,r in available if r]
    if available:
        closest_gap = min(abs(int(r['predicted_speakers'])-int(ref['reference_speakers'])) for _,r in available)
        closest = '、'.join(labels[m] for m,r in available if abs(int(r['predicted_speakers'])-int(ref['reference_speakers']))==closest_gap)
        out.append(f'三个新模型中，预测人数最接近 8 人的是 {closest}（相差 {closest_gap} 人）。')
        out.append('')
    out.append(f'参考标注 {ref["reference_speakers"]} 人，原始 DiariZen 预测 {ref["predicted_speakers"]} 人；'
               f'单人段 DER {num(ref["single_der_pct"],2,"%")}，重叠段 DER {num(ref["overlap_der_pct"],2,"%")}。')
    out.append('')
    base_overall = baseline[(track,'overall')]
    names = {'false_alarm':'误报','missed_detection':'漏检','confusion':'说话人混淆'}
    base_major = max(names,key=lambda k:float(base_overall[k+'_s']))
    out.append(f'原始 DiariZen 本轨总体最大错误项为{names[base_major]}（{num(base_overall[base_major+"_s"])} 说话人秒）。')
    out.append('')
    for m in models[1:]:
        r = next((r for r in summaries[m] if r['session']=='20260924-5' and r['track']==track),None)
        if not r: continue
        b = baseline[(track,'overall')]
        d = next(x for x in details[m] if x['session']=='20260924-5' and x['track']==track and x['region']=='overall')
        errors = {key:float(d[key+'_s']) for key in ('false_alarm','missed_detection','confusion')}
        major = max(errors,key=errors.get)
        changes = '，'.join(f'{names[key]} {float(d[key+"_s"])-float(b[key+"_s"]):+.2f} 秒' for key in errors)
        out.append(f'- {labels[m]}：预测 {r["predicted_speakers"]} 人；单人段 {num(r["single_der_pct"],2,"%")}，'
                   f'重叠段 {num(r["overlap_der_pct"],2,"%")}；总体最大错误项为{names[major]}。相对原始 DiariZen 预测：{changes}。')
    out.append('')
out.extend(['### 相对原始 DiariZen 预测的错误变化', '',
            '以下差值均在相同指定参考及完整录音所得全局说话人匹配下计算；正数代表错误增加。', '',
            '| 轨道 | 新模型 | 区域 | 误报秒差 | 漏检秒差 | 混淆秒差 | DER 百分点差 |',
            '|---|---|---|---:|---:|---:|---:|'])
for track in ('D1','R1'):
    for m in models[1:]:
        for region in ('overall','single','overlap','silence'):
            b = next((r for r in details['diarizen_baseline'] if r['session']=='20260924-5' and r['track']==track and r['region']==region),None)
            d = next((r for r in details[m] if r['session']=='20260924-5' and r['track']==track and r['region']==region),None)
            if not b or not d: continue
            delta = lambda key: float(d[key])-float(b[key])
            der_delta = num(delta('der_pct'),2) if d['der_pct'] and b['der_pct'] else '—'
            out.append(f'| {track} | {labels[m]} | {region} | {delta("false_alarm_s"):+.2f} | '
                       f'{delta("missed_detection_s"):+.2f} | {delta("confusion_s"):+.2f} | {der_delta} |')
out.append('')
out.extend(['## 可复现文件', '',
            '- 转换及评分代码：[evaluate.py](code/evaluate.py)；三个音频推理脚本见 `code/`。',
            '- 合并后的 [逐条结果 CSV](scores/all_models_per_audio.csv) 与 [区域错误拆解 CSV](scores/all_models_error_breakdown.csv)；各模型独立 CSV 也在 `scores/`。',
            '- 预测 RTTM：`predictions/<模型>/<轨道>/<场次>.rttm`；参考 RTTM：`reference/<轨道>/<场次>.rttm`。',
            '- 环境安装、推理和问题日志：`logs/`；模型状态：`config/model_status.json`。', '',
            '## 用户待处理事项', '',
            '无。Community-1 权限问题已由用户在服务器本机完成认证并解决，三个模型均已完成 14/14 条推理与评分。', '',
            '## 官方资料', '',
            '- [Nemotron 3 模型卡](https://huggingface.co/nvidia/Nemotron-3-Diarization)',
            '- [pyannote Community-1 模型卡](https://huggingface.co/pyannote/speaker-diarization-community-1)',
            '- [FunASR 官方仓库](https://github.com/modelscope/FunASR)',
            '- [EAR-1 说话人日志基线](https://github.com/earchallenge/ear1-baselines/blob/main/diarization/README.md)', ''])
dest = ROOT / 'EAR1_TASK1_MODEL_COMPARISON.md'
dest.write_text('\n'.join(out), encoding='utf-8')
print(dest)
