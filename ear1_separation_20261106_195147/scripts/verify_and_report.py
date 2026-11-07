import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ROOT, rows


def stats(path):
    info = sf.info(path)
    total = 0; squares = 0.; peak = 0.
    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=800000, dtype='float32'):
            total += len(block)
            squares += float(np.sum(block.astype(np.float64)**2))
            peak = max(peak, float(np.max(np.abs(block))))
    return info.samplerate, info.frames, math.sqrt(squares/total) if total else 0., peak


def main():
    with (ROOT/'config'/'enrollment_manifest.csv').open(newline='') as f:
        enrollments=list(csv.DictReader(f))
    enrollment_by_key={(e['session'],e['track'],e['speaker']):e for e in enrollments}
    inventory=[]; failures=[]; usef_repaired=0; tf_correlations=[]
    for row in rows():
        expected=math.ceil(int(row['frames'])/6)
        sess=row['session']; track=row['track']
        for model in ['tfgridnet','usef_tse_whamr']:
            meta_path=ROOT/'results'/model/sess/f'{track}.json'
            if not meta_path.is_file():
                failures.append(f'missing metadata: {meta_path}'); continue
            meta=json.loads(meta_path.read_text())
            if model=='tfgridnet':
                tf_correlations.append(meta['mixture_reconstruction_corr'])
                entries=[(f'source{i+1}',p,'none') for i,p in enumerate(meta['outputs'])]
                if len(entries)!=2: failures.append(f'{sess} {track} TF-GridNet sources != 2')
            else:
                entries=[(t['speaker'],t['output'],t['enrollment_status']) for t in meta['tracks']]
                usef_repaired += sum(bool(t.get('clipping_repaired')) for t in meta['tracks'])
                expected_labels={e['speaker'] for e in enrollments if e['session']==sess and e['track']==track}
                if {x[0] for x in entries} != expected_labels:
                    failures.append(f'{sess} {track} USEF speaker labels mismatch')
            for label,path,enrollment_status in entries:
                if not Path(path).is_file():
                    failures.append(f'missing audio: {path}'); continue
                sr,frames,rms,peak=stats(path)
                if sr!=8000 or frames!=expected:
                    failures.append(f'{model} {sess} {track} {label}: {sr} Hz, {frames} frames, expected {expected}')
                inventory.append(dict(model=model,session=sess,track=track,label=label,path=path,
                                      sample_rate=sr,frames=frames,expected_frames=expected,
                                      duration_s=round(frames/sr,3),rms=round(rms,6),peak=round(peak,6),
                                      enrollment_status=enrollment_status))
    path=ROOT/'results'/'output_inventory.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(inventory[0])); w.writeheader(); w.writerows(inventory)
    summary={model:sum(x['model']==model for x in inventory) for model in ['tfgridnet','usef_tse_whamr']}
    lines=[
        '# EAR-1 开发集语音分离结果', '',
        '## 实验范围与方法', '',
        '开发集 7 场、D1/R1 共 14 条录音。原始音频为 48 kHz 单声道，均只读；输出为 8 kHz、16-bit PCM WAV，音频时长与输入相差不超过一个 8 kHz 采样点。', '',
        'TF-GridNet 使用作者公开的 WSJ0-2mix 单声道双声源权重，直接对混合音频推理。每条录音输出 source1、source2 两条音轨。分块重叠区按音频相关性对齐；为避免原始模型输出幅度过大导致 WAV 削顶，对两条输出统一使用与输入混合音最小二乘一致的增益。原始异常输出与日志单独保存在 `logs/invalid_gain_trial/`，不作为正式结果。source 编号不保证在长录音中始终对应同一人。', '',
        'USEF-TSE 使用作者公开的 USEF-TFGridNet WHAMR! 权重。每个参考说话人输出一条音轨。参考声音从修正后的 `/data2/xujunxian/EAR1-Challenge/Dev` TextGrid 所转换的参考 RTTM 中选择；已复核 14/14 当前 TextGrid 与转换后的 RTTM 逐项一致。因此这是**参考标注辅助的目标提取实验**，不能与纯音频日志系统公平比较。参考片段及 TextGrid SHA-256 见 `config/enrollment_manifest.csv`。', '',
        'P034 在 20260924-2 没有足够长的独讲片段，改用开发集另一场同标签独讲音频，这依赖跨场次标签指向同一人的假设；P039 在 20260924-5 无独讲片段，只能使用包含其他说话人的混合参考，结果身份可靠性较低。USEF 输出文件名中的 P 标签是条件目标，不代表其声音身份已人工核实。', '',
        '## 完整性', '',
        f'- TF-GridNet 音轨：{summary["tfgridnet"]}/28。',
        f'- USEF-TSE 音轨：{summary["usef_tse_whamr"]}/70。',
        f'- 文件、采样率与长度检查：{"通过" if not failures else "存在问题"}。',
        f'- USEF-TSE 重新导出以避免 PCM16 削顶的音轨：{usef_repaired} 条。',
        f'- 全部正式音轨最大绝对峰值：{max(float(x["peak"]) for x in inventory):.4f}。',
        f'- TF-GridNet 两路合成与原混合音的相关系数范围：{min(tf_correlations):.5f}–{max(tf_correlations):.5f}；仅作信号一致性检查。',
        f'- 逐音轨清单：`results/output_inventory.csv`。',
        '', '## 逐条录音输出', '',
        '| 场次 | 轨道 | TF-GridNet 音轨 | USEF-TSE 音轨 |',
        '|---|---|---:|---:|',
    ]
    for r in rows():
        sess=r['session']; track=r['track']
        a=sum(x['model']=='tfgridnet' and x['session']==sess and x['track']==track for x in inventory)
        b=sum(x['model']=='usef_tse_whamr' and x['session']==sess and x['track']==track for x in inventory)
        lines.append(f'| {sess} | {track} | {a} | {b} |')
    lines += ['', '## 解释和后续使用', '',
              '本数据只有混合录音与说话时间标注，没有逐人干净声源，不能在真实开发集直接计算 SI-SDRi。此处完成的是分离音轨生成和文件完整性验证；输入与 TF-GridNet 两路输出之和的高相关性仅是信号一致性检查，不能证明两人被正确分开。音质、串音和身份准确度需要听辨或另建带干净声源的测试混合音频。', '',
              'TF-GridNet 的双音轨可作为重叠段前端，再接声纹关联与 ASR；它本身不会自动给 4–8 名整场说话人建立稳定音轨。USEF-TSE 可用于研究有可靠目标声音时的重叠恢复，但 P039 的混合参考尤其需要人工听辨。', '',
              '对齐的试听示例位于 `review_clips/20260924-5/D1/`、`review_clips/20260924-5/R1/` 和 `review_clips/20260908-2/D1/`。每个目录包含原混合音、两路 TF-GridNet 输出与选定的 USEF-TSE 目标音轨。', '',
              '模型元数据见 `config/model_metadata.json`，运行脚本见 `scripts/run_tf_all.sh`、`scripts/run_usef_all.sh`，原始日志见 `logs/`。', '']
    if failures:
        lines += ['## 完整性问题',''] + [f'- {x}' for x in failures]
    (ROOT/'EAR1_DEV_SPEECH_SEPARATION.md').write_text('\n'.join(lines))
    print(json.dumps({'summary':summary,'failures':failures},ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
