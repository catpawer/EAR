"""Run the author's 8 kHz single-microphone two-source TF-GridNet checkpoint."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from espnet2.enh.separator.tfgridnet_separator import TFGridNet

from common import ROOT, overlap_add, read_audio_8k, rows, save_wav, write_json

MODEL_DIR = ROOT / 'weights' / 'TF-GridNet' / 'exp' / 'enh_train_enh_tfgridnet_tf_lr-patience3_patience5_raw'


def load_model(device):
    conf = yaml.safe_load((MODEL_DIR/'config.yaml').read_text())
    assert conf['sample_rate'] == 8000
    assert conf['separator_conf']['n_imics'] == 1 and conf['separator_conf']['n_srcs'] == 2
    model = TFGridNet(input_dim=1, **conf['separator_conf'])
    raw = torch.load(MODEL_DIR/'98epoch.pth', map_location='cpu', weights_only=False)
    state = {k.removeprefix('separator.'): v for k, v in raw.items() if k.startswith('separator.')}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def normalized_dot(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b)/den) if den > 1e-6 else 0.0


def align(previous, current, overlap):
    if previous is None or overlap <= 0:
        return current, False, False
    old = previous[:, -overlap:]
    new = current[:, :overlap]
    energy = float(np.sqrt(np.mean(old**2) * np.mean(new**2)))
    if energy < 1e-6:
        return current, False, True
    keep = normalized_dot(old[0],new[0])+normalized_dot(old[1],new[1])
    swap = normalized_dot(old[0],new[1])+normalized_dot(old[1],new[0])
    if swap > keep + .15:
        return current[::-1].copy(), True, False
    return current, False, abs(swap-keep) < .15


def separate(model, audio, device, chunk_s=6, overlap_s=1):
    window = chunk_s*8000; step = (chunk_s-overlap_s)*8000
    starts = list(range(0, len(audio), step))
    chunks = []; swaps=0; ambiguous=0; skipped=0
    with torch.inference_mode():
        for start in starts:
            x = audio[start:start+window]
            if len(x) < window:
                x = np.pad(x, (0,window-len(x)))
            if float(np.max(np.abs(x))) < 0.0002:
                y = np.zeros((2,window),dtype=np.float32); skipped += 1
            else:
                speech = torch.from_numpy(x[None]).to(device)
                separated, _, _ = model(speech,torch.tensor([window],device=device))
                y = np.stack([z[0].float().cpu().numpy() for z in separated])
            overlap = max(0, chunks[-1].shape[1] + starts[len(chunks)-1] - start) if chunks else 0
            y, swapped, unclear = align(chunks[-1] if chunks else None,y,overlap)
            swaps += int(swapped); ambiguous += int(unclear)
            chunks.append(y)
    output = overlap_add(chunks, starts, len(audio))
    mixture_estimate = output.sum(axis=0)
    numerator = float(np.dot(audio.astype(np.float64), mixture_estimate.astype(np.float64)))
    denominator = float(np.dot(mixture_estimate.astype(np.float64), mixture_estimate.astype(np.float64)))
    scale = numerator / denominator if denominator > 1e-12 else 1.0
    if not 0 < scale < 100:
        raise RuntimeError(f'Invalid mixture reconstruction scale: {scale}')
    output *= scale
    reconstruction_corr = float(np.corrcoef(audio, output.sum(axis=0))[0, 1])
    return output,len(starts),swaps,ambiguous,skipped,scale,reconstruction_corr


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--shards',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(2)
    device=torch.device(f'cuda:{args.gpu}'); torch.cuda.set_device(device)
    model=load_model(device)
    for index,row in enumerate(rows()):
        if index % args.shards != args.shard: continue
        t0=time.time(); audio=read_audio_8k(row['audio'])
        separated,n_chunks,swaps,ambiguous,skipped,scale,reconstruction_corr=separate(model,audio,device)
        targets=[]
        for source in range(2):
            path=ROOT/'outputs'/'tfgridnet'/row['session']/row['track']/f'source{source+1}.wav'
            save_wav(path,separated[source]); targets.append(str(path))
        meta=dict(model='TF-GridNet-WSJ0-2mix',session=row['session'],track=row['track'],
                  source_audio=row['audio'],source_textgrid=row['reference_textgrid'],
                  sample_rate=8000,original_rate=48000,input_samples=len(audio),
                  output_samples=[len(x) for x in separated],outputs=targets,
                  chunks=n_chunks,swaps=swaps,ambiguous_alignments=ambiguous,
                  silent_chunks_skipped=skipped,output_scale=scale,
                  mixture_reconstruction_corr=reconstruction_corr,
                  elapsed_s=round(time.time()-t0,2),
                  note='source lanes are not guaranteed to be persistent speaker identities')
        write_json(ROOT/'results'/'tfgridnet'/row['session']/f"{row['track']}.json",meta)
        print(row['session'],row['track'],'chunks',n_chunks,'seconds',meta['elapsed_s'],flush=True)


if __name__=='__main__': main()
