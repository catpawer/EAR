import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from common import ROOT, SOURCE_MANIFEST


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


tf=ROOT/'weights'/'TF-GridNet'/'exp'/'enh_train_enh_tfgridnet_tf_lr-patience3_patience5_raw'/'98epoch.pth'
usef=ROOT/'weights'/'USEF-TSE'/'chkpt'/'USEF-TFGridNet'/'whamr!'/'temp_best.pth.tar'
revision=subprocess.check_output(['git','-C',str(ROOT/'code'/'USEF-TSE'),'rev-parse','HEAD'],text=True).strip()
meta={
    'experiment':'EAR-1 dev speech separation, 7 sessions x D1/R1',
    'source_manifest':str(SOURCE_MANIFEST),
    'reference_textgrid_root':'/data2/xujunxian/EAR1-Challenge/Dev',
    'tfgridnet':{
        'model':'TF-GridNet, WSJ0-2mix, single microphone, two sources, 8 kHz',
        'code':'ESPnet 202511, espnet2.enh.separator.tfgridnet_separator.TFGridNet',
        'original_model_url':'https://zenodo.org/records/7565926',
        'checkpoint':str(tf),'checkpoint_sha256':sha(tf),
        'env':'ear1-tfgridnet','python':'3.11',
        'inference':'6 s window, 1 s overlap, overlap correlation source alignment and crossfade; global least-squares mixture-consistency output scaling',
        'reference_used_for_inference':False,
    },
    'usef_tse':{
        'model':'USEF-TSE with TF-GridNet backbone, WHAMR! checkpoint, 8 kHz',
        'code_url':'https://github.com/ZBang/USEF-TSE',
        'code_revision':revision,
        'checkpoint_url':'https://huggingface.co/ZBang/USEF-TSE/tree/main/chkpt/USEF-TFGridNet/whamr!',
        'checkpoint':str(usef),'checkpoint_sha256':sha(usef),
        'env':'ear1-usef-tse','python':'3.11',
        'inference':'6 s window, 1 s overlap, crossfade; enrollment <=8 s selected from corrected reference TextGrid; tracks with PCM16 clipping peaks regenerated with shared track-level peak gain <=0.95',
        'reference_used_for_inference':True,
        'comparison_note':'Oracle enrollment: not comparable to audio-only diarization experiments',
    },
}
(ROOT/'config'/'model_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(ROOT/'config'/'model_metadata.json')
