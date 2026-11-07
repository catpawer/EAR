# EAR

EAR-1 Task 1 diarization baselines, model comparisons, speech separation, and
the TF-GridNet / USEF-TSE fusion experiments.

| Directory | Purpose |
| --- | --- |
| `EAR1/` | Baseline code and existing experiment files. |
| `ear1_task1_compare_20261106_1712/` | DiariZen, Nemotron 3, pyannote Community-1, and FunASR comparison. |
| `ear1_separation_20261106_195147/` | TF-GridNet and reference-assisted USEF-TSE separation outputs. |
| `ear1_tfgridnet_fusion_20261107_1515/` | Audio-only TF-GridNet / DiariZen fusion and DER analysis. |
| `ear1_usef_task1_validation_20261107_1540/` | Audio-only USEF-TSE / DiariZen fusion and DER analysis. |

## Large files

Audio recordings and generated WAV tracks are intentionally excluded because
this repository is public. The source scripts, configuration, scores, RTTM,
TextGrid files, and reports remain available. Model checkpoints are stored
with Git LFS. After cloning, run `git lfs pull` to retrieve them. The 3.09 GB Whisper weight was split into two
LFS files so that each part fits GitHub's per-file LFS limit on personal plans.
To reconstruct it, run:

```bash
bash EAR1/baseline/asr/models/whisper-large-v3/restore_model.sh
```

The restore script verifies the original SHA-256. Do not commit the
reconstructed `model.safetensors` file; the two parts are the tracked copy.

Downloaded Hugging Face cache files, nested Git history, Python caches, and
failed trial WAVs under `logs/invalid_gain_trial/` are also omitted. Original
challenge audio under `/data2` is not part of this repository. Some experiment
output directories therefore contain only metadata or are absent; rerun the
scripts on authorized audio to regenerate their WAV files.

Many experiment scripts contain paths from the original server. Update those
paths and install the environment dependencies before rerunning elsewhere.
