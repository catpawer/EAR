#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/xujunxian/ear1_separation_20261106_195147
PY=/home/xujunxian/miniconda3/envs/ear1-tfgridnet/bin/python
pids=()
for gpu in 0 1 2 3; do
  "$PY" -u "$ROOT/scripts/run_tfgridnet.py" --gpu "$gpu" --shard "$gpu" --shards 4 >"$ROOT/logs/tfgridnet_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
