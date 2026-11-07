#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cat model.safetensors.part-aa model.safetensors.part-ab > model.safetensors.tmp
echo 'a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95  model.safetensors.tmp' | sha256sum -c -
mv model.safetensors.tmp model.safetensors
echo 'Restored model.safetensors'
