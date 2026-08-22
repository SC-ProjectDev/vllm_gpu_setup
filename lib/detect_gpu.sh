#!/usr/bin/env bash
# Map nvidia-smi output to a profile id. Prints "<profile> <count>". Exit 2 if unknown.
set -euo pipefail

NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
PROFILE_TABLE='Known GPUs -> profiles:
  RTX 5090                      -> 5090
  A100-SXM4-80GB / A100 80GB    -> a100-80
  H100 (80GB)                   -> h100-80
  H200                          -> h200
Fallback by VRAM: >=130GB h200, >=75GB h100-80, >=30GB 5090.
Override with PROFILE=<id> in .env.'

mapfile -t lines < <("$NVIDIA_SMI" --query-gpu=name,memory.total --format=csv,noheader)
if [[ ${#lines[@]} -eq 0 ]]; then
  echo "detect_gpu: nvidia-smi returned no GPUs" >&2
  exit 2
fi

first="${lines[0]}"
name="${first%%,*}"
mem_raw="${first#*,}"
mem_mib="$(echo "$mem_raw" | tr -dc '0-9')"
mem_gb=$(( mem_mib / 1024 ))
count="${#lines[@]}"

profile=""
case "$name" in
  *"RTX 5090"*)                       profile="5090" ;;
  *"A100-SXM4-80GB"*|*"A100 80GB"*)   profile="a100-80" ;;
  *"H200"*)                            profile="h200" ;;
  *"H100"*)                            profile="h100-80" ;;
esac

if [[ -z "$profile" ]]; then
  # Reject unknown A100 variants
  if [[ "$name" == *"A100"* ]]; then
    : # A100 but unknown variant, will be rejected below
  elif   (( mem_gb >= 130 )); then profile="h200"
  elif (( mem_gb >= 75 ));  then profile="h100-80"
  elif (( mem_gb >= 30 ));  then profile="5090"
  fi
fi

if [[ -z "$profile" ]]; then
  echo "detect_gpu: unknown GPU: $first" >&2
  echo "$PROFILE_TABLE" >&2
  exit 2
fi

echo "$profile $count"
