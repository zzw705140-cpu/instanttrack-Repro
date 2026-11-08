#!/bin/bash
# Train one preset, then evaluate every checkpoint on all 40 clips.
#   scripts/train.sh <preset> <run_name> [extra train.py args...]
#   scripts/train.sh instanttrack it_s0 --seed 0 --max_hours 0.75
# Presets: supermimic, supermimic_ema, instanttrack (see scripts/presets.sh).
# Activate the Conda environment before running; use its Python for training and evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
source scripts/presets.sh
P=${1:?preset}; RUN=${2:?run name}; shift 2
[ -n "${PRESET[$P]+x}" ] || { echo "unknown preset $P (${!PRESET[*]})"; exit 1; }

# W&B 配置：默认上传训练指标；使用运行机器已有凭据，不在脚本中保存令牌。
# 可直接修改默认值，也可在启动命令前用同名环境变量覆盖。
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_DEBUG="${WANDB_DEBUG:-false}"          # true：调用日志接口但禁用上传
WANDB_PROJECT="${WANDB_PROJECT:-instanttrack}"
WANDB_ENTITY="${WANDB_ENTITY-zzw705140-}"     # 沿用 RoboForge；设为空则由 SDK 选择默认空间
WANDB_NAME="${WANDB_NAME:-$RUN}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-50}"      # 与本地 CSV 共用记录频率

for value in "$WANDB_ENABLED" "$WANDB_DEBUG"; do
  [[ "$value" == true || "$value" == false ]] || { echo "WANDB_ENABLED/WANDB_DEBUG must be true or false" >&2; exit 1; }
done
[[ "$WANDB_LOG_EVERY" =~ ^[1-9][0-9]*$ ]] || { echo "WANDB_LOG_EVERY must be a positive integer" >&2; exit 1; }
WANDB_ARGS=()
EVAL_WANDB_ARGS=()
if [[ "$WANDB_DEBUG" == true || "$WANDB_ENABLED" == true ]]; then
  if [[ "$WANDB_DEBUG" == true ]]; then
    WANDB_ARGS+=(--wandb_debug)
    EVAL_WANDB_ARGS+=(--wandb_debug)
  else
    WANDB_ARGS+=(--wandb)
    EVAL_WANDB_ARGS+=(--wandb)
  fi
  WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT" --wandb_name "$WANDB_NAME")
  [[ -z "$WANDB_ENTITY" ]] || WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
mkdir -p "logs/$RUN"
python -m instanttrack.train --log_dir "logs/$RUN" ${PRESET[$P]} --log_every "$WANDB_LOG_EVERY" "${WANDB_ARGS[@]}" "$@" 2>&1 | tee -a "logs/$RUN/stdout.log"
python tools/eval_batch.py --run_dir "logs/$RUN" --group 8 "${EVAL_WANDB_ARGS[@]}" 2>&1 | tee -a "logs/$RUN/eval_stdout.log"
