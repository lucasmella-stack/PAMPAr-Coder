#!/usr/bin/env bash
# launch_phase6_ablation.sh — Lanza las 5 variantes × 3 seeds en GPU.
#
# Uso (dentro del container o en host con las deps):
#   bash scripts/launch_phase6_ablation.sh
#
# Override:
#   SEEDS="42 1337" CONFIGS="B_full B_no_loop" bash scripts/launch_phase6_ablation.sh
#
# Salida en ablation_results/phase6/<config>_seed<S>/
set -euo pipefail

CONFIGS="${CONFIGS:-A_baseline A_hier B_full B_no_loop B_act}"
SEEDS="${SEEDS:-42 1337 2024}"
CONFIG_DIR="${CONFIG_DIR:-configs/phase6_ablation}"

START=$(date +%s)
echo "═══ PAMPAr V4 — Phase 6 ablation ═══"
echo "Configs: $CONFIGS"
echo "Seeds:   $SEEDS"
echo "GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo

for cfg in $CONFIGS; do
    for seed in $SEEDS; do
        run_id="${cfg}_seed${seed}"
        echo "─── [$run_id] ───────────────────────────────"
        python scripts/train_v4_ablation.py \
            --config "${CONFIG_DIR}/${cfg}.yaml" \
            --seed "${seed}"
        echo
    done
done

ELAPSED=$(( $(date +%s) - START ))
echo "Total: ${ELAPSED}s"

echo
echo "═══ Análisis ═══"
python scripts/analyze_phase6_ablation.py
