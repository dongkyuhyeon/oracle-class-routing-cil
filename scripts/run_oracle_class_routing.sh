#!/usr/bin/env bash
set -euo pipefail

RLO_ROOT="${1:-}"
GPU="${2:-0}"
TARGET="${3:-all}"

if [[ -z "$RLO_ROOT" ]]; then
    echo "Usage: bash scripts/run_oracle_class_routing.sh /path/to/R-LoRA [gpu] [e0|e1|e2|e3|e4|all]"
    exit 1
fi

PILOT_DIR="$RLO_ROOT/LAMDA-PILOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${RESULT_DIR:-$RLO_ROOT/results/oracle_class_routing}"

mkdir -p "$RESULT_DIR"
cd "$PILOT_DIR"

run_exp() {
    local config="$1"
    local logname="$2"
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') Starting $logname ====="
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True         "$PYTHON_BIN" main.py --config "./exps/${config}.json"         2>&1 | tee "$RESULT_DIR/${logname}.log"
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') Done $logname ====="
}

case "$TARGET" in
    e0) run_exp "oracle_ina_t20_e0_k1" "e0_k1" ;;
    e1) run_exp "oracle_ina_t20_e1_learned_top2" "e1_learned_top2" ;;
    e2) run_exp "oracle_ina_t20_e2_oracle_top2" "e2_oracle_top2" ;;
    e3) run_exp "oracle_ina_t20_e3_learned_top1" "e3_learned_top1" ;;
    e4) run_exp "oracle_ina_t20_e4_oracle_top1" "e4_oracle_top1" ;;
    all)
        run_exp "oracle_ina_t20_e1_learned_top2" "e1_learned_top2"
        run_exp "oracle_ina_t20_e2_oracle_top2" "e2_oracle_top2"
        run_exp "oracle_ina_t20_e3_learned_top1" "e3_learned_top1"
        run_exp "oracle_ina_t20_e4_oracle_top1" "e4_oracle_top1"
        run_exp "oracle_ina_t20_e0_k1" "e0_k1"
        ;;
    *)
        echo "Unknown target: $TARGET"
        exit 1
        ;;
esac

