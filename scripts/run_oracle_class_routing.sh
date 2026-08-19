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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_DIR="${EXPERIMENT_RUNS_DIR:-$EXPERIMENT_REPO/experiments/runs}"

mkdir -p "$RUNS_DIR"
cd "$PILOT_DIR"

run_exp() {
    local config="$1"
    local logname="$2"
    local stamp
    local run_dir
    local source_sha
    local status

    stamp="$(date '+%Y%m%d_%H%M%S')"
    run_dir="$RUNS_DIR/${stamp}_${logname}_seed1993"
    mkdir -p "$run_dir/8_graphs"

    cp "./exps/${config}.json" "$run_dir/1_config.json"
    source_sha="$(git -C "$RLO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf '%s\n' "$source_sha" > "$run_dir/3_git_commit_sha.txt"
    printf '%s\n' '#!/usr/bin/env bash' > "$run_dir/2_command.sh"
    printf '%s\n' "CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN main.py --config ./exps/${config}.json" >> "$run_dir/2_command.sh"

    echo "===== $(date '+%Y-%m-%d %H:%M:%S') Starting $logname ====="
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" main.py --config "./exps/${config}.json" 2>&1 | tee "$run_dir/7_full.log"
    status=${PIPESTATUS[0]}
    set -e

    "$PYTHON_BIN" "$SCRIPT_DIR/collect_experiment_results.py" "$run_dir" --exit-code "$status"

    echo "===== $(date '+%Y-%m-%d %H:%M:%S') Done $logname ====="
    echo "Experiment record: $run_dir"
    return "$status"
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

