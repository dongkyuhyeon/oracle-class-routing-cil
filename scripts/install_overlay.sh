#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RLO_ROOT="${1:-}"

if [[ -z "$RLO_ROOT" ]]; then
    echo "Usage: bash scripts/install_overlay.sh /path/to/R-LoRA"
    exit 1
fi

PILOT_DIR="$RLO_ROOT/LAMDA-PILOT"
TARGET_MODEL="$PILOT_DIR/models/crcl.py"

if [[ ! -f "$TARGET_MODEL" ]]; then
    echo "R-LoRA checkout not found: $TARGET_MODEL"
    exit 1
fi

BACKUP="$TARGET_MODEL.oracle-backup"
if [[ ! -f "$BACKUP" ]]; then
    cp "$TARGET_MODEL" "$BACKUP"
    echo "Backed up original model to $BACKUP"
fi

install -m 0644 "$EXP_ROOT/overlay/LAMDA-PILOT/models/crcl.py" "$TARGET_MODEL"
install -m 0644 "$EXP_ROOT/overlay/LAMDA-PILOT/utils/oracle_routing.py"     "$PILOT_DIR/utils/oracle_routing.py"

for config in "$EXP_ROOT"/overlay/LAMDA-PILOT/exps/*.json; do
    install -m 0644 "$config" "$PILOT_DIR/exps/$(basename "$config")"
done

echo "Oracle routing overlay installed into $PILOT_DIR"

