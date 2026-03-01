#!/usr/bin/env bash
# list-models.sh — Show models in the cache directory.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: list-models [cache-dir]"
  echo ""
  echo "Lists HuggingFace models in the cache directory with size,"
  echo "snapshot count, and detected quantization type."
  echo ""
  echo "  cache-dir  HF cache root (default: \$MODEL_CACHE_DIR or ./models)"
  exit 0
fi

CACHE_DIR="${1:-${MODEL_CACHE_DIR:-./models}}"
HUB_DIR="$CACHE_DIR/hub"

echo "Model cache: $CACHE_DIR"
echo ""

if [ ! -d "$HUB_DIR" ]; then
    echo "  (no models cached)"
    exit 0
fi

# Detect quantization type from config.json in a snapshot directory.
# Snapshot dirs are hex hashes, not branch names — pick the first one found.
detect_quant_type() {
    local model_dir="$1"
    local snap_dir="$model_dir/snapshots"
    [ -d "$snap_dir" ] || return 0

    local config=""
    for snap in "$snap_dir"/*/; do
        [ -f "$snap/config.json" ] && config="$snap/config.json" && break
    done
    [ -n "$config" ] || return 0

    if grep -q '"quant_method.*compressed-tensors"' "$config" 2>/dev/null; then
        echo " [compressed-tensors]"
    elif grep -q '"quant_method.*awq"' "$config" 2>/dev/null; then
        echo " [AWQ]"
    elif grep -q '"quant_method.*torchao"' "$config" 2>/dev/null; then
        echo " [FP8-TORCHAO]"
    elif grep -q '"quant_type"' "$config" 2>/dev/null; then
        echo " [quantized]"
    elif grep -q '"quantization_config"' "$config" 2>/dev/null; then
        echo " [quantized]"
    fi
}

found=0
for model_dir in "$HUB_DIR"/models--*; do
    [ -d "$model_dir" ] || continue
    found=1

    dirname="$(basename "$model_dir")"
    model_id="$(echo "$dirname" | sed 's/^models--//; s/--/\//g')"

    size="$(du -sh "$model_dir" 2>/dev/null | cut -f1)"

    snapshot_count=0
    if [ -d "$model_dir/snapshots" ]; then
        snapshot_count="$(ls -1 "$model_dir/snapshots" 2>/dev/null | wc -l)"
    fi

    quant_type="$(detect_quant_type "$model_dir")"

    echo "  $model_id  ($size, $snapshot_count snapshot(s))$quant_type"
done

if [ $found -eq 0 ]; then
    echo "  (no models cached)"
fi
