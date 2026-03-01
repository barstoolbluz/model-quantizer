#!/usr/bin/env bash
# list-models.sh — Show models in the cache directory.

set -euo pipefail

CACHE_DIR="${MODEL_CACHE_DIR:-./models}"
HUB_DIR="$CACHE_DIR/hub"

echo "Model cache: $CACHE_DIR"
echo ""

if [ ! -d "$HUB_DIR" ]; then
    echo "  (no models cached)"
    exit 0
fi

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

    # Detect quantization type from config
    quant_type=""
    config="$model_dir/snapshots/main/config.json"
    if [ -f "$config" ]; then
        if grep -q '"quant_method.*compressed-tensors"' "$config" 2>/dev/null; then
            quant_type=" [compressed-tensors]"
        elif grep -q '"quant_method.*awq"' "$config" 2>/dev/null; then
            quant_type=" [AWQ]"
        elif grep -q '"quant_method.*torchao"' "$config" 2>/dev/null; then
            quant_type=" [FP8-TORCHAO]"
        elif grep -q '"quant_type"' "$config" 2>/dev/null; then
            quant_type=" [quantized]"
        elif grep -q '"quantization_config"' "$config" 2>/dev/null; then
            quant_type=" [quantized]"
        fi
    fi

    echo "  $model_id  ($size, $snapshot_count snapshot(s))$quant_type"
done

if [ $found -eq 0 ]; then
    echo "  (no models cached)"
fi
