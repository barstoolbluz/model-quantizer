#!/usr/bin/env bash
# quantize-gguf.sh — Convert HuggingFace models to GGUF format via llama.cpp.
#
# Two-phase pipeline:
#   1. convert_hf_to_gguf.py: HF safetensors → F16/BF16 GGUF (intermediate)
#   2. llama-quantize: F16 GGUF → quantized GGUF (Q4_K_M, Q5_K_S, etc.)
#
# Output: single .gguf file in HF-cache-like layout for llama.cpp ecosystem
# (ollama, LM Studio, koboldcpp, etc.).
#
# Usage:
#   quantize-gguf [--json] <model-id> [quant-type] [options]
#
# Positional:
#   model-id              HuggingFace model ID (e.g., Qwen/Qwen3-8B)
#   quant-type            GGUF quantization type (default: Q4_K_M)
#                         Supported: Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_1,
#                         Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K,
#                         Q8_0, F16, F32, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
#                         IQ3_XS, IQ3_S, IQ4_XS, IQ4_NL
#
# Options:
#   -c, --cache-dir DIR         HF cache root (default: $MODEL_CACHE_DIR)
#   -o, --output-dir DIR        Output root (default: $QUANTIZED_OUTPUT_DIR)
#   -r, --revision REV          HF revision (default: main)
#       --suffix STR            Override output suffix (default: -GGUF-<TYPE>)
#       --online                Allow network access
#       --trust-remote-code     Allow model repo custom code
#       --force                 Rebuild even if output exists
#       --imatrix FILE          Importance matrix for quantization
#       --convert-type TYPE     Intermediate precision: f16, bf16 (default: f16)
#       --no-cache-f16          Do not cache intermediate F16 GGUF
#       --threads N             Threads for llama-quantize (default: nproc)
#       --smoke-test            Load output and generate tokens
#       --smoke-prompt STR      Smoke test prompt (default: "Hello")
#       --smoke-tokens N        Smoke test token count (default: 8)
#       --lock-timeout N        Lock wait seconds (0=fail-fast, -1=unlimited, default: 0)
#       --json                  JSON output mode
#
# Environment (optional):
#   MODEL_CACHE_DIR          HF cache root (default: ./models)
#   QUANTIZED_OUTPUT_DIR     Output root (default: $MODEL_CACHE_DIR)
#   WRITE_LOCAL_REPO_LAYOUT  1=repo-like layout (default from manifest)
#   FLOX_ENV_CACHE           Flox cache root (used for F16 staging cache)
#
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

require_value() {
  local opt="$1"
  local remaining="$2"
  [[ "$remaining" -ge 2 ]] || die "Missing value for $opt"
}

have() { command -v "$1" >/dev/null 2>&1; }

log() {
  if [[ "$JSON_MODE" == "1" ]]; then
    echo "$*" >&2
  else
    echo "$*"
  fi
}

slugify() {
  echo "${1%/}" | sed 's|/|--|g'
}

# Supported GGUF quantization types.
VALID_QUANT_TYPES="Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_0 Q4_1 Q4_K_S Q4_K_M Q5_0 Q5_1 Q5_K_S Q5_K_M Q6_K Q8_0 F16 F32 IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_XS IQ3_S IQ4_XS IQ4_NL"

is_valid_quant_type() {
  local t="$1"
  for v in $VALID_QUANT_TYPES; do
    [[ "$v" == "$t" ]] && return 0
  done
  return 1
}

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
MODEL_ID=""
QUANT_TYPE="Q4_K_M"

REVISION="main"
SUFFIX=""
ONLINE=0
TRUST_REMOTE_CODE=0
FORCE=0
IMATRIX=""
CONVERT_TYPE="f16"
CACHE_F16=1
THREADS=""
SMOKE_TEST=0
SMOKE_PROMPT="Hello"
SMOKE_TOKENS=8
LOCK_TIMEOUT=0
JSON_MODE=0

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '/^# Usage:/,/^[^#]/{ /^#/s/^# \{0,1\}//p; }' "$0"
      exit 0
      ;;
    -c|--cache-dir)         require_value "$1" "$#"; MODEL_CACHE_DIR="$2"; shift 2 ;;
    --cache-dir=*)          MODEL_CACHE_DIR="${1#*=}"; shift ;;
    -o|--output-dir)        require_value "$1" "$#"; QUANTIZED_OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*)         QUANTIZED_OUTPUT_DIR="${1#*=}"; shift ;;
    -r|--revision)          require_value "$1" "$#"; REVISION="$2"; shift 2 ;;
    --revision=*)           REVISION="${1#*=}"; shift ;;
    --suffix)               require_value "$1" "$#"; SUFFIX="$2"; shift 2 ;;
    --suffix=*)             SUFFIX="${1#*=}"; shift ;;
    --online)               ONLINE=1; shift ;;
    --trust-remote-code)    TRUST_REMOTE_CODE=1; shift ;;
    --force)                FORCE=1; shift ;;
    --imatrix)              require_value "$1" "$#"; IMATRIX="$2"; shift 2 ;;
    --imatrix=*)            IMATRIX="${1#*=}"; shift ;;
    --convert-type)         require_value "$1" "$#"; CONVERT_TYPE="$2"; shift 2 ;;
    --convert-type=*)       CONVERT_TYPE="${1#*=}"; shift ;;
    --no-cache-f16)         CACHE_F16=0; shift ;;
    --threads)              require_value "$1" "$#"; THREADS="$2"; shift 2 ;;
    --threads=*)            THREADS="${1#*=}"; shift ;;
    --smoke-test)           SMOKE_TEST=1; shift ;;
    --smoke-prompt)         require_value "$1" "$#"; SMOKE_PROMPT="$2"; shift 2 ;;
    --smoke-prompt=*)       SMOKE_PROMPT="${1#*=}"; shift ;;
    --smoke-tokens)         require_value "$1" "$#"; SMOKE_TOKENS="$2"; shift 2 ;;
    --smoke-tokens=*)       SMOKE_TOKENS="${1#*=}"; shift ;;
    --lock-timeout)         require_value "$1" "$#"; LOCK_TIMEOUT="$2"; shift 2 ;;
    --lock-timeout=*)       LOCK_TIMEOUT="${1#*=}"; shift ;;
    --json)                 JSON_MODE=1; shift ;;
    --) shift; break ;;
    --*) die "Unknown option: $1" ;;
    *)
      if [[ -z "$MODEL_ID" ]]; then
        MODEL_ID="$1"
      elif is_valid_quant_type "$1"; then
        QUANT_TYPE="$1"
      else
        die "Unexpected argument: $1 (not a model ID or valid quant type)"
      fi
      shift
      ;;
  esac
done

[[ -n "$MODEL_ID" ]] || { sed -n '2,48s/^# \?//p' "$0"; die "Missing <model-id>"; }

is_valid_quant_type "$QUANT_TYPE" || die "Unknown quant type: $QUANT_TYPE (supported: $VALID_QUANT_TYPES)"

case "$CONVERT_TYPE" in
  f16|bf16) ;;
  *) die "--convert-type must be f16 or bf16" ;;
esac

if [[ -n "$THREADS" ]]; then
  [[ "$THREADS" =~ ^[0-9]+$ ]] || die "--threads must be a positive integer"
fi
[[ "$SMOKE_TOKENS" =~ ^[0-9]+$ ]] || die "--smoke-tokens must be a positive integer"
[[ "$LOCK_TIMEOUT" =~ ^-?[0-9]+$ ]] || die "--lock-timeout must be an integer (0, >0, or -1)"

if [[ -n "$IMATRIX" ]]; then
  [[ -f "$IMATRIX" ]] || die "--imatrix file not found: $IMATRIX"
fi

# -----------------------------------------------------------------------------
# Preflight: required tools
# -----------------------------------------------------------------------------
have convert_hf_to_gguf.py || die "convert_hf_to_gguf.py not found on PATH. Install llama-cpp."
have llama-quantize     || die "llama-quantize not found on PATH. Install llama-cpp."
if [[ "$SMOKE_TEST" == "1" ]]; then
  have llama-completion || die "llama-completion not found on PATH (needed for --smoke-test). Install llama-cpp."
fi

# Verify the gguf Python module is importable (needed by convert_hf_to_gguf.py).
python3 -c "import gguf" 2>/dev/null \
  || die "Python 'gguf' module not found. Run: uv pip install gguf"

# llama.cpp version (build number).
# Try llama-completion first (has --version), fall back to llama-quantize help output.
LLAMA_CPP_VERSION="$(llama-completion --version 2>&1 | grep -oP 'version:\s*b?\K[0-9]+' | head -1 2>/dev/null || true)"
if [[ -z "$LLAMA_CPP_VERSION" ]]; then
  LLAMA_CPP_VERSION="$(llama-quantize 2>&1 | grep -oP 'build\s*=\s*\K[0-9]+' | head -1 || echo "unknown")"
fi

# Thread count defaults to nproc.
if [[ -z "$THREADS" ]]; then
  THREADS="$(nproc 2>/dev/null || echo 4)"
fi

# -----------------------------------------------------------------------------
# Paths / naming
# -----------------------------------------------------------------------------
CACHE_DIR="${MODEL_CACHE_DIR:-./models}"
OUTPUT_DIR="${QUANTIZED_OUTPUT_DIR:-$CACHE_DIR}"
WRITE_LAYOUT="${WRITE_LOCAL_REPO_LAYOUT:-0}"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

if [[ -z "$SUFFIX" ]]; then
  SUFFIX="-GGUF-${QUANT_TYPE}"
fi

OUTPUT_MODEL_ID="${MODEL_ID}${SUFFIX}"
OUTPUT_SLUG="$(slugify "$OUTPUT_MODEL_ID")"

# Model short name for the output GGUF filename (e.g., Qwen3-8B from Qwen/Qwen3-8B).
MODEL_SHORT="$(basename "${MODEL_ID}")"

# Resolve commit hash from HF cache for deterministic snapshot paths.
_SRC_CACHE_KEY="$(echo "$MODEL_ID" | sed 's|/|--|g')"
_SRC_REFS="$CACHE_DIR/hub/models--${_SRC_CACHE_KEY}/refs/${REVISION}"
if [[ -f "$_SRC_REFS" ]]; then
  RESOLVED_COMMIT="$(head -1 "$_SRC_REFS" | tr -d '[:space:]')"
elif [[ "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  RESOLVED_COMMIT="$REVISION"
else
  RESOLVED_COMMIT=""
fi

# Source snapshot directory.
_SRC_SNAP=""
if [[ -n "$RESOLVED_COMMIT" ]]; then
  _SRC_SNAP="$CACHE_DIR/hub/models--${_SRC_CACHE_KEY}/snapshots/${RESOLVED_COMMIT}"
  if [[ ! -d "$_SRC_SNAP" ]]; then
    _SRC_SNAP=""
  fi
fi

if [[ -z "$_SRC_SNAP" && "$ONLINE" == "0" ]]; then
  die "Source model not found in cache: $MODEL_ID (revision: $REVISION). Use --online to download."
fi

# imatrix hash for fingerprint (empty string if not used).
IMATRIX_SHA256=""
if [[ -n "$IMATRIX" ]]; then
  IMATRIX_SHA256="$(sha256sum "$IMATRIX" | cut -d' ' -f1)"
fi

# Script SHA for fingerprint.
SCRIPT_SHA="$(sha256sum "$0" | cut -d' ' -f1)"

# -----------------------------------------------------------------------------
# Fingerprint — deterministic snapshot hash
# -----------------------------------------------------------------------------
FINGERPRINT_JSON="$(python3 - "$MODEL_ID" "$RESOLVED_COMMIT" "$QUANT_TYPE" "$CONVERT_TYPE" "$IMATRIX_SHA256" "$SCRIPT_SHA" "$LLAMA_CPP_VERSION" <<'PY'
import hashlib, json, sys
model_id, source_commit, quant_type, convert_type, imatrix_sha, script_sha, llama_ver = sys.argv[1:8]
fp = {
    "model_id": model_id,
    "source_commit": source_commit,
    "quant": {
        "scheme": "gguf",
        "type": quant_type,
        "convert_type": convert_type,
        "imatrix_sha256": imatrix_sha or None,
    },
    "build": {
        "script_sha256": script_sha,
        "llama_cpp_version": llama_ver,
    },
}
canonical = json.dumps(fp, sort_keys=True, separators=(",", ":"))
snap_id = hashlib.sha256(canonical.encode()).hexdigest()[:40]
# Print fingerprint JSON then snapshot ID on separate lines
print(json.dumps(fp, indent=2, sort_keys=True))
print(snap_id)
PY
)"

# Last line is the snapshot ID, everything before is the fingerprint JSON.
SNAP_REV="$(echo "$FINGERPRINT_JSON" | tail -1)"
FINGERPRINT_BODY="$(echo "$FINGERPRINT_JSON" | sed '$d')"

if [[ "$WRITE_LAYOUT" == "1" ]]; then
  MODEL_CACHE_KEY="$(echo "$OUTPUT_MODEL_ID" | sed 's|/|--|g')"
  FINAL_OUT="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/snapshots/${SNAP_REV}"
  REFS_DIR="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/refs"
else
  FINAL_OUT="$OUTPUT_DIR/${OUTPUT_SLUG}"
  REFS_DIR=""
fi

# GGUF output filename.
GGUF_FILENAME="${MODEL_SHORT}-${QUANT_TYPE}.gguf"

# -----------------------------------------------------------------------------
# Exists check — skip if complete output found
# -----------------------------------------------------------------------------
if [[ -e "$FINAL_OUT" ]]; then
  # Verify the output is complete: has the GGUF file + metadata.
  if [[ -f "$FINAL_OUT/$GGUF_FILENAME" && -f "$FINAL_OUT/FINGERPRINT.json" && -f "$FINAL_OUT/gguf_quantize_info.json" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      log "Existing output found; will rebuild (--force): $FINAL_OUT"
    else
      log "Output already exists: $FINAL_OUT"
      log "Use --force to overwrite."
      if [[ "$JSON_MODE" == "1" ]]; then
        _gguf_size=0
        [[ -f "$FINAL_OUT/$GGUF_FILENAME" ]] && _gguf_size="$(stat --printf='%s' "$FINAL_OUT/$GGUF_FILENAME" 2>/dev/null || echo 0)"
        python3 - "$MODEL_ID" "$QUANT_TYPE" "$FINAL_OUT" "$GGUF_FILENAME" "$_gguf_size" <<'PY'
import json, sys, time
model_id, quant_type, out_path, gguf_file, gguf_size = sys.argv[1:6]
print(json.dumps({
    "status": "exists",
    "source": model_id,
    "quant_type": quant_type,
    "output_path": out_path,
    "gguf_file": gguf_file,
    "gguf_size_bytes": int(gguf_size),
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}))
PY
      fi
      exit 0
    fi
  fi
fi

# -----------------------------------------------------------------------------
# Concurrency guard: flock per output slug (LLMC pattern)
# -----------------------------------------------------------------------------
if ! have flock; then
  die "flock not found. Install util-linux (or equivalent) for safe concurrent use."
fi
LOCK_FILE="$OUTPUT_DIR/.quantize-${OUTPUT_SLUG}.lock"
exec 9>"$LOCK_FILE"

if [[ "$LOCK_TIMEOUT" -eq 0 ]]; then
  if ! flock -n 9; then
    die "Lock is held: $LOCK_FILE (use --lock-timeout SECONDS to wait, or -1 to wait without limit)"
  fi
elif [[ "$LOCK_TIMEOUT" -lt 0 ]]; then
  flock 9
else
  if ! flock -w "$LOCK_TIMEOUT" 9; then
    die "Timed out waiting for lock after ${LOCK_TIMEOUT}s: $LOCK_FILE"
  fi
fi

# -----------------------------------------------------------------------------
# Offline / cache settings
# -----------------------------------------------------------------------------
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"

if [[ "$ONLINE" == "0" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# Suppress noisy warnings.
export PYTHONWARNINGS=ignore

# -----------------------------------------------------------------------------
# Temp directory + cleanup
# -----------------------------------------------------------------------------
TMP_ROOT="$OUTPUT_DIR/.tmp_quantize"
mkdir -p "$TMP_ROOT"
TMP_OUT="$(mktemp -d "$TMP_ROOT/${OUTPUT_SLUG}.XXXXXX")"

cleanup() {
  if [[ -n "${TMP_OUT:-}" && -d "$TMP_OUT" ]]; then
    rm -rf "$TMP_OUT" || true
  fi
}
trap cleanup EXIT

safe_rm_rf() {
  local target="$1"
  [[ -n "$target" ]] || die "internal: empty path"
  [[ "$target" != "/" ]] || die "refusing to delete /"

  local abs_target abs_root
  abs_target="$(python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$target")"
  abs_root="$(python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$OUTPUT_DIR")"

  case "$abs_target" in
    "$abs_root"/*) ;;
    *) die "refusing to delete path outside output root: $abs_target" ;;
  esac
  rm -rf -- "$abs_target"
}

# If --force and output exists, remove it now (after lock acquired).
if [[ "$FORCE" == "1" && -e "$FINAL_OUT" ]]; then
  log "Removing existing output (--force): $FINAL_OUT"
  safe_rm_rf "$FINAL_OUT"
fi

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
log "=== GGUF Quantization ==="
log "  Source:         $MODEL_ID"
log "  Quant type:     $QUANT_TYPE"
log "  Convert type:   $CONVERT_TYPE"
log "  Output:         $OUTPUT_MODEL_ID"
log "  GGUF file:      $GGUF_FILENAME"
log "  Final out:      $FINAL_OUT"
log "  Tmp out:        $TMP_OUT"
log "  Cache:          $CACHE_DIR"
log "  Online:         $ONLINE"
log "  HF rev:         $REVISION"
log "  Commit:         ${RESOLVED_COMMIT:-<pending>}"
log "  Threads:        $THREADS"
log "  llama.cpp:      b${LLAMA_CPP_VERSION}"
if [[ -n "$IMATRIX" ]]; then
  log "  imatrix:        $IMATRIX"
  log "  imatrix SHA:    $IMATRIX_SHA256"
fi
log "  Lock:           $LOCK_FILE"
log "  Lock wait:      $LOCK_TIMEOUT"
log "  F16 caching:    $CACHE_F16"
log ""

# -----------------------------------------------------------------------------
# Resolve source model path
# -----------------------------------------------------------------------------
# If source snapshot isn't resolved yet (e.g. online mode), resolve now.
if [[ -z "$_SRC_SNAP" ]]; then
  if [[ "$ONLINE" == "1" ]]; then
    log "Downloading model (online mode)..."
    python3 - "$MODEL_ID" "$REVISION" <<'PY'
import sys
model_id, revision = sys.argv[1], sys.argv[2]
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_files_only=False,
)
PY
    # Re-resolve after download.
    if [[ -f "$_SRC_REFS" ]]; then
      RESOLVED_COMMIT="$(head -1 "$_SRC_REFS" | tr -d '[:space:]')"
    fi
    _SRC_SNAP="$CACHE_DIR/hub/models--${_SRC_CACHE_KEY}/snapshots/${RESOLVED_COMMIT}"
  fi
  [[ -d "$_SRC_SNAP" ]] || die "Source snapshot not found: $_SRC_SNAP"
fi

log "Source snapshot: $_SRC_SNAP"
log ""

# If RESOLVED_COMMIT was empty before download, re-compute fingerprint.
if [[ "$(echo "$FINGERPRINT_BODY" | python3 -c "import json,sys; print(json.load(sys.stdin)['source_commit'])")" != "$RESOLVED_COMMIT" ]]; then
  FINGERPRINT_JSON="$(python3 - "$MODEL_ID" "$RESOLVED_COMMIT" "$QUANT_TYPE" "$CONVERT_TYPE" "$IMATRIX_SHA256" "$SCRIPT_SHA" "$LLAMA_CPP_VERSION" <<'PY'
import hashlib, json, sys
model_id, source_commit, quant_type, convert_type, imatrix_sha, script_sha, llama_ver = sys.argv[1:8]
fp = {
    "model_id": model_id,
    "source_commit": source_commit,
    "quant": {
        "scheme": "gguf",
        "type": quant_type,
        "convert_type": convert_type,
        "imatrix_sha256": imatrix_sha or None,
    },
    "build": {
        "script_sha256": script_sha,
        "llama_cpp_version": llama_ver,
    },
}
canonical = json.dumps(fp, sort_keys=True, separators=(",", ":"))
snap_id = hashlib.sha256(canonical.encode()).hexdigest()[:40]
print(json.dumps(fp, indent=2, sort_keys=True))
print(snap_id)
PY
)"
  SNAP_REV="$(echo "$FINGERPRINT_JSON" | tail -1)"
  FINGERPRINT_BODY="$(echo "$FINGERPRINT_JSON" | sed '$d')"

  # Recompute FINAL_OUT with updated snapshot ID.
  if [[ "$WRITE_LAYOUT" == "1" ]]; then
    FINAL_OUT="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/snapshots/${SNAP_REV}"
    REFS_DIR="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/refs"
  else
    FINAL_OUT="$OUTPUT_DIR/${OUTPUT_SLUG}"
  fi

  # Re-check exists with updated fingerprint.
  if [[ -e "$FINAL_OUT" && -f "$FINAL_OUT/$GGUF_FILENAME" && -f "$FINAL_OUT/FINGERPRINT.json" && -f "$FINAL_OUT/gguf_quantize_info.json" ]]; then
    if [[ "$FORCE" != "1" ]]; then
      log "Output already exists (after commit resolution): $FINAL_OUT"
      log "Use --force to overwrite."
      if [[ "$JSON_MODE" == "1" ]]; then
        _gguf_size="$(stat --printf='%s' "$FINAL_OUT/$GGUF_FILENAME" 2>/dev/null || echo 0)"
        python3 - "$MODEL_ID" "$QUANT_TYPE" "$FINAL_OUT" "$GGUF_FILENAME" "$_gguf_size" <<'PY'
import json, sys, time
model_id, quant_type, out_path, gguf_file, gguf_size = sys.argv[1:6]
print(json.dumps({
    "status": "exists",
    "source": model_id,
    "quant_type": quant_type,
    "output_path": out_path,
    "gguf_file": gguf_file,
    "gguf_size_bytes": int(gguf_size),
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}))
PY
      fi
      exit 0
    fi
    safe_rm_rf "$FINAL_OUT"
  fi
fi

# Timer start.
T_START="$(date +%s.%N)"

# -----------------------------------------------------------------------------
# Phase 1: F16 conversion (with caching)
# -----------------------------------------------------------------------------
F16_CACHE_DIR="${FLOX_ENV_CACHE:-$CACHE_DIR/.gguf-cache}/gguf-staging"
F16_SLUG="$(slugify "$MODEL_ID")"
F16_CACHE_PATH="$F16_CACHE_DIR/${F16_SLUG}/${RESOLVED_COMMIT}"
F16_GGUF_NAME="${MODEL_SHORT}-${CONVERT_TYPE^^}.gguf"
F16_GGUF=""

# Check cache for existing F16 conversion.
if [[ "$CACHE_F16" == "1" && -f "$F16_CACHE_PATH/$F16_GGUF_NAME" ]]; then
  log "Using cached F16 GGUF: $F16_CACHE_PATH/$F16_GGUF_NAME"
  F16_GGUF="$F16_CACHE_PATH/$F16_GGUF_NAME"
fi

if [[ -z "$F16_GGUF" ]]; then
  log "Converting HF model to ${CONVERT_TYPE^^} GGUF..."

  if [[ "$CACHE_F16" == "1" ]]; then
    # Write to cache location.
    mkdir -p "$F16_CACHE_PATH"
    F16_GGUF="$F16_CACHE_PATH/$F16_GGUF_NAME"
  else
    # Write to temp dir (not cached).
    F16_GGUF="$TMP_OUT/$F16_GGUF_NAME"
  fi

  CONVERT_ARGS=("$_SRC_SNAP" "--outfile" "$F16_GGUF" "--outtype" "$CONVERT_TYPE")
  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    if convert_hf_to_gguf.py --help 2>&1 | grep -q 'trust.remote.code'; then
      CONVERT_ARGS+=("--trust-remote-code")
    else
      log "  NOTE: convert_hf_to_gguf.py does not support --trust-remote-code; ignoring"
    fi
  fi

  log "  Running: convert_hf_to_gguf.py ${CONVERT_ARGS[*]}"
  if [[ "$JSON_MODE" == "1" ]]; then
    convert_hf_to_gguf.py "${CONVERT_ARGS[@]}" >&2
  else
    convert_hf_to_gguf.py "${CONVERT_ARGS[@]}"
  fi

  if [[ ! -f "$F16_GGUF" ]]; then
    die "convert_hf_to_gguf.py did not produce expected output: $F16_GGUF"
  fi

  _f16_size="$(stat --printf='%s' "$F16_GGUF" 2>/dev/null || echo 0)"
  _f16_size_gb="$(python3 -c "import sys; print(f'{int(sys.argv[1])/(1024**3):.2f}')" "$_f16_size")"
  log "  F16 GGUF: $F16_GGUF (${_f16_size_gb} GB)"
  log ""
fi

# -----------------------------------------------------------------------------
# Phase 2: Quantization
# -----------------------------------------------------------------------------
QUANT_OUTPUT="$TMP_OUT/$GGUF_FILENAME"

log "Quantizing ${CONVERT_TYPE^^} → $QUANT_TYPE..."

QUANT_ARGS=()
if [[ -n "$IMATRIX" ]]; then
  QUANT_ARGS+=("--imatrix" "$IMATRIX")
fi
QUANT_ARGS+=("$F16_GGUF" "$QUANT_OUTPUT" "$QUANT_TYPE")
QUANT_ARGS+=("$THREADS")

log "  Running: llama-quantize ${QUANT_ARGS[*]}"
if [[ "$JSON_MODE" == "1" ]]; then
  llama-quantize "${QUANT_ARGS[@]}" >&2
else
  llama-quantize "${QUANT_ARGS[@]}"
fi

if [[ ! -f "$QUANT_OUTPUT" ]]; then
  die "llama-quantize did not produce expected output: $QUANT_OUTPUT"
fi

GGUF_SIZE="$(stat --printf='%s' "$QUANT_OUTPUT" 2>/dev/null || echo 0)"
GGUF_SIZE_GB="$(python3 -c "import sys; print(f'{int(sys.argv[1])/(1024**3):.2f}')" "$GGUF_SIZE")"
log "  Quantized GGUF: $QUANT_OUTPUT (${GGUF_SIZE_GB} GB)"
log ""

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
echo "$FINGERPRINT_BODY" > "$TMP_OUT/FINGERPRINT.json"

T_END="$(date +%s.%N)"
ELAPSED="$(python3 -c "import sys; print(round(float(sys.argv[1]) - float(sys.argv[2]), 3))" "$T_END" "$T_START")"

export TMP_OUT
python3 - "$MODEL_ID" "$QUANT_TYPE" "$CONVERT_TYPE" "$GGUF_FILENAME" \
  "$GGUF_SIZE" "$ELAPSED" "$LLAMA_CPP_VERSION" "$RESOLVED_COMMIT" \
  "$IMATRIX" "$IMATRIX_SHA256" "$THREADS" <<'PY'
import json, sys, time, platform
model_id = sys.argv[1]
quant_type = sys.argv[2]
convert_type = sys.argv[3]
gguf_filename = sys.argv[4]
gguf_size = int(sys.argv[5])
elapsed = float(sys.argv[6])
llama_ver = sys.argv[7]
source_commit = sys.argv[8]
imatrix_path = sys.argv[9] or None
imatrix_sha = sys.argv[10] or None
threads = int(sys.argv[11])

meta = {
    "model_id": model_id,
    "source_commit": source_commit,
    "quant_type": quant_type,
    "convert_type": convert_type,
    "gguf_filename": gguf_filename,
    "gguf_size_bytes": gguf_size,
    "imatrix": imatrix_path,
    "imatrix_sha256": imatrix_sha,
    "threads": threads,
    "elapsed_seconds": elapsed,
    "llama_cpp_version": llama_ver,
    "platform": platform.platform(),
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

import os
out_dir = os.environ.get("TMP_OUT", ".")
with open(os.path.join(out_dir, "gguf_quantize_info.json"), "w") as f:
    json.dump(meta, f, indent=2, sort_keys=True)
PY

log "Wrote metadata to $TMP_OUT/gguf_quantize_info.json"

# Remove the intermediate F16 from TMP_OUT if it was written there (non-cached).
if [[ "$CACHE_F16" != "1" && -f "$TMP_OUT/$F16_GGUF_NAME" ]]; then
  rm -f "$TMP_OUT/$F16_GGUF_NAME"
fi

# -----------------------------------------------------------------------------
# Smoke test (optional)
# -----------------------------------------------------------------------------
SMOKE_RAN=0
SMOKE_OK=0

if [[ "$SMOKE_TEST" == "1" ]]; then
  log "Running smoke test..."
  SMOKE_RAN=1
  if _smoke_out="$(llama-completion -m "$QUANT_OUTPUT" -p "$SMOKE_PROMPT" -n "$SMOKE_TOKENS" -ngl 99 --no-display-prompt 2>/dev/null)"; then
    [[ "$JSON_MODE" != "1" ]] && [[ -n "$_smoke_out" ]] && echo "$_smoke_out"
    SMOKE_OK=1
    log "Smoke test: PASS"
  else
    log "WARNING: Smoke test failed (non-fatal)"
  fi
  log ""
fi

# -----------------------------------------------------------------------------
# Atomic publish
# -----------------------------------------------------------------------------
mkdir -p "$(dirname "$FINAL_OUT")"

if [[ "$WRITE_LAYOUT" == "1" ]]; then
  mkdir -p "$REFS_DIR"
  echo "$SNAP_REV" > "$REFS_DIR/main"
fi

if [[ -e "$FINAL_OUT" ]]; then
  safe_rm_rf "$FINAL_OUT"
fi

mv "$TMP_OUT" "$FINAL_OUT"
TMP_OUT=""

log "Published: $FINAL_OUT"
log ""

# -----------------------------------------------------------------------------
# JSON output / summary
# -----------------------------------------------------------------------------
T_FINAL="$(date +%s.%N)"
TOTAL_ELAPSED="$(python3 -c "import sys; print(round(float(sys.argv[1]) - float(sys.argv[2]), 3))" "$T_FINAL" "$T_START")"

if [[ "$JSON_MODE" == "1" ]]; then
  python3 - "$MODEL_ID" "$QUANT_TYPE" "$FINAL_OUT" "$GGUF_FILENAME" \
    "$GGUF_SIZE" "$TOTAL_ELAPSED" "$SMOKE_RAN" "$SMOKE_OK" <<'PY'
import json, sys, time
model_id = sys.argv[1]
quant_type = sys.argv[2]
out_path = sys.argv[3]
gguf_file = sys.argv[4]
gguf_size = int(sys.argv[5])
elapsed = float(sys.argv[6])
smoke_ran = sys.argv[7] == "1"
smoke_ok = sys.argv[8] == "1"

summary = {
    "status": "ok",
    "source": model_id,
    "quant_type": quant_type,
    "output_path": out_path,
    "gguf_file": gguf_file,
    "gguf_size_bytes": gguf_size,
    "elapsed_seconds": elapsed,
    "smoke_test": {"ran": smoke_ran, "ok": smoke_ok},
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
print(json.dumps(summary))
PY
else
  log "=== Done ==="
  log "  Source:     $MODEL_ID"
  log "  Type:       $QUANT_TYPE"
  log "  Output:     $FINAL_OUT"
  log "  GGUF:       $GGUF_FILENAME"
  log "  Size:       ${GGUF_SIZE_GB} GB"
  log "  Elapsed:    ${TOTAL_ELAPSED}s"
  if [[ "$SMOKE_RAN" == "1" ]]; then
    if [[ "$SMOKE_OK" == "1" ]]; then
      log "  Smoke test: PASS"
    else
      log "  Smoke test: FAIL"
    fi
  fi
fi
