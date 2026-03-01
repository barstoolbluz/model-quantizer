#!/usr/bin/env bash
# quantize-llmc.sh — Quantize models using vLLM's LLM Compressor (llmcompressor).
#
# Output: compressed-tensors checkpoints that vLLM can load directly.
# Updated for behavior/docs as of 2026-02-27.
#
# Usage:
#   quantize-llmc [--json] <model-id-or-path> [scheme] [options]
#
# Schemes:
#   fp8       FP8 quantization (data-free; default)
#   gptq      W4A16 GPTQ (calibration-based)
#   w8a8      W8A8 SmoothQuant + GPTQ (calibration-based)
#   nvfp4     NVFP4 (calibration-based; inference has SM100 behavior notes)
#
# Options:
#   --fp8-scheme NAME        FP8 scheme: dynamic|block (default: dynamic)
#   --fp8-pathway NAME       FP8 pathway: oneshot|model_free (default: oneshot)
#                            NOTE: model_free_ptq has no revision arg; local path required.
#   --ignore LIST            Extra ignore patterns (comma-separated; repeats OK)
#   --model-free-device STR  Device for model_free_ptq (default: cuda:0)
#   --model-free-workers N   Worker count for model_free_ptq (default: 8)
#
#   --num-samples N          Calibration samples (default: 512)
#   --seq-length N           Max sequence length (default: 2048)
#   --batch-size N           Calibration batch size (default: 1)
#   --dataset NAME           HF dataset id (default: open_platypus)
#   --dataset-config NAME    HF dataset config name (optional)
#   --dataset-path PATH      Local dataset input:
#                              - file: .json/.jsonl/.csv/.parquet
#                              - dir:  contains files of a single type above
#                              - dvc:  dvc://... (may contact remote)
#                            For local file/dir, this script loads it via datasets and
#                            passes a Dataset/DatasetDict to oneshot via dataset=...
#   --text-column KEY        Dataset text column (default: text)
#   --no-shuffle             Do not shuffle calibration samples
#   --seed N                 RNG seed (default: 1234)
#
#   --pipeline NAME          oneshot pipeline: basic|datafree|sequential|independent
#                            (default: llmcompressor's default)
#   --sequential-targets L   Comma list of decoder layer class names
#   --sequential-offload D   Offload device between sequential layers (default: cpu)
#   --no-qac                 Disable quantization-aware calibration
#
#   --streaming              Stream dataset (only for hub id or dvc:// path; online only)
#   --splits SPEC            Optional split percentages spec passed to oneshot (string)
#   --preprocessing-workers N  Dataset preprocessing workers (default: unset)
#   --dataloader-workers N     DataLoader workers (default: 0)
#
#   --model-revision REV     HF revision (default: main)
#   --use-auth-token         Use Hugging Face auth (for private repos) via oneshot(use_auth_token=True)
#   --suffix STR             Override output suffix
#   --force                  Overwrite existing output
#   --trust-remote-code      Allow custom model code
#   --online                 Allow network access (default: offline)
#
#   --lock-timeout SECONDS   Lock wait time for concurrent runs:
#                              0  = fail fast if locked (default)
#                              >0 = wait up to SECONDS
#                              -1 = wait without limit
#
#   --validate               Load output in vLLM and run checks (runs in a separate process)
#   --validate-prompt TEXT   Prompt for smoke test (default: "Hello!")
#   --validate-suite PATH    JSONL or txt prompts for a small regression suite
#                            JSONL schema per line (all optional):
#                              {"prompt": "...", "max_tokens": 64, "min_chars": 1, "must_contain": ["..."]}
#   --validate-seed N        Seed for vLLM sampling (default: 1)
#   --validate-max-tokens N  Max tokens per prompt (default: 64)
#   --validate-min-chars N   Minimal chars in output to count as pass (default: 1)
#
#   --log-dir PATH           llmcompressor log dir (optional)
#   --json                    JSON output mode (progress to stderr, single JSON object on stdout)
#
# Environment (optional):
#   MODEL_CACHE_DIR          HF cache root (default: ./models)
#   QUANTIZED_OUTPUT_DIR     Output root (default: $MODEL_CACHE_DIR)
#   WRITE_LOCAL_REPO_LAYOUT  1=repo-like layout under $OUTPUT_DIR/hub/models--.../snapshots/<hash>
#                            and refs/main=<hash>. Default: 0
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
  local s="$1"
  s="${s%/}"
  if [[ -d "$s" || -f "$s" ]]; then
    s="$(basename "$s")"
  fi
  echo "$s" | tr "/" "--"
}

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
MODEL_ID=""
SCHEME="fp8"

FP8_SCHEME="dynamic"      # dynamic|block -> FP8_DYNAMIC|FP8_BLOCK
FP8_PATHWAY="oneshot"     # oneshot|model_free
IGNORE_EXTRA=""           # comma-separated; may repeat
MODEL_FREE_DEVICE="cuda:0"
MODEL_FREE_WORKERS=8

NUM_SAMPLES=512
SEQ_LENGTH=2048
BATCH_SIZE=1
DATASET="open_platypus"
DATASET_CONFIG=""
DATASET_PATH=""
TEXT_COLUMN="text"
SHUFFLE=1
SEED=1234

PIPELINE=""               # empty => let llmcompressor choose default
SEQUENTIAL_TARGETS=""     # comma-separated list => python list[str]
SEQUENTIAL_OFFLOAD="cpu"
QAC=1

STREAMING=0
SPLITS_SPEC=""
PREPROCESSING_WORKERS=""  # unset => llmcompressor default
DATALOADER_WORKERS=0

MODEL_REVISION="main"
USE_AUTH_TOKEN=0
SUFFIX=""
FORCE=0
TRUST_REMOTE_CODE=0
ONLINE=0

LOCK_TIMEOUT=0            # 0 fail-fast, >0 wait seconds, -1 wait without limit

VALIDATE=0
VALIDATE_PROMPT="Hello!"
VALIDATE_SUITE=""
VALIDATE_SEED=1
VALIDATE_MAX_TOKENS=64
VALIDATE_MIN_CHARS=1

LOG_DIR=""
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
    --fp8-scheme)           require_value "$1" "$#"; FP8_SCHEME="$2"; shift 2 ;;
    --fp8-scheme=*)         FP8_SCHEME="${1#*=}"; shift ;;
    --fp8-pathway)          require_value "$1" "$#"; FP8_PATHWAY="$2"; shift 2 ;;
    --fp8-pathway=*)        FP8_PATHWAY="${1#*=}"; shift ;;
    --ignore)               require_value "$1" "$#"; IGNORE_EXTRA="${IGNORE_EXTRA:+$IGNORE_EXTRA,}$2"; shift 2 ;;
    --ignore=*)             IGNORE_EXTRA="${IGNORE_EXTRA:+$IGNORE_EXTRA,}${1#*=}"; shift ;;

    --model-free-device)    require_value "$1" "$#"; MODEL_FREE_DEVICE="$2"; shift 2 ;;
    --model-free-device=*)  MODEL_FREE_DEVICE="${1#*=}"; shift ;;
    --model-free-workers)   require_value "$1" "$#"; MODEL_FREE_WORKERS="$2"; shift 2 ;;
    --model-free-workers=*) MODEL_FREE_WORKERS="${1#*=}"; shift ;;

    --num-samples)          require_value "$1" "$#"; NUM_SAMPLES="$2"; shift 2 ;;
    --num-samples=*)        NUM_SAMPLES="${1#*=}"; shift ;;
    --seq-length)           require_value "$1" "$#"; SEQ_LENGTH="$2"; shift 2 ;;
    --seq-length=*)         SEQ_LENGTH="${1#*=}"; shift ;;
    --batch-size)           require_value "$1" "$#"; BATCH_SIZE="$2"; shift 2 ;;
    --batch-size=*)         BATCH_SIZE="${1#*=}"; shift ;;
    --dataset)              require_value "$1" "$#"; DATASET="$2"; shift 2 ;;
    --dataset=*)            DATASET="${1#*=}"; shift ;;
    --dataset-config)       require_value "$1" "$#"; DATASET_CONFIG="$2"; shift 2 ;;
    --dataset-config=*)     DATASET_CONFIG="${1#*=}"; shift ;;
    --dataset-path)         require_value "$1" "$#"; DATASET_PATH="$2"; shift 2 ;;
    --dataset-path=*)       DATASET_PATH="${1#*=}"; shift ;;
    --text-column)          require_value "$1" "$#"; TEXT_COLUMN="$2"; shift 2 ;;
    --text-column=*)        TEXT_COLUMN="${1#*=}"; shift ;;
    --no-shuffle)           SHUFFLE=0; shift ;;
    --seed)                 require_value "$1" "$#"; SEED="$2"; shift 2 ;;
    --seed=*)               SEED="${1#*=}"; shift ;;

    --pipeline)             require_value "$1" "$#"; PIPELINE="$2"; shift 2 ;;
    --pipeline=*)           PIPELINE="${1#*=}"; shift ;;
    --sequential-targets)   require_value "$1" "$#"; SEQUENTIAL_TARGETS="$2"; shift 2 ;;
    --sequential-targets=*) SEQUENTIAL_TARGETS="${1#*=}"; shift ;;
    --sequential-offload)   require_value "$1" "$#"; SEQUENTIAL_OFFLOAD="$2"; shift 2 ;;
    --sequential-offload=*) SEQUENTIAL_OFFLOAD="${1#*=}"; shift ;;
    --no-qac)               QAC=0; shift ;;

    --streaming)            STREAMING=1; shift ;;
    --splits)               require_value "$1" "$#"; SPLITS_SPEC="$2"; shift 2 ;;
    --splits=*)             SPLITS_SPEC="${1#*=}"; shift ;;
    --preprocessing-workers) require_value "$1" "$#"; PREPROCESSING_WORKERS="$2"; shift 2 ;;
    --preprocessing-workers=*) PREPROCESSING_WORKERS="${1#*=}"; shift ;;
    --dataloader-workers)   require_value "$1" "$#"; DATALOADER_WORKERS="$2"; shift 2 ;;
    --dataloader-workers=*) DATALOADER_WORKERS="${1#*=}"; shift ;;

    --model-revision)       require_value "$1" "$#"; MODEL_REVISION="$2"; shift 2 ;;
    --model-revision=*)     MODEL_REVISION="${1#*=}"; shift ;;
    --use-auth-token)       USE_AUTH_TOKEN=1; shift ;;
    --suffix)               require_value "$1" "$#"; SUFFIX="$2"; shift 2 ;;
    --suffix=*)             SUFFIX="${1#*=}"; shift ;;
    --force)                FORCE=1; shift ;;
    --trust-remote-code)    TRUST_REMOTE_CODE=1; shift ;;
    --online)               ONLINE=1; shift ;;

    --lock-timeout)         require_value "$1" "$#"; LOCK_TIMEOUT="$2"; shift 2 ;;
    --lock-timeout=*)       LOCK_TIMEOUT="${1#*=}"; shift ;;

    --validate)             VALIDATE=1; shift ;;
    --validate-prompt)      require_value "$1" "$#"; VALIDATE_PROMPT="$2"; shift 2 ;;
    --validate-prompt=*)    VALIDATE_PROMPT="${1#*=}"; shift ;;
    --validate-suite)       require_value "$1" "$#"; VALIDATE_SUITE="$2"; shift 2 ;;
    --validate-suite=*)     VALIDATE_SUITE="${1#*=}"; shift ;;
    --validate-seed)        require_value "$1" "$#"; VALIDATE_SEED="$2"; shift 2 ;;
    --validate-seed=*)      VALIDATE_SEED="${1#*=}"; shift ;;
    --validate-max-tokens)  require_value "$1" "$#"; VALIDATE_MAX_TOKENS="$2"; shift 2 ;;
    --validate-max-tokens=*) VALIDATE_MAX_TOKENS="${1#*=}"; shift ;;
    --validate-min-chars)   require_value "$1" "$#"; VALIDATE_MIN_CHARS="$2"; shift 2 ;;
    --validate-min-chars=*) VALIDATE_MIN_CHARS="${1#*=}"; shift ;;

    --log-dir)              require_value "$1" "$#"; LOG_DIR="$2"; shift 2 ;;
    --log-dir=*)            LOG_DIR="${1#*=}"; shift ;;
    --json)                 JSON_MODE=1; shift ;;

    --) shift; break ;;
    --*) die "Unknown option: $1" ;;
    *)
      if [[ -z "$MODEL_ID" ]]; then
        MODEL_ID="$1"
      elif [[ "$1" =~ ^(fp8|gptq|w8a8|nvfp4)$ ]]; then
        SCHEME="$1"
      else
        die "Unexpected argument: $1"
      fi
      shift
      ;;
  esac
done

[[ -n "$MODEL_ID" ]] || die "Usage: quantize-llmc <model-id-or-path> [scheme] [options]"

case "$SCHEME" in
  fp8|gptq|w8a8|nvfp4) ;;
  *) die "Unknown scheme: $SCHEME (use fp8, gptq, w8a8, nvfp4)" ;;
esac

case "$FP8_SCHEME" in
  dynamic|block) ;;
  *) die "--fp8-scheme must be dynamic or block" ;;
esac

case "$FP8_PATHWAY" in
  oneshot|model_free) ;;
  *) die "--fp8-pathway must be oneshot or model_free" ;;
esac

[[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]] || die "--num-samples must be an integer"
[[ "$SEQ_LENGTH" =~ ^[0-9]+$ ]]  || die "--seq-length must be an integer"
[[ "$BATCH_SIZE" =~ ^[0-9]+$ ]]  || die "--batch-size must be an integer"
[[ "$SEED" =~ ^[0-9]+$ ]]        || die "--seed must be an integer"
[[ "$MODEL_FREE_WORKERS" =~ ^[0-9]+$ ]] || die "--model-free-workers must be an integer"
[[ "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] || die "--dataloader-workers must be an integer"
if [[ -n "$PREPROCESSING_WORKERS" ]]; then
  [[ "$PREPROCESSING_WORKERS" =~ ^[0-9]+$ ]] || die "--preprocessing-workers must be an integer"
fi
[[ "$LOCK_TIMEOUT" =~ ^-?[0-9]+$ ]] || die "--lock-timeout must be an integer (0, >0, or -1)"
[[ "$VALIDATE_SEED" =~ ^[0-9]+$ ]] || die "--validate-seed must be an integer"
[[ "$VALIDATE_MAX_TOKENS" =~ ^[0-9]+$ ]] || die "--validate-max-tokens must be an integer"
[[ "$VALIDATE_MIN_CHARS" =~ ^[0-9]+$ ]] || die "--validate-min-chars must be an integer"

# -----------------------------------------------------------------------------
# Paths / naming
# -----------------------------------------------------------------------------
CACHE_DIR="${MODEL_CACHE_DIR:-./models}"
OUTPUT_DIR="${QUANTIZED_OUTPUT_DIR:-$CACHE_DIR}"
WRITE_LAYOUT="${WRITE_LOCAL_REPO_LAYOUT:-0}"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

if [[ -z "$SUFFIX" ]]; then
  case "$SCHEME" in
    fp8)
      if [[ "$FP8_SCHEME" == "dynamic" ]]; then SUFFIX="-FP8-DYNAMIC"; else SUFFIX="-FP8-BLOCK"; fi
      ;;
    gptq)  SUFFIX="-W4A16-GPTQ" ;;
    w8a8)  SUFFIX="-W8A8-SQ-GPTQ" ;;
    nvfp4) SUFFIX="-NVFP4" ;;
  esac
fi

OUTPUT_MODEL_ID="${MODEL_ID}${SUFFIX}"
OUTPUT_SLUG="$(slugify "$OUTPUT_MODEL_ID")"

# Resolve commit hash from HF cache for deterministic snapshot paths.
# Direct filesystem lookup avoids Python imports and HF library overhead.
_SRC_CACHE_KEY="$(echo "$MODEL_ID" | sed 's|/|--|g')"
_SRC_REFS="$CACHE_DIR/hub/models--${_SRC_CACHE_KEY}/refs/${MODEL_REVISION}"
if [[ -f "$_SRC_REFS" ]]; then
  RESOLVED_COMMIT="$(head -1 "$_SRC_REFS" | tr -d '[:space:]')"
elif [[ "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  RESOLVED_COMMIT="$MODEL_REVISION"
else
  RESOLVED_COMMIT=""
fi

# Deterministic snapshot hash — same parameters produce the same hash,
# enabling the exists check to detect and skip prior identical runs.
# Bump the "v1" prefix to invalidate all cached outputs after major changes.
SNAP_REV="$(printf '%s' \
  "v1|${MODEL_ID}|${RESOLVED_COMMIT}|${MODEL_REVISION}|${SCHEME}" \
  "|${FP8_SCHEME}|${FP8_PATHWAY}|${SUFFIX}|${TRUST_REMOTE_CODE}" \
  "|${NUM_SAMPLES}|${SEQ_LENGTH}|${BATCH_SIZE}|${DATASET}" \
  "|${DATASET_CONFIG}|${DATASET_PATH}|${TEXT_COLUMN}|${SHUFFLE}" \
  "|${SEED}|${IGNORE_EXTRA}|${PIPELINE}|${SEQUENTIAL_TARGETS}" \
  "|${QAC}|${STREAMING}|${SPLITS_SPEC}" \
  | sha1sum | cut -d' ' -f1)"

if [[ "$WRITE_LAYOUT" == "1" ]]; then
  MODEL_CACHE_KEY="$(echo "$OUTPUT_MODEL_ID" | tr "/" "--")"
  FINAL_OUT="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/snapshots/${SNAP_REV}"
  REFS_DIR="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/refs"
else
  FINAL_OUT="$OUTPUT_DIR/${OUTPUT_SLUG}"
  REFS_DIR=""
fi

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
  abs_target="$(python3 - "$target" <<'PY'
import os,sys
print(os.path.realpath(sys.argv[1]))
PY
)"
  abs_root="$(python3 - "$OUTPUT_DIR" <<'PY'
import os,sys
print(os.path.realpath(sys.argv[1]))
PY
)"

  case "$abs_target" in
    "$abs_root"/*) ;;
    *) die "refusing to delete path outside output root: $abs_target" ;;
  esac
  rm -rf -- "$abs_target"
}

# -----------------------------------------------------------------------------
# Concurrency guard: lock per output slug, with timeout/fail-fast
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
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"

if [[ "$ONLINE" == "0" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

# Suppress noisy deprecation/future warnings from third-party libraries
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error

# Repro: set hash seed before Python starts.
export PYTHONHASHSEED="$SEED"

# Fast guard: streaming implies remote read; block in offline mode.
if [[ "$ONLINE" == "0" && "$STREAMING" == "1" ]]; then
  die "--streaming is not allowed in offline mode. Fix: drop --streaming or pass --online."
fi

# -----------------------------------------------------------------------------
# Output directory handling (final path) — checked before VRAM so that
# "exists" can short-circuit without needing GPU access.
# -----------------------------------------------------------------------------
if [[ -e "$FINAL_OUT" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    log "Removing existing output (--force): $FINAL_OUT"
    safe_rm_rf "$FINAL_OUT"
  else
    log "Output already exists: $FINAL_OUT"
    log "Use --force to overwrite."
    if [[ "$JSON_MODE" == "1" ]]; then
      python3 - "$MODEL_ID" "$SCHEME" "$FINAL_OUT" <<'PY'
import json, sys
model_id, scheme, out_path = sys.argv[1:4]
print(json.dumps({
  "status": "exists",
  "source": model_id,
  "scheme": scheme,
  "output_path": out_path,
}))
PY
    fi
    exit 0
  fi
fi

mkdir -p "$(dirname "$FINAL_OUT")"
if [[ "$WRITE_LAYOUT" == "1" ]]; then
  mkdir -p "$REFS_DIR"
fi

# VRAM check: verify enough free GPU memory before proceeding.
python3 -c "
import sys, torch
if torch.cuda.is_available():
    try:
        free, total = torch.cuda.mem_get_info(0)
        free_mb, total_mb = free >> 20, total >> 20
        if free_mb < 512:
            print(f'GPU has only {free_mb} MB free of {total_mb} MB. '
                  f'Another process may be using the GPU. '
                  f'Free GPU memory before quantizing.', file=sys.stderr)
            sys.exit(1)
    except SystemExit:
        raise
    except Exception:
        print('Failed to query GPU memory — VRAM may be fully consumed '
              'by another process. Free GPU memory before quantizing.', file=sys.stderr)
        sys.exit(1)
" || exit 1

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
log "=== LLM Compressor Quantization ==="
log "  Source:       $MODEL_ID"
log "  Scheme:       $SCHEME"
if [[ "$SCHEME" == "fp8" ]]; then
  log "  FP8 mode:     $FP8_SCHEME ($FP8_PATHWAY)"
fi
log "  Output:       $OUTPUT_MODEL_ID"
log "  Final out:    $FINAL_OUT"
log "  Tmp out:      $TMP_OUT"
log "  Cache:        $CACHE_DIR"
log "  Online:       $ONLINE"
log "  HF rev:       $MODEL_REVISION"
log "  Auth token:   $USE_AUTH_TOKEN"
if [[ "$SCHEME" != "fp8" ]]; then
  log "  Dataset:      ${DATASET_PATH:-$DATASET}"
  [[ -n "$DATASET_CONFIG" ]] && log "  DS cfg:       $DATASET_CONFIG"
  log "  Samples:      $NUM_SAMPLES"
  log "  Seq len:      $SEQ_LENGTH"
  log "  Batch:        $BATCH_SIZE"
  log "  Shuffle:      $SHUFFLE"
  log "  Seed:         $SEED"
  log "  Streaming:    $STREAMING"
  [[ -n "$SPLITS_SPEC" ]] && log "  Splits:       $SPLITS_SPEC"
  [[ -n "$PREPROCESSING_WORKERS" ]] && log "  Prep workers: $PREPROCESSING_WORKERS"
  log "  DL workers:   $DATALOADER_WORKERS"
fi
if [[ -n "$PIPELINE" ]]; then
  log "  Pipeline:     $PIPELINE"
fi
if [[ -n "$SEQUENTIAL_TARGETS" ]]; then
  log "  Seq targets:  $SEQUENTIAL_TARGETS"
  log "  Seq offload:  $SEQUENTIAL_OFFLOAD"
fi
log "  Lock:         $LOCK_FILE"
log "  Lock wait:    $LOCK_TIMEOUT"
log ""

# -----------------------------------------------------------------------------
# Quantize (Python) — quantization only (no vLLM load here)
# -----------------------------------------------------------------------------
export JSON_MODE
export LLMC_MODEL_ID="$MODEL_ID"
export LLMC_SCHEME="$SCHEME"
export LLMC_FP8_SCHEME="$FP8_SCHEME"
export LLMC_FP8_PATHWAY="$FP8_PATHWAY"
export LLMC_OUTPUT_DIR="$TMP_OUT"
export LLMC_TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE"
export LLMC_MODEL_REVISION="$MODEL_REVISION"
export LLMC_USE_AUTH_TOKEN="$USE_AUTH_TOKEN"
export LLMC_PIPELINE="$PIPELINE"
export LLMC_SEQUENTIAL_TARGETS="$SEQUENTIAL_TARGETS"
export LLMC_SEQUENTIAL_OFFLOAD="$SEQUENTIAL_OFFLOAD"
export LLMC_QAC="$QAC"
export LLMC_LOG_DIR="$LOG_DIR"

export LLMC_NUM_SAMPLES="$NUM_SAMPLES"
export LLMC_SEQ_LENGTH="$SEQ_LENGTH"
export LLMC_BATCH_SIZE="$BATCH_SIZE"
export LLMC_DATASET="$DATASET"
export LLMC_DATASET_CONFIG="$DATASET_CONFIG"
export LLMC_DATASET_PATH="$DATASET_PATH"
export LLMC_TEXT_COLUMN="$TEXT_COLUMN"
export LLMC_SHUFFLE="$SHUFFLE"
export LLMC_SEED="$SEED"

export LLMC_STREAMING="$STREAMING"
export LLMC_SPLITS_SPEC="$SPLITS_SPEC"
export LLMC_PREPROCESSING_WORKERS="$PREPROCESSING_WORKERS"
export LLMC_DATALOADER_WORKERS="$DATALOADER_WORKERS"

export LLMC_IGNORE_EXTRA="$IGNORE_EXTRA"
export LLMC_MODEL_FREE_DEVICE="$MODEL_FREE_DEVICE"
export LLMC_MODEL_FREE_WORKERS="$MODEL_FREE_WORKERS"

export LLMC_VALIDATE="$VALIDATE"

python3 <<'PY'
import json, os, sys, time, platform
from datetime import datetime, timezone
from pathlib import Path

json_mode = os.environ.get("JSON_MODE", "0") == "1"
out = sys.stderr if json_mode else sys.stdout

# In JSON mode, redirect sys.stdout to stderr so that any library
# (llmcompressor uses loguru) writing to stdout gets captured on stderr.
# The single JSON object is emitted by a later bash block (not this one).
if json_mode:
    import logging
    # stdlib logging: force all handlers to stderr
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    # loguru / direct stdout writes: replace sys.stdout itself
    sys.stdout = sys.stderr

def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
model_id = os.environ["LLMC_MODEL_ID"]
scheme = os.environ["LLMC_SCHEME"]
fp8_scheme_flag = os.environ.get("LLMC_FP8_SCHEME", "dynamic")
fp8_pathway = os.environ.get("LLMC_FP8_PATHWAY", "oneshot")
out_dir = os.environ["LLMC_OUTPUT_DIR"]
trust_remote = os.environ["LLMC_TRUST_REMOTE_CODE"] == "1"
model_revision = os.environ.get("LLMC_MODEL_REVISION") or "main"
use_auth_token = os.environ.get("LLMC_USE_AUTH_TOKEN", "0") == "1"

pipeline = (os.environ.get("LLMC_PIPELINE") or "").strip() or None
seq_targets_raw = (os.environ.get("LLMC_SEQUENTIAL_TARGETS") or "").strip()
seq_offload = (os.environ.get("LLMC_SEQUENTIAL_OFFLOAD") or "").strip() or "cpu"
qac = os.environ.get("LLMC_QAC", "1") == "1"
log_dir = (os.environ.get("LLMC_LOG_DIR") or "").strip() or None

num_samples = int(os.environ.get("LLMC_NUM_SAMPLES") or 512)
seq_len = int(os.environ.get("LLMC_SEQ_LENGTH") or 2048)
batch_size = int(os.environ.get("LLMC_BATCH_SIZE") or 1)
dataset = os.environ.get("LLMC_DATASET") or "open_platypus"
dataset_config = (os.environ.get("LLMC_DATASET_CONFIG") or "").strip() or None
dataset_path = (os.environ.get("LLMC_DATASET_PATH") or "").strip() or None
text_column = os.environ.get("LLMC_TEXT_COLUMN") or "text"
shuffle = os.environ.get("LLMC_SHUFFLE") == "1"
seed = int(os.environ.get("LLMC_SEED") or 1234)

streaming = os.environ.get("LLMC_STREAMING", "0") == "1"
splits_spec = (os.environ.get("LLMC_SPLITS_SPEC") or "").strip() or None
preprocessing_workers_raw = (os.environ.get("LLMC_PREPROCESSING_WORKERS") or "").strip()
preprocessing_workers = int(preprocessing_workers_raw) if preprocessing_workers_raw else None
dataloader_workers = int(os.environ.get("LLMC_DATALOADER_WORKERS") or 0)

ignore_extra = (os.environ.get("LLMC_IGNORE_EXTRA") or "").strip()
model_free_device = os.environ.get("LLMC_MODEL_FREE_DEVICE") or "cuda:0"
model_free_workers = int(os.environ.get("LLMC_MODEL_FREE_WORKERS") or 8)

validate_requested = os.environ.get("LLMC_VALIDATE") == "1"

offline = (os.environ.get("HF_HUB_OFFLINE") == "1") or (os.environ.get("TRANSFORMERS_OFFLINE") == "1")

# Backstop: streaming + offline should fail early with a clear message.
if offline and streaming:
    die("--streaming is not allowed in offline mode. Fix: drop --streaming or run with --online.")

# -----------------------------------------------------------------------------
# Local-vs-hub model detection (tightened)
# -----------------------------------------------------------------------------
def looks_like_model_path(p: str) -> bool:
    pp = Path(p)
    if not pp.exists():
        return False
    if pp.is_file():
        name = pp.name.lower()
        if name == "config.json":
            return True
        if name.endswith(".safetensors") or name.endswith(".bin") or name.endswith(".pt"):
            return True
        return False
    # directory: require evidence
    if (pp / "config.json").is_file():
        return True
    # weights shards or index
    if list(pp.glob("*.safetensors")):
        return True
    if list(pp.glob("*.bin")):
        return True
    if (pp / "model.safetensors.index.json").is_file():
        return True
    return False

is_local_model = looks_like_model_path(model_id)

# -----------------------------------------------------------------------------
# Dependency + version gating
# -----------------------------------------------------------------------------
def pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None

def parse_ver(v: str):
    try:
        from packaging.version import Version
        return Version(v)
    except Exception:
        # minimal fallback (no packaging)
        parts = []
        for p in v.split("."):
            digits = "".join(ch for ch in p if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

def ver_ge(v: str, minimum: str) -> bool:
    return parse_ver(v) >= parse_ver(minimum)

# Base requirements
required = [
    ("torch", "2.1.0", "pip install torch"),
    ("transformers", "4.45.0", "pip install -U transformers"),
    ("huggingface_hub", "0.22.0", "pip install -U huggingface_hub"),
    ("llmcompressor", "0.7.1", "pip install -U llmcompressor"),
]

# Conditional bump: model_free_ptq landed in llmcompressor v0.9.0
if scheme == "fp8" and fp8_pathway == "model_free":
    required = [(k, ("0.9.0" if k == "llmcompressor" else v), h) for (k, v, h) in required]

if scheme != "fp8":
    required.append(("datasets", "2.14.0", "pip install -U datasets"))

# If validation is requested, fail early if vLLM is missing
if validate_requested:
    required.append(("vllm", "0.6.0", "pip install -U vllm"))

missing = []
too_old = []
for pkg, minv, hint in required:
    v = pkg_version(pkg)
    if v is None:
        missing.append((pkg, hint))
    elif not ver_ge(v, minv):
        too_old.append((pkg, v, minv, hint))

if missing:
    lines = ["Missing dependencies:"]
    for pkg, hint in missing:
        lines.append(f"  - {pkg}  ({hint})")
    die("\n".join(lines))

if too_old:
    lines = ["Dependencies are present but older than this script expects:"]
    for pkg, v, minv, hint in too_old:
        lines.append(f"  - {pkg} {v}  (need >= {minv})  [{hint}]")
    die("\n".join(lines))

print("Python:", sys.version.split()[0], file=out)
print("Platform:", platform.platform(), file=out)
print("torch:", pkg_version("torch"), file=out)
print("transformers:", pkg_version("transformers"), file=out)
print("llmcompressor:", pkg_version("llmcompressor"), file=out)
if scheme != "fp8":
    print("datasets:", pkg_version("datasets"), file=out)
print("huggingface_hub:", pkg_version("huggingface_hub"), file=out)
if validate_requested:
    print("vllm:", pkg_version("vllm"), file=out)
print(file=out)

# -----------------------------------------------------------------------------
# Repro controls: seed before dataset/model operations
# -----------------------------------------------------------------------------
import random
random.seed(seed)
try:
    import numpy as np
    np.random.seed(seed)
except Exception:
    pass
try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
except Exception:
    pass
try:
    from transformers import set_seed as hf_set_seed
    hf_set_seed(seed)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Hardware messaging (FP8 thresholds match vLLM docs)
#   - FP8 compute: CC > 8.9
#   - FP8 weight-only W8A16 via Marlin: CC > 8.0 (Ampere)
# -----------------------------------------------------------------------------
cuda_cc = None
cuda_name = None
try:
    import torch
    if torch.cuda.is_available():
        cuda_cc = torch.cuda.get_device_capability()
        cuda_name = torch.cuda.get_device_name(0)
        cc_num = cuda_cc[0] * 10 + cuda_cc[1]
        print(f"CUDA device: {cuda_name} (compute capability {cuda_cc[0]}.{cuda_cc[1]})", file=out)

        if scheme == "fp8":
            if cc_num >= 90:
                print("FP8 note: CC >= 9.0; aligns with vLLM's FP8 compute support band (> 8.9).", file=out)
            elif cc_num >= 80:
                print("FP8 note: Ampere-class CC >= 8.0; FP8 checkpoints can run as weight-only W8A16 via Marlin.", file=out)
            else:
                print("FP8 note: CC < 8.0; FP8 behavior in vLLM is not in the documented paths.", file=out)
        elif scheme == "nvfp4":
            if cc_num >= 100:
                print("NVFP4 note: SM100-class GPU detected; activation quantization is available.", file=out)
            else:
                print("NVFP4 note: < SM100; activation quantization is not available, weight-only behavior only.", file=out)
        print(file=out)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Offline checks: model revision availability in cache (hub ids only)
# -----------------------------------------------------------------------------
if offline and (not is_local_model):
    try:
        from huggingface_hub import hf_hub_download
        _ = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=model_revision,
            local_files_only=True,
            token=True if use_auth_token else None,
        )
        print(f"Offline check: model revision '{model_revision}' is present in cache.", file=out)
        print(file=out)
    except Exception as e:
        die(
            "Offline mode is set, but the requested model revision is not available locally.\n"
            f"Model: {model_id}\nRevision: {model_revision}\n"
            f"Details: {e}\n"
            "Fix: pre-download this revision (run once with --online), or change --model-revision."
        )

# -----------------------------------------------------------------------------
# FP8 model_free rule: require local model path that looks like weights
# -----------------------------------------------------------------------------
if scheme == "fp8" and fp8_pathway == "model_free":
    if not is_local_model:
        die(
            "fp8 model_free pathway requires a local model path (a directory with config.json and safetensors, or weights files).\n"
            "Reason: model_free_ptq has no revision parameter, so --model-revision cannot be applied for hub IDs.\n"
            "Fix: pass a local snapshot directory, or switch to --fp8-pathway oneshot."
        )

# -----------------------------------------------------------------------------
# DVC preflight hint
# -----------------------------------------------------------------------------
if scheme != "fp8" and dataset_path and dataset_path.startswith("dvc://"):
    if offline:
        die(
            "Offline mode is set, but --dataset-path uses dvc://...\n"
            "DVC may contact remotes; run with --online or provide a local dataset file/dir."
        )
    import importlib.util, shutil
    has_dvc = (importlib.util.find_spec("dvc") is not None) or (shutil.which("dvc") is not None)
    if not has_dvc:
        die(
            "Dataset path uses dvc://..., but DVC is not installed.\n"
            "Fix: install DVC (for example: pip install dvc) plus any remote extras you use."
        )

# -----------------------------------------------------------------------------
# Dataset handling (local file/dir -> DatasetDict; validate column via DatasetDict)
# -----------------------------------------------------------------------------
loaded_dataset = None
dataset_mode = "hub"  # hub|local_loaded|dvc

if scheme != "fp8":
    if dataset_path:
        if dataset_path.startswith("dvc://"):
            dataset_mode = "dvc"
        else:
            from datasets import load_dataset, DatasetDict
            p = Path(dataset_path)
            if not p.exists():
                die(f"--dataset-path does not exist: {dataset_path}")

            def detect_files(dir_path: Path):
                exts = [".jsonl", ".json", ".csv", ".parquet"]
                found = {e: [] for e in exts}
                for e in exts:
                    found[e] = sorted([str(x) for x in dir_path.rglob(f"*{e}") if x.is_file()])
                nonempty = {e: v for e, v in found.items() if v}
                return nonempty

            if p.is_file():
                ext = p.suffix.lower()
                if ext in (".jsonl", ".json"):
                    loader = "json"
                elif ext == ".csv":
                    loader = "csv"
                elif ext == ".parquet":
                    loader = "parquet"
                else:
                    die("Unsupported file type for --dataset-path. Use .json/.jsonl/.csv/.parquet, or a directory containing one of those.")
                loaded_dataset = load_dataset(loader, data_files={"train": [str(p)]}, name=dataset_config, streaming=False)
            else:
                nonempty = detect_files(p)
                if not nonempty:
                    die("No .json/.jsonl/.csv/.parquet files found under --dataset-path directory.")
                if len(nonempty) > 1:
                    kinds = ", ".join(sorted(nonempty.keys()))
                    die(f"Mixed dataset file types found in directory ({kinds}). Use a directory with a single type.")
                ext, files = next(iter(nonempty.items()))
                if ext in (".json", ".jsonl"):
                    loader = "json"
                elif ext == ".csv":
                    loader = "csv"
                elif ext == ".parquet":
                    loader = "parquet"
                else:
                    die("Unsupported dataset directory contents.")
                loaded_dataset = load_dataset(loader, data_files={"train": files}, name=dataset_config, streaming=False)

            dataset_mode = "local_loaded"

            # Correct DatasetDict handling
            try:
                from datasets import DatasetDict
                if isinstance(loaded_dataset, DatasetDict):
                    ds_train = loaded_dataset["train"]
                else:
                    ds_train = loaded_dataset
                cols = set(ds_train.column_names)
                if text_column not in cols:
                    die(f"text column '{text_column}' not found in dataset columns: {sorted(cols)}")
            except Exception as e:
                die(f"Failed to validate dataset columns: {e}")

            if offline:
                print("Offline check: local dataset loaded OK.", file=out)
                print(file=out)

    else:
        if offline:
            try:
                import datasets as ds
                _ = ds.load_dataset(dataset, name=dataset_config, split="train", download_mode="reuse_dataset_if_exists")
                print("Offline check: dataset id is loadable from cache (train split).", file=out)
                print(file=out)
            except Exception as e:
                die(
                    "Offline mode is set, but the dataset is not available locally.\n"
                    f"Dataset: {dataset}\nConfig: {dataset_config or '(none)'}\n"
                    f"Details: {e}\n"
                    "Fix: pre-download the dataset (run once with --online), or pass --dataset-path to a local dataset."
                )

# -----------------------------------------------------------------------------
# Build recipe and run
# -----------------------------------------------------------------------------
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

fp8_scheme = "FP8_DYNAMIC" if fp8_scheme_flag == "dynamic" else "FP8_BLOCK"

ignore = ["lm_head"]
if ignore_extra:
    for part in ignore_extra.split(","):
        p = part.strip()
        if p:
            ignore.append(p)

seq_targets = None
if seq_targets_raw:
    seq_targets = [s.strip() for s in seq_targets_raw.split(",") if s.strip()]

common_oneshot = dict(
    model=model_id,
    trust_remote_code_model=trust_remote,
    model_revision=model_revision,
    use_auth_token=use_auth_token,
    output_dir=out_dir,
    log_dir=log_dir,
    quantization_aware_calibration=qac,
)
if pipeline is not None:
    common_oneshot["pipeline"] = pipeline
if seq_targets is not None:
    common_oneshot["sequential_targets"] = seq_targets
    common_oneshot["sequential_offload_device"] = seq_offload

t0 = time.time()

if scheme == "fp8":
    if fp8_pathway == "model_free":
        # Import exists only on newer llmcompressor; version gate already checked.
        from llmcompressor.entrypoints.model_free import model_free_ptq
        model_free_ptq(
            model_stub=model_id,
            save_directory=out_dir,
            scheme=fp8_scheme,
            ignore=ignore,
            max_workers=model_free_workers,
            device=model_free_device,
        )
    else:
        recipe = QuantizationModifier(targets="Linear", scheme=fp8_scheme, ignore=ignore)
        oneshot(recipe=recipe, **common_oneshot)

else:
    if scheme == "gptq":
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)
    elif scheme == "w8a8":
        recipe = [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(targets="Linear", scheme="W8A8", ignore=ignore),
        ]
    elif scheme == "nvfp4":
        recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)
    else:
        die(f"Unknown scheme: {scheme}")

    kwargs = dict(
        recipe=recipe,
        num_calibration_samples=num_samples,
        max_seq_length=seq_len,
        batch_size=batch_size,
        shuffle_calibration_samples=shuffle,
        text_column=text_column,
        dataloader_num_workers=dataloader_workers,
        **common_oneshot,
    )
    if preprocessing_workers is not None:
        kwargs["preprocessing_num_workers"] = preprocessing_workers

    if dataset_mode in ("hub", "dvc"):
        kwargs["streaming"] = bool(streaming)
        if splits_spec:
            kwargs["splits"] = splits_spec

    # Dataset routing: avoid dataset + dataset_path combos.
    if dataset_mode == "local_loaded":
        kwargs["dataset"] = loaded_dataset
    elif dataset_mode == "dvc":
        kwargs["dataset_path"] = dataset_path
        if dataset_config is not None:
            kwargs["dataset_config_name"] = dataset_config
    else:
        kwargs["dataset"] = dataset
        if dataset_config is not None:
            kwargs["dataset_config_name"] = dataset_config

    oneshot(**kwargs)

elapsed = time.time() - t0

meta = {
    "model": model_id,
    "scheme": scheme,
    "fp8_scheme": fp8_scheme if scheme == "fp8" else None,
    "fp8_pathway": fp8_pathway if scheme == "fp8" else None,
    "model_revision": model_revision,
    "use_auth_token": use_auth_token,
    "output_dir": out_dir,
    "pipeline": pipeline,
    "sequential_targets": seq_targets,
    "sequential_offload_device": seq_offload if seq_targets else None,
    "quantization_aware_calibration": qac,
    "dataset_args": None if scheme == "fp8" else {
        "dataset_mode": dataset_mode,
        "dataset": (dataset_path or dataset),
        "dataset_config": dataset_config,
        "num_samples": num_samples,
        "max_seq_length": seq_len,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "seed": seed,
        "text_column": text_column,
        "streaming": bool(streaming) if dataset_mode in ("hub", "dvc") else False,
        "splits": splits_spec if dataset_mode in ("hub", "dvc") else None,
        "preprocessing_num_workers": preprocessing_workers,
        "dataloader_num_workers": dataloader_workers,
    },
    "cuda": {"name": cuda_name, "capability": list(cuda_cc) if cuda_cc else None},
    "versions": {
        "torch": pkg_version("torch"),
        "transformers": pkg_version("transformers"),
        "huggingface_hub": pkg_version("huggingface_hub"),
        "datasets": pkg_version("datasets") if scheme != "fp8" else None,
        "llmcompressor": pkg_version("llmcompressor"),
    },
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "elapsed_seconds": round(elapsed, 3),
}
Path(out_dir).mkdir(parents=True, exist_ok=True)
with open(os.path.join(out_dir, "quantize_info.json"), "w") as f:
    json.dump(meta, f, indent=2, sort_keys=True)

print(f"Quantization finished in {elapsed:.1f}s", file=out)
print(f"Wrote: {os.path.join(out_dir, 'quantize_info.json')}", file=out)
print(file=out)
PY

# -----------------------------------------------------------------------------
# Output sanity check (before validation/move)
# -----------------------------------------------------------------------------
python3 - "$TMP_OUT" <<'PY'
import os, sys
from pathlib import Path

json_mode = os.environ.get("JSON_MODE", "0") == "1"
_out = sys.stderr if json_mode else sys.stdout

out_dir = Path(sys.argv[1])

def die(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)

if not out_dir.exists():
    die(f"Output dir missing: {out_dir}")

qinfo = out_dir / "quantize_info.json"
if not qinfo.is_file():
    die("quantize_info.json missing (quantization did not finish cleanly).")

cfg = out_dir / "config.json"
if not cfg.is_file():
    die("config.json missing in output.")

has_safetensors = any(out_dir.glob("*.safetensors")) or (out_dir / "model.safetensors.index.json").is_file()
has_bin = any(out_dir.glob("*.bin"))
if not (has_safetensors or has_bin):
    die("No weight files found (*.safetensors / *.bin / safetensors index).")

print("Output sanity check: OK", file=_out)
PY

# -----------------------------------------------------------------------------
# Validation (separate Python process; avoids VRAM residue from quantization)
# -----------------------------------------------------------------------------
VALIDATION_STATUS="skipped"
if [[ "$VALIDATE" == "1" ]]; then
  export LLMC_VALIDATE_MODEL_DIR="$TMP_OUT"
  export LLMC_VALIDATE_TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE"
  export LLMC_VALIDATE_PROMPT="$VALIDATE_PROMPT"
  export LLMC_VALIDATE_SUITE="$VALIDATE_SUITE"
  export LLMC_VALIDATE_SEED="$VALIDATE_SEED"
  export LLMC_VALIDATE_MAX_TOKENS="$VALIDATE_MAX_TOKENS"
  export LLMC_VALIDATE_MIN_CHARS="$VALIDATE_MIN_CHARS"

  python3 <<'PY'
import os, sys

json_mode = os.environ.get("JSON_MODE", "0") == "1"
_out = sys.stderr if json_mode else sys.stdout

def die(msg: str, code: int = 3):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

model_dir = os.environ["LLMC_VALIDATE_MODEL_DIR"]
trust_remote = os.environ.get("LLMC_VALIDATE_TRUST_REMOTE_CODE") == "1"
prompt0 = os.environ.get("LLMC_VALIDATE_PROMPT") or "Hello!"
suite_path = (os.environ.get("LLMC_VALIDATE_SUITE") or "").strip() or None
seed = int(os.environ.get("LLMC_VALIDATE_SEED") or 1)
max_tokens_default = int(os.environ.get("LLMC_VALIDATE_MAX_TOKENS") or 64)
min_chars_default = int(os.environ.get("LLMC_VALIDATE_MIN_CHARS") or 1)

# Dependency check (separate process)
try:
    import importlib.metadata as md
    _ = md.version("vllm")
except Exception:
    die("vLLM not installed but --validate was set. Fix: pip install -U vllm", code=2)

from vllm import LLM, SamplingParams

def load_suite(path: str):
    prompts = []
    if path.endswith(".jsonl"):
        import json
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    prompts.append({"prompt": obj})
                else:
                    prompts.append(obj)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    prompts.append({"prompt": line})
    return prompts

cases = [{"prompt": prompt0}]
if suite_path:
    cases = load_suite(suite_path)

llm = LLM(model=model_dir, trust_remote_code=trust_remote)

failures = 0
exceptions = 0

for i, case in enumerate(cases, 1):
    prompt = case.get("prompt", "")
    if not isinstance(prompt, str) or not prompt:
        failures += 1
        print(f"[validate {i}] FAIL invalid prompt", file=_out)
        continue

    max_tokens = int(case.get("max_tokens", max_tokens_default))
    min_chars = int(case.get("min_chars", min_chars_default))
    must_contain = case.get("must_contain") or []
    if isinstance(must_contain, str):
        must_contain = [must_contain]

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
    )

    try:
        out = llm.generate([prompt], params)
        text = ""
        if out and out[0].outputs:
            text = out[0].outputs[0].text or ""
        text_stripped = text.strip()

        ok = True
        if len(text_stripped) < min_chars:
            ok = False
        for needle in must_contain:
            if needle not in text:
                ok = False

        status = "PASS" if ok else "FAIL"
        print(f"[validate {i}] {status}  prompt_len={len(prompt)}  out_len={len(text_stripped)}", file=_out)

        if not ok:
            failures += 1
            snippet = (text_stripped[:500] + ("..." if len(text_stripped) > 500 else ""))
            print("  output:", snippet, file=_out)

    except Exception as e:
        exceptions += 1
        failures += 1
        print(f"[validate {i}] FAIL exception: {type(e).__name__}: {e}", file=_out)

print(f"Validation summary: {len(cases)-failures} pass, {failures} fail ({exceptions} exceptions)", file=_out)
if failures:
    die("Validation failed (see per-case results above).", code=3)

print("Validation passed.", file=_out)
PY

  VALIDATION_STATUS="pass"
fi

# -----------------------------------------------------------------------------
# Finalize output (atomic move into FINAL_OUT)
# -----------------------------------------------------------------------------
if [[ "$WRITE_LAYOUT" == "1" ]]; then
  mkdir -p "$REFS_DIR"
  echo "$SNAP_REV" > "$REFS_DIR/main"
fi

if [[ -e "$FINAL_OUT" ]]; then
  safe_rm_rf "$FINAL_OUT"
fi

mv "$TMP_OUT" "$FINAL_OUT"
TMP_OUT=""

# -----------------------------------------------------------------------------
# Size stats + machine-readable summary line
# -----------------------------------------------------------------------------
if [[ "$JSON_MODE" == "1" ]]; then
  python3 - "$FINAL_OUT" "$VALIDATION_STATUS" <<'PY'
import json, os, sys, time
from pathlib import Path

out = Path(sys.argv[1])
status = sys.argv[2]

def dir_size(p: Path) -> int:
    total = 0
    for dp, _, fns in os.walk(p):
        for fn in fns:
            fp = Path(dp) / fn
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total

qinfo = out / "quantize_info.json"
meta = {}
if qinfo.is_file():
    meta = json.loads(qinfo.read_text())

size_bytes = dir_size(out)
elapsed = meta.get("elapsed_seconds")

summary = {
    "status": "ok",
    "source": meta.get("model"),
    "scheme": meta.get("scheme"),
    "output_path": str(out),
    "elapsed_seconds": elapsed,
    "validation": status,
    "size_bytes": size_bytes,
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
print(json.dumps(summary))
PY
else
  python3 - "$FINAL_OUT" "$VALIDATION_STATUS" <<'PY'
import json, os, sys, time
from pathlib import Path

out = Path(sys.argv[1])
status = sys.argv[2]

def dir_size(p: Path) -> int:
    total = 0
    for dp, _, fns in os.walk(p):
        for fn in fns:
            fp = Path(dp) / fn
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total

qinfo = out / "quantize_info.json"
meta = {}
if qinfo.is_file():
    meta = json.loads(qinfo.read_text())

size_bytes = dir_size(out)
elapsed = meta.get("elapsed_seconds")

print(f"Saved to: {out}")
print(f"Size:     {size_bytes/(1024**3):.2f} GB")

summary = {
    "source": meta.get("model"),
    "scheme": meta.get("scheme"),
    "output_path": str(out),
    "elapsed_seconds": elapsed,
    "validation": status,
    "size_bytes": size_bytes,
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
print(json.dumps(summary, sort_keys=True))
PY
  log "Done."
fi
