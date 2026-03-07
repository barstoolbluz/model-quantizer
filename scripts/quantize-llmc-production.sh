#!/usr/bin/env bash
# quantize-llmc-hardened.sh — CI/prod-oriented wrapper for LLM Compressor quantization.
#
# Output: compressed-tensors checkpoints that vLLM can load directly.
#
# Main properties:
#   - strict shell mode + ERR trap
#   - lock-based concurrency control
#   - fingerprinted outputs
#   - atomic publish from temp dir
#   - artifact manifest + output validation
#   - optional validation via vLLM in a fresh Python process
#   - JSON and JSON-strict modes for automation
#
# Usage:
#   quantize-llmc-hardened [--json|--json-strict] <model-id-or-path> [scheme] [options]
#
# Schemes:
#   fp8       FP8 quantization (data-free; default)
#   gptq      W4A16 GPTQ (calibration-based)
#   w8a8      W8A8 SmoothQuant + GPTQ (calibration-based)
#   nvfp4     NVFP4 (calibration-based)
#
# Options:
#   --fp8-scheme NAME          FP8 scheme: dynamic|block (default: dynamic)
#   --fp8-pathway NAME         FP8 pathway: oneshot|model_free (default: oneshot)
#   --ignore LIST              Extra ignore patterns (comma-separated; repeats OK)
#   --model-free-device STR    Device for model_free_ptq (default: cuda:0)
#   --model-free-workers N     Worker count for model_free_ptq (default: 8)
#
#   --num-samples N            Calibration samples (default: 512)
#   --seq-length N             Max sequence length (default: 2048)
#   --batch-size N             Calibration batch size (default: 1)
#   --dataset NAME             HF dataset id (default: open_platypus)
#   --dataset-config NAME      HF dataset config name (optional)
#   --dataset-path PATH        Local dataset input or dvc:// path
#   --text-column KEY          Dataset text column (default: text)
#   --no-shuffle               Do not shuffle calibration samples
#   --seed N                   RNG seed (default: 1234)
#
#   --pipeline NAME            oneshot pipeline: basic|datafree|sequential|independent
#   --sequential-targets L     Comma list of decoder layer class names
#   --sequential-offload D     Offload device between sequential layers (default: cpu)
#   --no-qac                   Disable quantization-aware calibration
#
#   --streaming                Stream dataset (hub id or dvc:// only; online only)
#   --splits SPEC              Split percentages spec passed to oneshot
#   --preprocessing-workers N  Dataset preprocessing workers
#   --dataloader-workers N     DataLoader workers (default: 0)
#
#   --model-revision REV       HF revision (default: main)
#   --use-auth-token           Use Hugging Face auth
#   --suffix STR               Override output suffix
#   --force                    Rebuild even if a verified output exists
#   --trust-remote-code        Allow custom model code
#   --online                   Allow network access (default: offline)
#
#   --lock-timeout SECONDS     Lock wait time for concurrent runs:
#                                0  = fail fast if locked (default)
#                                >0 = wait up to SECONDS
#                                -1 = wait without limit
#
#   --validate                 Load output in vLLM and run checks
#   --validate-prompt TEXT     Prompt for smoke test (default: "Hello!")
#   --validate-suite PATH      JSONL or txt prompts for a small regression suite
#   --validate-seed N          Seed for vLLM sampling (default: 1)
#   --validate-max-tokens N    Max tokens per prompt (default: 64)
#   --validate-min-chars N     Minimal chars in output to count as pass (default: 1)
#
#   --log-dir PATH             llmcompressor log dir (optional)
#   --json                     JSON success/exists output; logs go to stderr
#   --json-strict              Like --json, and failures also emit JSON to stdout
#
# Environment (optional):
#   MODEL_CACHE_DIR          HF cache root (default: ./models)
#   QUANTIZED_OUTPUT_DIR     Output root (default: $MODEL_CACHE_DIR)
#   WRITE_LOCAL_REPO_LAYOUT  1=repo-style layout under $OUTPUT_DIR/hub/models--.../snapshots/<hash>
#                            and refs/<revision>=<hash>. Default: 0

set -Eeuo pipefail

JSON_MODE=0
JSON_STRICT=0
ERROR_EMITTED=0
CURRENT_STAGE="startup"

emit_error_json() {
  local msg="$1"
  local exit_code="${2:-1}"
  local stage="${CURRENT_STAGE:-unknown}"
  ERROR_EMITTED=1
  python3 - "$msg" "$exit_code" "$stage" <<'PY'
import json, sys, time
message, exit_code, stage = sys.argv[1], int(sys.argv[2]), sys.argv[3]
print(json.dumps({
    "status": "error",
    "error": {
        "message": message,
        "exit_code": exit_code,
        "stage": stage,
    },
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}))
PY
}

die() {
  local msg="$*"
  if [[ "${JSON_STRICT:-0}" == "1" ]]; then
    emit_error_json "$msg" 1
  else
    echo "ERROR: $msg" >&2
  fi
  exit 1
}

on_err() {
  local status=$?
  local lineno="${BASH_LINENO[0]:-0}"
  local cmd="${BASH_COMMAND:-unknown}"
  if [[ "$status" -ne 0 && "${JSON_STRICT:-0}" == "1" && "${ERROR_EMITTED:-0}" != "1" ]]; then
    emit_error_json "Unhandled error at line ${lineno}: ${cmd}" "$status"
  fi
  exit "$status"
}
trap on_err ERR

require_value() {
  local opt="$1"
  local remaining="$2"
  [[ "$remaining" -ge 2 ]] || die "Missing value for $opt"
}

have() { command -v "$1" >/dev/null 2>&1; }

log() {
  if [[ "${JSON_MODE:-0}" == "1" ]]; then
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
  printf '%s\n' "$s" | sed 's|/|--|g'
}

sha256_file() {
  local path="$1"
  if have sha256sum; then
    sha256sum "$path" | awk '{print $1}'
  elif have shasum; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    python3 - "$path" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
PY
  fi
}

python_module_version() {
  python3 - "$1" <<'PY'
import importlib.metadata, sys
name = sys.argv[1]
try:
    print(importlib.metadata.version(name))
except Exception:
    print("unknown")
PY
}

build_fingerprint_json() {
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" "${18}" "${19}" "${20}" "${21}" "${22}" "${23}" "${24}" "${25}" "${26}" "${27}" "${28}" "${29}" "${30}" "${31}" "${32}" <<'PY'
import hashlib, json, sys
(
    model_id,
    source_commit,
    source_kind,
    source_digest,
    source_resolved,
    scheme,
    fp8_scheme,
    fp8_pathway,
    model_revision,
    trust_remote_code,
    use_auth_token,
    num_samples,
    seq_length,
    batch_size,
    dataset,
    dataset_config,
    dataset_path,
    text_column,
    shuffle,
    seed,
    pipeline,
    sequential_targets,
    qac,
    streaming,
    splits_spec,
    ignore_extra,
    script_sha,
    python_ver,
    torch_ver,
    transformers_ver,
    llmcompressor_ver,
) = sys.argv[1:32]
fp = {
    "model": {
        "id": model_id,
        "source_kind": source_kind,
        "source_commit": source_commit or None,
        "source_digest": source_digest or None,
        "source_resolved": source_resolved or None,
        "revision": model_revision,
        "trust_remote_code": trust_remote_code == "1",
        "use_auth_token": use_auth_token == "1",
    },
    "quant": {
        "scheme": scheme,
        "fp8_scheme": fp8_scheme or None,
        "fp8_pathway": fp8_pathway or None,
        "num_samples": int(num_samples),
        "seq_length": int(seq_length),
        "batch_size": int(batch_size),
        "dataset": dataset or None,
        "dataset_config": dataset_config or None,
        "dataset_path": dataset_path or None,
        "text_column": text_column or None,
        "shuffle": shuffle == "1",
        "seed": int(seed),
        "pipeline": pipeline or None,
        "sequential_targets": sequential_targets or None,
        "quantization_aware_calibration": qac == "1",
        "streaming": streaming == "1",
        "splits": splits_spec or None,
        "ignore_extra": ignore_extra or None,
    },
    "build": {
        "script_sha256": script_sha,
        "python_version": python_ver,
        "torch_version": torch_ver,
        "transformers_version": transformers_ver,
        "llmcompressor_version": llmcompressor_ver,
    },
}
canonical = json.dumps(fp, sort_keys=True, separators=(",", ":"))
snap_id = hashlib.sha256(canonical.encode()).hexdigest()[:40]
print(json.dumps(fp, indent=2, sort_keys=True))
print(snap_id)
PY
}

LAST_VALIDATE_REASON=""
LAST_VALIDATED_ENTRIES_SHA256=""
LAST_VALIDATED_MANIFEST_FILE_SHA256=""
LAST_VALIDATED_SIZE_BYTES="0"

validate_output_dir() {
  local out_dir="$1"
  local expected_fingerprint="$2"
  local result
  if ! result="$(python3 - "$out_dir" "$expected_fingerprint" <<'PY'
import hashlib, json, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_fp = json.loads(sys.argv[2])
EXCLUDED_FROM_MANIFEST = {'artifact_manifest.json', 'quantize_info.json'}
REQUIRED_FILES = {'config.json', 'quantize_info.json', 'artifact_manifest.json', 'FINGERPRINT.json'}

def fail(msg):
    print(msg)
    raise SystemExit(1)

def load_json(path, label):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except FileNotFoundError:
        fail(f"missing {label}: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid {label} JSON: {exc}")

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

if not out_dir.is_dir():
    fail(f"output path is not a directory: {out_dir}")
for name in REQUIRED_FILES:
    if not (out_dir / name).is_file():
        fail(f"required output file missing: {name}")

fp = load_json(out_dir / 'FINGERPRINT.json', 'fingerprint')
if canon(fp) != canon(expected_fp):
    fail('stored fingerprint does not match requested build fingerprint')

meta = load_json(out_dir / 'quantize_info.json', 'metadata')
manifest_path = out_dir / 'artifact_manifest.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes.decode('utf-8'))
entries = manifest.get('files')
if not isinstance(entries, list) or not entries:
    fail('artifact_manifest.json has no file entries')

seen = set()
payload_size = 0
manifest_has_weights = False
entries_hash = hashlib.sha256()
for item in sorted(entries, key=lambda x: x.get('path', '')):
    rel = item.get('path')
    sha = item.get('sha256')
    size = item.get('size_bytes')
    if not isinstance(rel, str) or not rel or rel.startswith('/') or '..' in rel.split('/'):
        fail(f"invalid manifest path: {rel!r}")
    if rel in EXCLUDED_FROM_MANIFEST:
        fail(f"manifest must not include self-referential metadata file: {rel}")
    if rel in seen:
        fail(f"duplicate manifest path: {rel}")
    seen.add(rel)
    path = out_dir / rel
    if not path.is_file():
        fail(f"manifest file missing on disk: {rel}")
    st = path.stat()
    if int(size) != st.st_size:
        fail(f"manifest size mismatch for {rel}: {size} != {st.st_size}")
    actual_sha = sha256_file(path)
    if sha != actual_sha:
        fail(f"manifest sha256 mismatch for {rel}")
    entries_hash.update(rel.encode('utf-8'))
    entries_hash.update(b'\0')
    entries_hash.update(actual_sha.encode('ascii'))
    entries_hash.update(b'\0')
    entries_hash.update(str(st.st_size).encode('ascii'))
    entries_hash.update(b'\n')
    payload_size += st.st_size
    if rel.endswith('.safetensors') or rel.endswith('.bin') or rel == 'model.safetensors.index.json':
        manifest_has_weights = True

if not manifest_has_weights:
    fail('artifact manifest does not include model weight files')

entries_sha = entries_hash.hexdigest()
manifest_file_sha = sha256_bytes(manifest_bytes)
if manifest.get('artifact_entries_sha256') != entries_sha:
    fail('artifact manifest entries digest mismatch')
if meta.get('artifact_entries_sha256') != entries_sha:
    fail('metadata artifact_entries_sha256 mismatch')
if meta.get('artifact_manifest_sha256') != manifest_file_sha:
    fail('metadata artifact_manifest_sha256 mismatch')

try:
    payload_meta = int(meta.get('payload_size_bytes', -1))
    output_meta = int(meta.get('output_size_bytes', -1))
except Exception:
    fail('metadata size fields are not valid integers')

manifest_file_size = manifest_path.stat().st_size
quantize_info_size = (out_dir / 'quantize_info.json').stat().st_size
expected_output_size = payload_size + manifest_file_size + quantize_info_size
if payload_meta != payload_size:
    fail(f"metadata payload_size_bytes mismatch: {payload_meta} != {payload_size}")
if output_meta != expected_output_size:
    fail(f"metadata output_size_bytes mismatch: {output_meta} != {expected_output_size}")

print(f"{expected_output_size}\t{entries_sha}\t{manifest_file_sha}")
PY
)"; then
    LAST_VALIDATE_REASON="${result:-artifact validation failed}"
    LAST_VALIDATED_ENTRIES_SHA256=""
    LAST_VALIDATED_MANIFEST_FILE_SHA256=""
    LAST_VALIDATED_SIZE_BYTES="0"
    return 1
  fi
  LAST_VALIDATE_REASON=""
  IFS=$'\t' read -r LAST_VALIDATED_SIZE_BYTES LAST_VALIDATED_ENTRIES_SHA256 LAST_VALIDATED_MANIFEST_FILE_SHA256 <<< "$result"
  LAST_VALIDATED_SIZE_BYTES="${LAST_VALIDATED_SIZE_BYTES:-0}"
  LAST_VALIDATED_ENTRIES_SHA256="${LAST_VALIDATED_ENTRIES_SHA256:-}"
  LAST_VALIDATED_MANIFEST_FILE_SHA256="${LAST_VALIDATED_MANIFEST_FILE_SHA256:-}"
  return 0
}

emit_exists_json() {
  local out_dir="$1"
  python3 - "$MODEL_ID" "$SCHEME" "$out_dir" "$LAST_VALIDATED_SIZE_BYTES" "$LAST_VALIDATED_ENTRIES_SHA256" "$LAST_VALIDATED_MANIFEST_FILE_SHA256" "$SOURCE_RESOLVED" <<'PY'
import json, sys, time
model_id, scheme, out_path, size_bytes, entries_sha, manifest_file_sha, source_resolved = sys.argv[1:8]
print(json.dumps({
    "status": "exists",
    "source": model_id,
    "source_resolved": source_resolved,
    "scheme": scheme,
    "output_path": out_path,
    "size_bytes": int(size_bytes),
    "artifact_entries_sha256": entries_sha,
    "artifact_manifest_sha256": manifest_file_sha,
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}))
PY
}

maybe_exit_if_output_exists() {
  local out_dir="$1"
  local expected_fingerprint="$2"

  [[ -e "$out_dir" ]] || return 0

  if validate_output_dir "$out_dir" "$expected_fingerprint"; then
    if [[ "$FORCE" == "1" ]]; then
      log "Existing output found and verified; will rebuild (--force): $out_dir"
    else
      log "Output already exists and matches requested fingerprint: $out_dir"
      if [[ "$JSON_MODE" == "1" ]]; then
        emit_exists_json "$out_dir"
      fi
      exit 0
    fi
  else
    log "Ignoring existing output at $out_dir: $LAST_VALIDATE_REASON"
  fi
}

acquire_lock_fd() {
  local fd="$1"
  local lock_file="$2"
  local timeout="$3"

  if [[ "$timeout" -eq 0 ]]; then
    flock -n "$fd" || die "Lock is held: $lock_file (use --lock-timeout SECONDS to wait, or -1 to wait without limit)"
  elif [[ "$timeout" -lt 0 ]]; then
    flock "$fd"
  else
    flock -w "$timeout" "$fd" || die "Timed out waiting for lock after ${timeout}s: $lock_file"
  fi
}

write_repo_ref() {
  local refs_dir="$1"
  local revision="$2"
  local snap_rev="$3"

  [[ -n "$refs_dir" && -n "$revision" ]] || return 0
  if [[ "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    return 0
  fi
  case "$revision" in
    *[!A-Za-z0-9._/-]*)
      log "WARNING: refusing to write ref for unsupported revision name: $revision"
      return 0
      ;;
  esac
  local ref_path="$refs_dir/$revision"
  mkdir -p "$(dirname "$ref_path")"
  printf '%s\n' "$snap_rev" > "$ref_path"
}

MODEL_ID=""
SCHEME="fp8"

FP8_SCHEME="dynamic"
FP8_PATHWAY="oneshot"
IGNORE_EXTRA=""
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

PIPELINE=""
SEQUENTIAL_TARGETS=""
SEQUENTIAL_OFFLOAD="cpu"
QAC=1

STREAMING=0
SPLITS_SPEC=""
PREPROCESSING_WORKERS=""
DATALOADER_WORKERS=0

MODEL_REVISION="main"
USE_AUTH_TOKEN=0
SUFFIX=""
FORCE=0
TRUST_REMOTE_CODE=0
ONLINE=0

LOCK_TIMEOUT=0

VALIDATE=0
VALIDATE_PROMPT="Hello!"
VALIDATE_SUITE=""
VALIDATE_SEED=1
VALIDATE_MAX_TOKENS=64
VALIDATE_MIN_CHARS=1

LOG_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '/^# Usage:/,/^set -Eeuo pipefail/{ /^#/s/^# \{0,1\}//p; }' "$0"
      exit 0
      ;;
    --fp8-scheme)             require_value "$1" "$#"; FP8_SCHEME="$2"; shift 2 ;;
    --fp8-scheme=*)           FP8_SCHEME="${1#*=}"; shift ;;
    --fp8-pathway)            require_value "$1" "$#"; FP8_PATHWAY="$2"; shift 2 ;;
    --fp8-pathway=*)          FP8_PATHWAY="${1#*=}"; shift ;;
    --ignore)                 require_value "$1" "$#"; IGNORE_EXTRA="${IGNORE_EXTRA:+$IGNORE_EXTRA,}$2"; shift 2 ;;
    --ignore=*)               IGNORE_EXTRA="${IGNORE_EXTRA:+$IGNORE_EXTRA,}${1#*=}"; shift ;;

    --model-free-device)      require_value "$1" "$#"; MODEL_FREE_DEVICE="$2"; shift 2 ;;
    --model-free-device=*)    MODEL_FREE_DEVICE="${1#*=}"; shift ;;
    --model-free-workers)     require_value "$1" "$#"; MODEL_FREE_WORKERS="$2"; shift 2 ;;
    --model-free-workers=*)   MODEL_FREE_WORKERS="${1#*=}"; shift ;;

    --num-samples)            require_value "$1" "$#"; NUM_SAMPLES="$2"; shift 2 ;;
    --num-samples=*)          NUM_SAMPLES="${1#*=}"; shift ;;
    --seq-length)             require_value "$1" "$#"; SEQ_LENGTH="$2"; shift 2 ;;
    --seq-length=*)           SEQ_LENGTH="${1#*=}"; shift ;;
    --batch-size)             require_value "$1" "$#"; BATCH_SIZE="$2"; shift 2 ;;
    --batch-size=*)           BATCH_SIZE="${1#*=}"; shift ;;
    --dataset)                require_value "$1" "$#"; DATASET="$2"; shift 2 ;;
    --dataset=*)              DATASET="${1#*=}"; shift ;;
    --dataset-config)         require_value "$1" "$#"; DATASET_CONFIG="$2"; shift 2 ;;
    --dataset-config=*)       DATASET_CONFIG="${1#*=}"; shift ;;
    --dataset-path)           require_value "$1" "$#"; DATASET_PATH="$2"; shift 2 ;;
    --dataset-path=*)         DATASET_PATH="${1#*=}"; shift ;;
    --text-column)            require_value "$1" "$#"; TEXT_COLUMN="$2"; shift 2 ;;
    --text-column=*)          TEXT_COLUMN="${1#*=}"; shift ;;
    --no-shuffle)             SHUFFLE=0; shift ;;
    --seed)                   require_value "$1" "$#"; SEED="$2"; shift 2 ;;
    --seed=*)                 SEED="${1#*=}"; shift ;;

    --pipeline)               require_value "$1" "$#"; PIPELINE="$2"; shift 2 ;;
    --pipeline=*)             PIPELINE="${1#*=}"; shift ;;
    --sequential-targets)     require_value "$1" "$#"; SEQUENTIAL_TARGETS="$2"; shift 2 ;;
    --sequential-targets=*)   SEQUENTIAL_TARGETS="${1#*=}"; shift ;;
    --sequential-offload)     require_value "$1" "$#"; SEQUENTIAL_OFFLOAD="$2"; shift 2 ;;
    --sequential-offload=*)   SEQUENTIAL_OFFLOAD="${1#*=}"; shift ;;
    --no-qac)                 QAC=0; shift ;;

    --streaming)              STREAMING=1; shift ;;
    --splits)                 require_value "$1" "$#"; SPLITS_SPEC="$2"; shift 2 ;;
    --splits=*)               SPLITS_SPEC="${1#*=}"; shift ;;
    --preprocessing-workers)  require_value "$1" "$#"; PREPROCESSING_WORKERS="$2"; shift 2 ;;
    --preprocessing-workers=*) PREPROCESSING_WORKERS="${1#*=}"; shift ;;
    --dataloader-workers)     require_value "$1" "$#"; DATALOADER_WORKERS="$2"; shift 2 ;;
    --dataloader-workers=*)   DATALOADER_WORKERS="${1#*=}"; shift ;;

    --model-revision)         require_value "$1" "$#"; MODEL_REVISION="$2"; shift 2 ;;
    --model-revision=*)       MODEL_REVISION="${1#*=}"; shift ;;
    --use-auth-token)         USE_AUTH_TOKEN=1; shift ;;
    --suffix)                 require_value "$1" "$#"; SUFFIX="$2"; shift 2 ;;
    --suffix=*)               SUFFIX="${1#*=}"; shift ;;
    --force)                  FORCE=1; shift ;;
    --trust-remote-code)      TRUST_REMOTE_CODE=1; shift ;;
    --online)                 ONLINE=1; shift ;;

    --lock-timeout)           require_value "$1" "$#"; LOCK_TIMEOUT="$2"; shift 2 ;;
    --lock-timeout=*)         LOCK_TIMEOUT="${1#*=}"; shift ;;

    --validate)               VALIDATE=1; shift ;;
    --validate-prompt)        require_value "$1" "$#"; VALIDATE_PROMPT="$2"; shift 2 ;;
    --validate-prompt=*)      VALIDATE_PROMPT="${1#*=}"; shift ;;
    --validate-suite)         require_value "$1" "$#"; VALIDATE_SUITE="$2"; shift 2 ;;
    --validate-suite=*)       VALIDATE_SUITE="${1#*=}"; shift ;;
    --validate-seed)          require_value "$1" "$#"; VALIDATE_SEED="$2"; shift 2 ;;
    --validate-seed=*)        VALIDATE_SEED="${1#*=}"; shift ;;
    --validate-max-tokens)    require_value "$1" "$#"; VALIDATE_MAX_TOKENS="$2"; shift 2 ;;
    --validate-max-tokens=*)  VALIDATE_MAX_TOKENS="${1#*=}"; shift ;;
    --validate-min-chars)     require_value "$1" "$#"; VALIDATE_MIN_CHARS="$2"; shift 2 ;;
    --validate-min-chars=*)   VALIDATE_MIN_CHARS="${1#*=}"; shift ;;

    --log-dir)                require_value "$1" "$#"; LOG_DIR="$2"; shift 2 ;;
    --log-dir=*)              LOG_DIR="${1#*=}"; shift ;;
    --json)                   JSON_MODE=1; shift ;;
    --json-strict)            JSON_MODE=1; JSON_STRICT=1; shift ;;

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

[[ -n "$MODEL_ID" ]] || die "Missing <model-id-or-path>"
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
[[ "$SEQ_LENGTH" =~ ^[0-9]+$ ]] || die "--seq-length must be an integer"
[[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || die "--batch-size must be an integer"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be an integer"
[[ "$MODEL_FREE_WORKERS" =~ ^[0-9]+$ ]] || die "--model-free-workers must be an integer"
[[ "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] || die "--dataloader-workers must be an integer"
[[ "$VALIDATE_SEED" =~ ^[0-9]+$ ]] || die "--validate-seed must be an integer"
[[ "$VALIDATE_MAX_TOKENS" =~ ^[0-9]+$ ]] || die "--validate-max-tokens must be an integer"
[[ "$VALIDATE_MIN_CHARS" =~ ^[0-9]+$ ]] || die "--validate-min-chars must be an integer"
[[ "$LOCK_TIMEOUT" =~ ^-?[0-9]+$ ]] || die "--lock-timeout must be an integer (0, >0, or -1)"
if [[ -n "$PREPROCESSING_WORKERS" ]]; then
  [[ "$PREPROCESSING_WORKERS" =~ ^[0-9]+$ ]] || die "--preprocessing-workers must be an integer"
fi

CURRENT_STAGE="preflight"
have python3 || die "python3 not found on PATH"
have flock || die "flock not found. Install util-linux (or equivalent) for safe concurrent use."

SCRIPT_SHA="$(sha256_file "$0")"
PYTHON_VERSION="$(python3 - <<'PY'
import sys
if sys.version_info < (3, 8):
    raise SystemExit('python3 >= 3.8 is required')
print(sys.version.split()[0])
PY
)" || die "python3 >= 3.8 is required"
TORCH_VERSION="$(python_module_version torch)"
TRANSFORMERS_VERSION="$(python_module_version transformers)"
LLMCOMPRESSOR_VERSION="$(python_module_version llmcompressor)"

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

SOURCE_IDENTITY_JSON="$(python3 - "$MODEL_ID" "$MODEL_REVISION" "$CACHE_DIR" "$ONLINE" "$USE_AUTH_TOKEN" <<'PY'
import hashlib, json, re, sys
from pathlib import Path

model_id, revision, cache_dir, online, use_auth_token = sys.argv[1:6]
online = online == '1'
use_auth_token = use_auth_token == '1'

def die(msg):
    print(msg, file=sys.stderr)
    raise SystemExit(2)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_tree(root: Path) -> str:
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if '.git' in rel_parts or '__pycache__' in rel_parts:
            continue
        files.append(path)
    if not files:
        die(f'Local model path has no files to fingerprint: {root}')
    h = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        st = path.stat()
        h.update(rel.encode('utf-8'))
        h.update(b'\0')
        h.update(str(st.st_size).encode('ascii'))
        h.update(b'\0')
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                h.update(chunk)
        h.update(b'\n')
    return h.hexdigest()

pp = Path(model_id)
if pp.exists():
    resolved = str(pp.resolve())
    digest = sha256_file(pp.resolve()) if pp.is_file() else sha256_tree(pp.resolve())
    print(json.dumps({
        'source_kind': 'local',
        'source_commit': '',
        'source_digest': digest,
        'source_resolved': resolved,
    }))
    raise SystemExit(0)

resolved_commit = ''
cache_key = model_id.replace('/', '--')
ref_path = Path(cache_dir) / 'hub' / f'models--{cache_key}' / 'refs' / revision
if ref_path.is_file():
    resolved_commit = ref_path.read_text(encoding='utf-8').strip()
elif re.fullmatch(r'[0-9a-f]{40}', revision):
    resolved_commit = revision

if not resolved_commit and online:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=True if use_auth_token else None)
        info = api.model_info(repo_id=model_id, revision=revision)
        resolved_commit = (getattr(info, 'sha', '') or '').strip()
    except Exception as exc:
        die(f'Failed to resolve exact upstream commit for {model_id}@{revision}: {exc}')

if not resolved_commit:
    die(
        'Could not resolve an exact source commit for the hub model. '
        'For CI/prod this script requires an immutable source identity. '
        f'Model: {model_id}  Revision: {revision}. '
        'Fix: use an exact commit revision, pre-populate the local hub refs cache, or run with --online.'
    )

print(json.dumps({
    'source_kind': 'hub',
    'source_commit': resolved_commit,
    'source_digest': resolved_commit,
    'source_resolved': f'{model_id}@{resolved_commit}',
}))
PY
)"
SOURCE_KIND="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["source_kind"])' "$SOURCE_IDENTITY_JSON")"
SOURCE_COMMIT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["source_commit"])' "$SOURCE_IDENTITY_JSON")"
SOURCE_DIGEST="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["source_digest"])' "$SOURCE_IDENTITY_JSON")"
SOURCE_RESOLVED="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["source_resolved"])' "$SOURCE_IDENTITY_JSON")"

FINGERPRINT_JSON="$(build_fingerprint_json   "$MODEL_ID" "$SOURCE_COMMIT" "$SOURCE_KIND" "$SOURCE_DIGEST" "$SOURCE_RESOLVED" "$SCHEME" "$FP8_SCHEME" "$FP8_PATHWAY"   "$MODEL_REVISION" "$TRUST_REMOTE_CODE" "$USE_AUTH_TOKEN" "$NUM_SAMPLES" "$SEQ_LENGTH"   "$BATCH_SIZE" "$DATASET" "$DATASET_CONFIG" "$DATASET_PATH" "$TEXT_COLUMN" "$SHUFFLE"   "$SEED" "$PIPELINE" "$SEQUENTIAL_TARGETS" "$QAC" "$STREAMING" "$SPLITS_SPEC"   "$IGNORE_EXTRA" "$SCRIPT_SHA" "$PYTHON_VERSION" "$TORCH_VERSION" "$TRANSFORMERS_VERSION" "$LLMCOMPRESSOR_VERSION")"
SNAP_REV="$(echo "$FINGERPRINT_JSON" | tail -1)"
FINGERPRINT_BODY="$(echo "$FINGERPRINT_JSON" | sed '$d')"

if [[ "$WRITE_LAYOUT" == "1" ]]; then
  MODEL_CACHE_KEY="$(printf '%s' "$OUTPUT_MODEL_ID" | sed 's|/|--|g')"
  FINAL_OUT="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/snapshots/${SNAP_REV}"
  REFS_DIR="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/refs"
else
  FINAL_OUT="$OUTPUT_DIR/${OUTPUT_SLUG}/${SNAP_REV}"
  REFS_DIR=""
fi

CURRENT_STAGE="locking"
LOCK_FILE="$OUTPUT_DIR/.quantize-${OUTPUT_SLUG}.lock"
exec 9>"$LOCK_FILE"
acquire_lock_fd 9 "$LOCK_FILE" "$LOCK_TIMEOUT"
maybe_exit_if_output_exists "$FINAL_OUT" "$FINGERPRINT_BODY"

export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
if [[ "$ONLINE" == "0" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error
export PYTHONHASHSEED="$SEED"

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
  abs_target="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$target")"
  abs_root="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$OUTPUT_DIR")"
  case "$abs_target" in
    "$abs_root"/*) ;;
    *) die "refusing to delete path outside output root: $abs_target" ;;
  esac
  rm -rf -- "$abs_target"
}


sync_path() {
  local target="$1"
  python3 - "$target" <<'PY'
import os, sys
path = sys.argv[1]
flags = os.O_RDONLY
if os.path.isdir(path):
    flags |= getattr(os, 'O_DIRECTORY', 0)
fd = os.open(path, flags)
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

sync_tree() {
  local root="$1"
  python3 - "$root" <<'PY'
import os, sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(0)

for path in sorted(p for p in root.rglob('*') if p.is_file()):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

dirs = [root] + sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
for path in dirs:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
PY
}

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
log "  Source ID:    $SOURCE_RESOLVED"
log "  Commit:       ${SOURCE_COMMIT:-<local>}"
log "  Auth token:   $USE_AUTH_TOKEN"
log "  Lock:         $LOCK_FILE"
log "  Lock wait:    $LOCK_TIMEOUT"
log "  Script SHA:   ${SCRIPT_SHA:0:16}"
log ""

CURRENT_STAGE="quantization"
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
export LLMC_SCRIPT_SHA="$SCRIPT_SHA"
export LLMC_FINGERPRINT_JSON="$FINGERPRINT_BODY"

python3 <<'PY'
import json, os, sys, time, platform, hashlib
from datetime import datetime, timezone
from pathlib import Path

json_mode = os.environ.get("JSON_MODE", "0") == "1"
out = sys.stderr if json_mode else sys.stdout
if json_mode:
    import logging
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    sys.stdout = sys.stderr

def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

def pkg_version(name: str):
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
        parts = []
        for p in v.split('.'):
            digits = ''.join(ch for ch in p if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

def ver_ge(v: str, minimum: str) -> bool:
    return parse_ver(v) >= parse_ver(minimum)

def looks_like_model_path(p: str) -> bool:
    pp = Path(p)
    if not pp.exists():
        return False
    if pp.is_file():
        name = pp.name.lower()
        return name == 'config.json' or name.endswith('.safetensors') or name.endswith('.bin') or name.endswith('.pt')
    return (
        (pp / 'config.json').is_file()
        or (pp / 'model.safetensors.index.json').is_file()
        or bool(list(pp.glob('*.safetensors')))
        or bool(list(pp.glob('*.bin')))
    )

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

model_id = os.environ['LLMC_MODEL_ID']
scheme = os.environ['LLMC_SCHEME']
fp8_scheme_flag = os.environ.get('LLMC_FP8_SCHEME', 'dynamic')
fp8_pathway = os.environ.get('LLMC_FP8_PATHWAY', 'oneshot')
out_dir = os.environ['LLMC_OUTPUT_DIR']
trust_remote = os.environ['LLMC_TRUST_REMOTE_CODE'] == '1'
model_revision = os.environ.get('LLMC_MODEL_REVISION') or 'main'
use_auth_token = os.environ.get('LLMC_USE_AUTH_TOKEN', '0') == '1'
pipeline = (os.environ.get('LLMC_PIPELINE') or '').strip() or None
seq_targets_raw = (os.environ.get('LLMC_SEQUENTIAL_TARGETS') or '').strip()
seq_offload = (os.environ.get('LLMC_SEQUENTIAL_OFFLOAD') or '').strip() or 'cpu'
qac = os.environ.get('LLMC_QAC', '1') == '1'
log_dir = (os.environ.get('LLMC_LOG_DIR') or '').strip() or None
num_samples = int(os.environ.get('LLMC_NUM_SAMPLES') or 512)
seq_len = int(os.environ.get('LLMC_SEQ_LENGTH') or 2048)
batch_size = int(os.environ.get('LLMC_BATCH_SIZE') or 1)
dataset = os.environ.get('LLMC_DATASET') or 'open_platypus'
dataset_config = (os.environ.get('LLMC_DATASET_CONFIG') or '').strip() or None
dataset_path = (os.environ.get('LLMC_DATASET_PATH') or '').strip() or None
text_column = os.environ.get('LLMC_TEXT_COLUMN') or 'text'
shuffle = os.environ.get('LLMC_SHUFFLE') == '1'
seed = int(os.environ.get('LLMC_SEED') or 1234)
streaming = os.environ.get('LLMC_STREAMING', '0') == '1'
splits_spec = (os.environ.get('LLMC_SPLITS_SPEC') or '').strip() or None
preprocessing_workers_raw = (os.environ.get('LLMC_PREPROCESSING_WORKERS') or '').strip()
preprocessing_workers = int(preprocessing_workers_raw) if preprocessing_workers_raw else None
dataloader_workers = int(os.environ.get('LLMC_DATALOADER_WORKERS') or 0)
ignore_extra = (os.environ.get('LLMC_IGNORE_EXTRA') or '').strip()
model_free_device = os.environ.get('LLMC_MODEL_FREE_DEVICE') or 'cuda:0'
model_free_workers = int(os.environ.get('LLMC_MODEL_FREE_WORKERS') or 8)
validate_requested = os.environ.get('LLMC_VALIDATE') == '1'
script_sha = os.environ.get('LLMC_SCRIPT_SHA') or ''
fingerprint_json = json.loads(os.environ['LLMC_FINGERPRINT_JSON'])
started_utc = datetime.now(timezone.utc).isoformat()
offline = (os.environ.get('HF_HUB_OFFLINE') == '1') or (os.environ.get('TRANSFORMERS_OFFLINE') == '1')

if offline and streaming:
    die('--streaming is not allowed in offline mode. Fix: drop --streaming or run with --online.')

required = [
    ('torch', '2.1.0', 'pip install torch'),
    ('transformers', '4.45.0', 'pip install -U transformers'),
    ('huggingface_hub', '0.22.0', 'pip install -U huggingface_hub'),
    ('llmcompressor', '0.7.1', 'pip install -U llmcompressor'),
]
if scheme == 'fp8' and fp8_pathway == 'model_free':
    required = [(k, ('0.9.0' if k == 'llmcompressor' else v), h) for (k, v, h) in required]
if scheme != 'fp8':
    required.append(('datasets', '2.14.0', 'pip install -U datasets'))
if validate_requested:
    required.append(('vllm', '0.6.0', 'pip install -U vllm'))

missing = []
too_old = []
for pkg, minv, hint in required:
    v = pkg_version(pkg)
    if v is None:
        missing.append((pkg, hint))
    elif not ver_ge(v, minv):
        too_old.append((pkg, v, minv, hint))
if missing:
    lines = ['Missing dependencies:']
    for pkg, hint in missing:
        lines.append(f'  - {pkg}  ({hint})')
    die('\n'.join(lines))
if too_old:
    lines = ['Dependencies are present but older than this script expects:']
    for pkg, v, minv, hint in too_old:
        lines.append(f'  - {pkg} {v}  (need >= {minv})  [{hint}]')
    die('\n'.join(lines))

print('Python:', sys.version.split()[0], file=out)
print('Platform:', platform.platform(), file=out)
print('torch:', pkg_version('torch'), file=out)
print('transformers:', pkg_version('transformers'), file=out)
print('llmcompressor:', pkg_version('llmcompressor'), file=out)
if scheme != 'fp8':
    print('datasets:', pkg_version('datasets'), file=out)
print('huggingface_hub:', pkg_version('huggingface_hub'), file=out)
if validate_requested:
    print('vllm:', pkg_version('vllm'), file=out)
print(file=out)

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

cuda_cc = None
cuda_name = None
try:
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        free_mb, total_mb = free >> 20, total >> 20
        cuda_cc = torch.cuda.get_device_capability()
        cuda_name = torch.cuda.get_device_name(0)
        print(f'CUDA device: {cuda_name} (compute capability {cuda_cc[0]}.{cuda_cc[1]})', file=out)
        print(f'CUDA free:   {free_mb} MB / {total_mb} MB', file=out)
        if free_mb < 512:
            die(f'GPU has only {free_mb} MB free of {total_mb} MB. Free GPU memory before quantizing.')
        if scheme == 'fp8':
            cc_num = cuda_cc[0] * 10 + cuda_cc[1]
            if cc_num >= 90:
                print('FP8 note: CC >= 9.0; matches vLLM FP8 compute support band.', file=out)
            elif cc_num >= 80:
                print('FP8 note: Ampere-class CC >= 8.0; FP8 can run as weight-only W8A16 via Marlin.', file=out)
            else:
                print('FP8 note: CC < 8.0; FP8 behavior is outside the documented paths.', file=out)
        elif scheme == 'nvfp4':
            cc_num = cuda_cc[0] * 10 + cuda_cc[1]
            if cc_num >= 100:
                print('NVFP4 note: SM100-class GPU detected; activation quantization is available.', file=out)
            else:
                print('NVFP4 note: < SM100; activation quantization is not available, weight-only path only.', file=out)
        print(file=out)
except SystemExit:
    raise
except Exception as e:
    print(f'CUDA info unavailable: {e}', file=out)
    print(file=out)

is_local_model = looks_like_model_path(model_id)
if offline and (not is_local_model):
    try:
        from huggingface_hub import hf_hub_download
        _ = hf_hub_download(
            repo_id=model_id,
            filename='config.json',
            revision=model_revision,
            local_files_only=True,
            token=True if use_auth_token else None,
        )
        print(f"Offline check: model revision '{model_revision}' is present in cache.", file=out)
        print(file=out)
    except Exception as e:
        die(
            'Offline mode is set, but the requested model revision is not available locally.\n'
            f'Model: {model_id}\nRevision: {model_revision}\nDetails: {e}\n'
            'Fix: pre-download this revision with --online, or change --model-revision.'
        )

if scheme == 'fp8' and fp8_pathway == 'model_free' and not is_local_model:
    die(
        'fp8 model_free pathway requires a local model path. '
        'Reason: model_free_ptq has no revision argument for hub IDs. '
        'Fix: pass a local snapshot directory, or switch to --fp8-pathway oneshot.'
    )

if scheme != 'fp8' and dataset_path and dataset_path.startswith('dvc://'):
    if offline:
        die('Offline mode is set, but --dataset-path uses dvc://... Run with --online or pass a local dataset.')
    import importlib.util, shutil
    has_dvc = (importlib.util.find_spec('dvc') is not None) or (shutil.which('dvc') is not None)
    if not has_dvc:
        die('Dataset path uses dvc://..., but DVC is not installed. Fix: install DVC and any remote extras you use.')

loaded_dataset = None
dataset_mode = 'hub'
if scheme != 'fp8':
    if dataset_path:
        if dataset_path.startswith('dvc://'):
            dataset_mode = 'dvc'
        else:
            from datasets import load_dataset, DatasetDict
            p = Path(dataset_path)
            if not p.exists():
                die(f'--dataset-path does not exist: {dataset_path}')
            def detect_files(dir_path: Path):
                exts = ['.jsonl', '.json', '.csv', '.parquet']
                found = {e: [] for e in exts}
                for e in exts:
                    found[e] = sorted([str(x) for x in dir_path.rglob(f'*{e}') if x.is_file()])
                return {e: v for e, v in found.items() if v}
            if p.is_file():
                ext = p.suffix.lower()
                if ext in ('.jsonl', '.json'):
                    loader = 'json'
                elif ext == '.csv':
                    loader = 'csv'
                elif ext == '.parquet':
                    loader = 'parquet'
                else:
                    die('Unsupported file type for --dataset-path. Use .json/.jsonl/.csv/.parquet.')
                loaded_dataset = load_dataset(loader, data_files={'train': [str(p)]}, name=dataset_config, streaming=False)
            else:
                nonempty = detect_files(p)
                if not nonempty:
                    die('No .json/.jsonl/.csv/.parquet files found under --dataset-path directory.')
                if len(nonempty) > 1:
                    die(f"Mixed dataset file types found in directory ({', '.join(sorted(nonempty.keys()))}).")
                ext, files = next(iter(nonempty.items()))
                loader = 'json' if ext in ('.json', '.jsonl') else ('csv' if ext == '.csv' else 'parquet')
                loaded_dataset = load_dataset(loader, data_files={'train': files}, name=dataset_config, streaming=False)
            dataset_mode = 'local_loaded'
            ds_train = loaded_dataset['train'] if isinstance(loaded_dataset, DatasetDict) else loaded_dataset
            cols = set(ds_train.column_names)
            if text_column not in cols:
                die(f"text column '{text_column}' not found in dataset columns: {sorted(cols)}")
            if offline:
                print('Offline check: local dataset loaded OK.', file=out)
                print(file=out)
    else:
        if offline:
            try:
                import datasets as ds
                _ = ds.load_dataset(dataset, name=dataset_config, split='train', download_mode='reuse_dataset_if_exists')
                print('Offline check: dataset id is loadable from cache (train split).', file=out)
                print(file=out)
            except Exception as e:
                die(
                    'Offline mode is set, but the dataset is not available locally.\n'
                    f'Dataset: {dataset}\nConfig: {dataset_config or "(none)"}\nDetails: {e}\n'
                    'Fix: pre-download the dataset with --online, or pass --dataset-path to a local dataset.'
                )

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

fp8_scheme = 'FP8_DYNAMIC' if fp8_scheme_flag == 'dynamic' else 'FP8_BLOCK'
ignore = ['lm_head']
if ignore_extra:
    for part in ignore_extra.split(','):
        p = part.strip()
        if p:
            ignore.append(p)
seq_targets = [s.strip() for s in seq_targets_raw.split(',') if s.strip()] if seq_targets_raw else None
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
    common_oneshot['pipeline'] = pipeline
if seq_targets is not None:
    common_oneshot['sequential_targets'] = seq_targets
    common_oneshot['sequential_offload_device'] = seq_offload

t0 = time.time()
if scheme == 'fp8':
    if fp8_pathway == 'model_free':
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
        recipe = QuantizationModifier(targets='Linear', scheme=fp8_scheme, ignore=ignore)
        oneshot(recipe=recipe, **common_oneshot)
else:
    if scheme == 'gptq':
        recipe = GPTQModifier(targets='Linear', scheme='W4A16', ignore=ignore)
    elif scheme == 'w8a8':
        recipe = [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(targets='Linear', scheme='W8A8', ignore=ignore),
        ]
    elif scheme == 'nvfp4':
        recipe = QuantizationModifier(targets='Linear', scheme='NVFP4', ignore=ignore)
    else:
        die(f'Unknown scheme: {scheme}')
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
        kwargs['preprocessing_num_workers'] = preprocessing_workers
    if dataset_mode in ('hub', 'dvc'):
        kwargs['streaming'] = bool(streaming)
        if splits_spec:
            kwargs['splits'] = splits_spec
    if dataset_mode == 'local_loaded':
        kwargs['dataset'] = loaded_dataset
    elif dataset_mode == 'dvc':
        kwargs['dataset_path'] = dataset_path
        if dataset_config is not None:
            kwargs['dataset_config_name'] = dataset_config
    else:
        kwargs['dataset'] = dataset
        if dataset_config is not None:
            kwargs['dataset_config_name'] = dataset_config
    oneshot(**kwargs)
elapsed = time.time() - t0

out_path = Path(out_dir)
out_path.mkdir(parents=True, exist_ok=True)
qinfo_path = out_path / 'quantize_info.json'
meta = {
    'model': model_id,
    'scheme': scheme,
    'fp8_scheme': fp8_scheme if scheme == 'fp8' else None,
    'fp8_pathway': fp8_pathway if scheme == 'fp8' else None,
    'model_revision': model_revision,
    'use_auth_token': use_auth_token,
    'output_dir': out_dir,
    'pipeline': pipeline,
    'sequential_targets': seq_targets,
    'sequential_offload_device': seq_offload if seq_targets else None,
    'quantization_aware_calibration': qac,
    'dataset_args': None if scheme == 'fp8' else {
        'dataset_mode': dataset_mode,
        'dataset': (dataset_path or dataset),
        'dataset_config': dataset_config,
        'num_samples': num_samples,
        'max_seq_length': seq_len,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'seed': seed,
        'text_column': text_column,
        'streaming': bool(streaming) if dataset_mode in ('hub', 'dvc') else False,
        'splits': splits_spec if dataset_mode in ('hub', 'dvc') else None,
        'preprocessing_num_workers': preprocessing_workers,
        'dataloader_num_workers': dataloader_workers,
    },
    'cuda': {'name': cuda_name, 'capability': list(cuda_cc) if cuda_cc else None},
    'versions': {
        'python': sys.version.split()[0],
        'torch': pkg_version('torch'),
        'transformers': pkg_version('transformers'),
        'huggingface_hub': pkg_version('huggingface_hub'),
        'datasets': pkg_version('datasets') if scheme != 'fp8' else None,
        'llmcompressor': pkg_version('llmcompressor'),
        'vllm': pkg_version('vllm') if validate_requested else None,
    },
    'script_sha256': script_sha,
    'fingerprint': fingerprint_json,
    'started_utc': started_utc,
    'finished_utc': datetime.now(timezone.utc).isoformat(),
    'elapsed_seconds': round(elapsed, 3),
}
with open(qinfo_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, sort_keys=True)
print(f'Quantization finished in {elapsed:.1f}s', file=out)
print(f'Wrote: {qinfo_path}', file=out)
print(file=out)
PY

CURRENT_STAGE="artifact-validation"
echo "$FINGERPRINT_BODY" > "$TMP_OUT/FINGERPRINT.json"
python3 - "$TMP_OUT" <<'PY'
import json, hashlib, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
EXCLUDED_FROM_MANIFEST = {'artifact_manifest.json', 'quantize_info.json'}

def die(msg: str):
    print(f'ERROR: {msg}', file=sys.stderr)
    raise SystemExit(2)

if not out_dir.exists():
    die(f'Output dir missing: {out_dir}')
if not (out_dir / 'quantize_info.json').is_file():
    die('quantize_info.json missing (quantization did not finish cleanly).')
if not (out_dir / 'config.json').is_file():
    die('config.json missing in output.')
if not (out_dir / 'FINGERPRINT.json').is_file():
    die('FINGERPRINT.json missing in output.')

has_safetensors = any(out_dir.glob('*.safetensors')) or (out_dir / 'model.safetensors.index.json').is_file()
has_bin = any(out_dir.glob('*.bin'))
if not (has_safetensors or has_bin):
    die('No weight files found (*.safetensors / *.bin / safetensors index).')

files = []
payload_size = 0
for path in sorted(p for p in out_dir.rglob('*') if p.is_file()):
    rel = path.relative_to(out_dir).as_posix()
    if rel in EXCLUDED_FROM_MANIFEST:
        continue
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    st = path.stat()
    payload_size += st.st_size
    files.append({'path': rel, 'sha256': h.hexdigest(), 'size_bytes': st.st_size})

if not files:
    die('No payload files found for artifact manifest.')

entries_hash = hashlib.sha256()
for item in files:
    entries_hash.update(item['path'].encode('utf-8'))
    entries_hash.update(b'\0')
    entries_hash.update(item['sha256'].encode('ascii'))
    entries_hash.update(b'\0')
    entries_hash.update(str(item['size_bytes']).encode('ascii'))
    entries_hash.update(b'\n')
entries_sha = entries_hash.hexdigest()

qinfo_path = out_dir / 'quantize_info.json'
with open(qinfo_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
meta['artifact_entries_sha256'] = entries_sha
meta['payload_size_bytes'] = payload_size

manifest_doc = {'files': files, 'artifact_entries_sha256': entries_sha}
manifest_path = out_dir / 'artifact_manifest.json'
manifest_bytes = json.dumps(manifest_doc, indent=2, sort_keys=True).encode('utf-8')
manifest_file_sha = hashlib.sha256(manifest_bytes).hexdigest()
meta['artifact_manifest_sha256'] = manifest_file_sha

prev_output_size = None
while True:
    meta_bytes = json.dumps(meta, indent=2, sort_keys=True).encode('utf-8')
    output_size = payload_size + len(manifest_bytes) + len(meta_bytes)
    if output_size == prev_output_size:
        break
    meta['output_size_bytes'] = output_size
    prev_output_size = output_size
meta_bytes = json.dumps(meta, indent=2, sort_keys=True).encode('utf-8')

with open(qinfo_path, 'wb') as f:
    f.write(meta_bytes)
    f.flush()
with open(manifest_path, 'wb') as f:
    f.write(manifest_bytes)
    f.flush()
print('Output sanity check: OK')
PY
validate_output_dir "$TMP_OUT" "$FINGERPRINT_BODY" || die "Refusing to publish invalid artifact: $LAST_VALIDATE_REASON"
sync_tree "$TMP_OUT"
sync_path "$(dirname "$TMP_OUT")"

CURRENT_STAGE="smoke-test"
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
json_mode = os.environ.get('JSON_MODE', '0') == '1'
_out = sys.stderr if json_mode else sys.stdout

def die(msg: str, code: int = 3):
    print(f'ERROR: {msg}', file=sys.stderr)
    raise SystemExit(code)

model_dir = os.environ['LLMC_VALIDATE_MODEL_DIR']
trust_remote = os.environ.get('LLMC_VALIDATE_TRUST_REMOTE_CODE') == '1'
prompt0 = os.environ.get('LLMC_VALIDATE_PROMPT') or 'Hello!'
suite_path = (os.environ.get('LLMC_VALIDATE_SUITE') or '').strip() or None
seed = int(os.environ.get('LLMC_VALIDATE_SEED') or 1)
max_tokens_default = int(os.environ.get('LLMC_VALIDATE_MAX_TOKENS') or 64)
min_chars_default = int(os.environ.get('LLMC_VALIDATE_MIN_CHARS') or 1)

try:
    import importlib.metadata as md
    _ = md.version('vllm')
except Exception:
    die('vLLM not installed but --validate was set. Fix: pip install -U vllm', code=2)

from vllm import LLM, SamplingParams

def load_suite(path: str):
    prompts = []
    if path.endswith('.jsonl'):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompts.append({'prompt': obj} if isinstance(obj, str) else obj)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.strip():
                    prompts.append({'prompt': line})
    return prompts

cases = [{'prompt': prompt0}]
if suite_path:
    cases = load_suite(suite_path)
llm = LLM(model=model_dir, trust_remote_code=trust_remote)
failures = 0
exceptions = 0
for i, case in enumerate(cases, 1):
    prompt = case.get('prompt', '')
    if not isinstance(prompt, str) or not prompt:
        failures += 1
        print(f'[validate {i}] FAIL invalid prompt', file=_out)
        continue
    max_tokens = int(case.get('max_tokens', max_tokens_default))
    min_chars = int(case.get('min_chars', min_chars_default))
    must_contain = case.get('must_contain') or []
    if isinstance(must_contain, str):
        must_contain = [must_contain]
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0, seed=seed)
    try:
        out = llm.generate([prompt], params)
        text = out[0].outputs[0].text if out and out[0].outputs else ''
        text_stripped = (text or '').strip()
        ok = len(text_stripped) >= min_chars and all(needle in text for needle in must_contain)
        status = 'PASS' if ok else 'FAIL'
        print(f'[validate {i}] {status}  prompt_len={len(prompt)}  out_len={len(text_stripped)}', file=_out)
        if not ok:
            failures += 1
            snippet = text_stripped[:500] + ('...' if len(text_stripped) > 500 else '')
            print('  output:', snippet, file=_out)
    except Exception as e:
        exceptions += 1
        failures += 1
        print(f'[validate {i}] FAIL exception: {type(e).__name__}: {e}', file=_out)
print(f'Validation summary: {len(cases)-failures} pass, {failures} fail ({exceptions} exceptions)', file=_out)
if failures:
    die('Validation failed (see per-case results above).', code=3)
print('Validation passed.', file=_out)
PY
  VALIDATION_STATUS="pass"
fi

CURRENT_STAGE="publish"
mkdir -p "$(dirname "$FINAL_OUT")"
if [[ -e "$FINAL_OUT" ]]; then
  log "Replacing existing output: $FINAL_OUT"
  safe_rm_rf "$FINAL_OUT"
fi
mv "$TMP_OUT" "$FINAL_OUT"
TMP_OUT=""
sync_tree "$FINAL_OUT"
sync_path "$(dirname "$FINAL_OUT")"
if [[ "$WRITE_LAYOUT" == "1" ]]; then
  mkdir -p "$REFS_DIR"
  write_repo_ref "$REFS_DIR" "$MODEL_REVISION" "$SNAP_REV"
  sync_tree "$REFS_DIR"
  sync_path "$(dirname "$REFS_DIR")"
fi

CURRENT_STAGE="complete"
if [[ "$JSON_MODE" == "1" ]]; then
  python3 - "$MODEL_ID" "$SCHEME" "$FINAL_OUT" "$VALIDATION_STATUS" "$LAST_VALIDATED_SIZE_BYTES" "$LAST_VALIDATED_ENTRIES_SHA256" "$LAST_VALIDATED_MANIFEST_FILE_SHA256" "$SOURCE_RESOLVED" <<'PY'
import json, sys, time
model_id, scheme, out_path, validation, size_bytes, entries_sha, manifest_file_sha, source_resolved = sys.argv[1:9]
print(json.dumps({
    'status': 'ok',
    'source': model_id,
    'source_resolved': source_resolved,
    'scheme': scheme,
    'output_path': out_path,
    'validation': validation,
    'size_bytes': int(size_bytes),
    'artifact_entries_sha256': entries_sha,
    'artifact_manifest_sha256': manifest_file_sha,
    'ts_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}))
PY
else
  log "Saved to: $FINAL_OUT"
  log "Size:     $(python3 -c "import sys; print(f'{int(sys.argv[1])/(1024**3):.2f} GB')" "$LAST_VALIDATED_SIZE_BYTES")"
  log "Validation: $VALIDATION_STATUS"
  log "Entries:    $LAST_VALIDATED_ENTRIES_SHA256"
  log "Manifest:   $LAST_VALIDATED_MANIFEST_FILE_SHA256"
  log "Done."
fi
