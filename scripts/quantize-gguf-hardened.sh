#!/usr/bin/env bash
# quantize-gguf.sh — Convert Hugging Face models to GGUF via llama.cpp.
#
# Two-phase pipeline:
#   1. convert_hf_to_gguf.py: HF safetensors → F16/BF16 GGUF
#   2. llama-quantize: F16/BF16 GGUF → quantized GGUF
#
# Usage:
#   quantize-gguf [--json|--json-strict] <model-id> [quant-type] [options]
#
# Positional:
#   model-id              Hugging Face model ID (e.g. Qwen/Qwen3-8B)
#   quant-type            GGUF quantization type (default: Q4_K_M)
#
# Options:
#   -c, --cache-dir DIR         HF cache root (default: $MODEL_CACHE_DIR or ./models)
#   -o, --output-dir DIR        Output root (default: $QUANTIZED_OUTPUT_DIR or cache dir)
#   -r, --revision REV          HF revision (default: main)
#       --suffix STR            Override output suffix (default: -GGUF-<TYPE>)
#       --online                Allow network access
#       --trust-remote-code     Allow model repo custom code
#       --force                 Rebuild even if validated output exists
#       --imatrix FILE          Importance matrix for quantization
#       --convert-type TYPE     Intermediate precision: f16, bf16 (default: f16)
#       --no-cache-f16          Do not cache intermediate F16/BF16 GGUF
#       --threads N             Threads for llama-quantize (default: available CPUs)
#       --smoke-test            Load output and generate tokens after quantization
#       --smoke-prompt STR      Smoke test prompt (default: Hello)
#       --smoke-tokens N        Smoke test token count (default: 8)
#       --smoke-ngl N           GPU layers for smoke test (default: 0)
#       --require-smoke-pass    Fail the run if the smoke test fails (implies --smoke-test)
#       --lock-timeout N        Lock wait seconds (0=fail-fast, -1=unlimited, default: 0)
#       --json                  JSON success/exists output; logs go to stderr
#       --json-strict           Like --json, and failures also emit JSON to stdout
#
# Environment:
#   MODEL_CACHE_DIR          HF cache root
#   QUANTIZED_OUTPUT_DIR     Output root
#   WRITE_LOCAL_REPO_LAYOUT  1=repo-like output layout, 0=flat snapshot layout
#   FLOX_ENV_CACHE           Cache root for shared F16/BF16 staging

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

slugify() { echo "${1%/}" | sed 's|/|--|g'; }

file_size_bytes() {
  python3 - "$1" <<'PY'
import os, sys
try:
    print(os.path.getsize(sys.argv[1]))
except OSError:
    print(0)
PY
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
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" <<'PY'
import hashlib, json, sys
(
    model_id,
    source_commit,
    quant_type,
    convert_type,
    imatrix_sha,
    script_sha,
    llama_ver,
    converter_sha,
    quantize_sha,
    gguf_ver,
    trust_remote_code,
) = sys.argv[1:12]
fp = {
    "model_id": model_id,
    "source_commit": source_commit or None,
    "quant": {
        "scheme": "gguf",
        "type": quant_type,
        "convert_type": convert_type,
        "imatrix_sha256": imatrix_sha or None,
        "trust_remote_code": trust_remote_code == "1",
    },
    "build": {
        "script_sha256": script_sha,
        "llama_cpp_version": llama_ver,
        "convert_hf_to_gguf_sha256": converter_sha,
        "llama_quantize_sha256": quantize_sha,
        "gguf_python_version": gguf_ver,
    },
}
canonical = json.dumps(fp, sort_keys=True, separators=(",", ":"))
snap_id = hashlib.sha256(canonical.encode()).hexdigest()[:40]
print(json.dumps(fp, indent=2, sort_keys=True))
print(snap_id)
PY
}

LAST_VALIDATE_REASON=""
LAST_VALIDATED_GGUF_SIZE="0"
LAST_VALIDATED_GGUF_SHA256=""

validate_gguf_file() {
  local path="$1"
  local with_sha="${2:-0}"
  local result
  if ! result="$(python3 - "$path" "$with_sha" <<'PY'
import hashlib
import os
import struct
import sys

path = sys.argv[1]
need_sha = sys.argv[2] == "1"

TYPE_UINT8 = 0
TYPE_INT8 = 1
TYPE_UINT16 = 2
TYPE_INT16 = 3
TYPE_UINT32 = 4
TYPE_INT32 = 5
TYPE_FLOAT32 = 6
TYPE_BOOL = 7
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT64 = 10
TYPE_INT64 = 11
TYPE_FLOAT64 = 12

FIXED_SIZES = {
    TYPE_UINT8: 1,
    TYPE_INT8: 1,
    TYPE_UINT16: 2,
    TYPE_INT16: 2,
    TYPE_UINT32: 4,
    TYPE_INT32: 4,
    TYPE_FLOAT32: 4,
    TYPE_BOOL: 1,
    TYPE_UINT64: 8,
    TYPE_INT64: 8,
    TYPE_FLOAT64: 8,
}
SCALAR_FORMATS = {
    TYPE_UINT8: '<B',
    TYPE_INT8: '<b',
    TYPE_UINT16: '<H',
    TYPE_INT16: '<h',
    TYPE_UINT32: '<I',
    TYPE_INT32: '<i',
    TYPE_FLOAT32: '<f',
    TYPE_BOOL: '<?',
    TYPE_UINT64: '<Q',
    TYPE_INT64: '<q',
    TYPE_FLOAT64: '<d',
}

def fail(msg):
    print(msg)
    raise SystemExit(1)

def read_exact(fh, n):
    data = fh.read(n)
    if len(data) != n:
        fail(f"truncated GGUF while reading {n} bytes at offset {fh.tell() - len(data)}")
    return data

def read_u32(fh):
    return struct.unpack('<I', read_exact(fh, 4))[0]

def read_u64(fh):
    return struct.unpack('<Q', read_exact(fh, 8))[0]

def read_string(fh):
    length = read_u64(fh)
    if length > 1024 * 1024:
        fail(f"implausible GGUF string length: {length}")
    raw = read_exact(fh, length)
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        fail(f"invalid UTF-8 in GGUF string: {exc}")

def read_scalar(fh, value_type):
    raw = read_exact(fh, FIXED_SIZES[value_type])
    return struct.unpack(SCALAR_FORMATS[value_type], raw)[0]

def skip_or_read_value(fh, value_type, depth=0):
    if depth > 8:
        fail("GGUF metadata nesting too deep")
    if value_type in FIXED_SIZES:
        return read_scalar(fh, value_type)
    if value_type == TYPE_STRING:
        return read_string(fh)
    if value_type == TYPE_ARRAY:
        elem_type = read_u32(fh)
        count = read_u64(fh)
        if count > 100_000_000:
            fail(f"implausible GGUF array length: {count}")
        if elem_type in FIXED_SIZES:
            read_exact(fh, FIXED_SIZES[elem_type] * count)
            return None
        vals = []
        for _ in range(count):
            vals.append(skip_or_read_value(fh, elem_type, depth + 1))
        return vals
    fail(f"unsupported GGUF metadata value type: {value_type}")

try:
    st = os.stat(path)
except OSError as exc:
    fail(f"cannot stat GGUF file: {exc}")
if st.st_size <= 0:
    fail("GGUF file is empty")

with open(path, 'rb') as fh:
    magic = read_exact(fh, 4)
    if magic != b'GGUF':
        fail(f"GGUF magic mismatch: {magic!r}")
    version = read_u32(fh)
    if version < 1 or version > 10:
        fail(f"unsupported or implausible GGUF version: {version}")
    tensor_count = read_u64(fh)
    kv_count = read_u64(fh)
    if tensor_count > 10_000_000:
        fail(f"implausible tensor count: {tensor_count}")
    if kv_count > 10_000_000:
        fail(f"implausible metadata count: {kv_count}")

    alignment = 32
    for _ in range(kv_count):
        key = read_string(fh)
        if not key:
            fail("empty GGUF metadata key")
        value_type = read_u32(fh)
        value = skip_or_read_value(fh, value_type)
        if key == 'general.alignment' and isinstance(value, int):
            alignment = value

    if alignment <= 0 or alignment > 65536 or (alignment & (alignment - 1)) != 0:
        fail(f"invalid GGUF alignment: {alignment}")

    offsets = []
    for _ in range(tensor_count):
        name = read_string(fh)
        if not name:
            fail("empty tensor name in GGUF")
        n_dims = read_u32(fh)
        if n_dims > 16:
            fail(f"implausible tensor rank: {n_dims}")
        for _ in range(n_dims):
            dim = read_u64(fh)
            if dim == 0:
                fail(f"tensor {name!r} has zero dimension")
        _ = read_u32(fh)  # ggml type
        offset = read_u64(fh)
        offsets.append(offset)

    tensor_info_end = fh.tell()
    data_offset = ((tensor_info_end + alignment - 1) // alignment) * alignment
    if data_offset > st.st_size:
        fail(f"GGUF tensor data offset beyond file end: {data_offset} > {st.st_size}")
    data_size = st.st_size - data_offset
    if tensor_count > 0 and data_size <= 0:
        fail("GGUF has tensors but no tensor data region")
    for offset in offsets:
        if offset >= data_size:
            fail(f"tensor data offset out of bounds: {offset} >= {data_size}")

if need_sha:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    digest = h.hexdigest()
else:
    digest = ""

print(f"{st.st_size}\t{digest}")
PY
)"; then
    LAST_VALIDATE_REASON="${result:-GGUF validation failed}"
    LAST_VALIDATED_GGUF_SIZE="0"
    LAST_VALIDATED_GGUF_SHA256=""
    return 1
  fi

  LAST_VALIDATE_REASON=""
  IFS=$'\t' read -r LAST_VALIDATED_GGUF_SIZE LAST_VALIDATED_GGUF_SHA256 <<< "$result"
  LAST_VALIDATED_GGUF_SIZE="${LAST_VALIDATED_GGUF_SIZE:-0}"
  LAST_VALIDATED_GGUF_SHA256="${LAST_VALIDATED_GGUF_SHA256:-}"
  return 0
}

validate_output_dir() {
  local out_dir="$1"
  local expected_fingerprint="$2"
  local gguf_path="$out_dir/$GGUF_FILENAME"

  validate_gguf_file "$gguf_path" 1 || {
    LAST_VALIDATE_REASON="artifact validation failed: $LAST_VALIDATE_REASON"
    return 1
  }

  local gguf_size="$LAST_VALIDATED_GGUF_SIZE"
  local gguf_sha256="$LAST_VALIDATED_GGUF_SHA256"
  local result

  if ! result="$(python3 - "$out_dir/FINGERPRINT.json" "$out_dir/gguf_quantize_info.json" "$expected_fingerprint" "$MODEL_ID" "$QUANT_TYPE" "$CONVERT_TYPE" "$GGUF_FILENAME" "$gguf_size" "$gguf_sha256" <<'PY'
import json, sys
fp_path, meta_path, expected_fp_text, model_id, quant_type, convert_type, gguf_filename, gguf_size, gguf_sha256 = sys.argv[1:10]

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

expected_fp = json.loads(expected_fp_text)
actual_fp = load_json(fp_path, "fingerprint")
if canon(actual_fp) != canon(expected_fp):
    fail("stored fingerprint does not match requested build fingerprint")

meta = load_json(meta_path, "metadata")
if meta.get("model_id") != model_id:
    fail(f"metadata model_id mismatch: {meta.get('model_id')!r}")
if meta.get("quant_type") != quant_type:
    fail(f"metadata quant_type mismatch: {meta.get('quant_type')!r}")
if meta.get("convert_type") != convert_type:
    fail(f"metadata convert_type mismatch: {meta.get('convert_type')!r}")
if meta.get("gguf_filename") != gguf_filename:
    fail(f"metadata gguf_filename mismatch: {meta.get('gguf_filename')!r}")
expected_commit = expected_fp.get("source_commit")
meta_commit = meta.get("source_commit")
if (meta_commit or None) != (expected_commit or None):
    fail(f"metadata source_commit mismatch: {meta_commit!r}")
try:
    meta_size = int(meta.get("gguf_size_bytes", -1))
except Exception:
    fail(f"metadata gguf_size_bytes is not an integer: {meta.get('gguf_size_bytes')!r}")
if meta_size != int(gguf_size):
    fail(f"metadata gguf_size_bytes mismatch: {meta_size} != {gguf_size}")
meta_sha = meta.get("artifact_sha256")
if not meta_sha:
    fail("metadata artifact_sha256 is missing")
if meta_sha != gguf_sha256:
    fail(f"metadata artifact_sha256 mismatch: {meta_sha!r}")
print(f"{gguf_size}\t{gguf_sha256}")
PY
)"; then
    LAST_VALIDATE_REASON="${result:-metadata validation failed}"
    LAST_VALIDATED_GGUF_SIZE="0"
    LAST_VALIDATED_GGUF_SHA256=""
    return 1
  fi

  LAST_VALIDATE_REASON=""
  IFS=$'\t' read -r LAST_VALIDATED_GGUF_SIZE LAST_VALIDATED_GGUF_SHA256 <<< "$result"
  LAST_VALIDATED_GGUF_SIZE="${LAST_VALIDATED_GGUF_SIZE:-0}"
  LAST_VALIDATED_GGUF_SHA256="${LAST_VALIDATED_GGUF_SHA256:-}"
  return 0
}

emit_exists_json() {
  local out_dir="$1"
  local gguf_size="${LAST_VALIDATED_GGUF_SIZE:-0}"
  local gguf_sha256="${LAST_VALIDATED_GGUF_SHA256:-}"
  python3 - "$MODEL_ID" "$QUANT_TYPE" "$out_dir" "$GGUF_FILENAME" "$gguf_size" "$gguf_sha256" <<'PY'
import json, sys, time
model_id, quant_type, out_path, gguf_file, gguf_size, gguf_sha256 = sys.argv[1:7]
print(json.dumps({
    "status": "exists",
    "source": model_id,
    "quant_type": quant_type,
    "output_path": out_path,
    "gguf_file": gguf_file,
    "gguf_size_bytes": int(gguf_size),
    "artifact_sha256": gguf_sha256,
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
      log "Existing output found and validated; will rebuild (--force): $out_dir"
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

VALID_QUANT_TYPES="Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_0 Q4_1 Q4_K_S Q4_K_M Q5_0 Q5_1 Q5_K_S Q5_K_M Q6_K Q8_0 F16 F32 IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_XS IQ3_S IQ4_XS IQ4_NL"

is_valid_quant_type() {
  local t="$1"
  for v in $VALID_QUANT_TYPES; do
    [[ "$v" == "$t" ]] && return 0
  done
  return 1
}

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
SMOKE_NGL=0
REQUIRE_SMOKE_PASS=0
LOCK_TIMEOUT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '/^# Usage:/,/^set -Eeuo pipefail/{ /^#/s/^# \{0,1\}//p; }' "$0"
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
    --smoke-ngl)            require_value "$1" "$#"; SMOKE_NGL="$2"; shift 2 ;;
    --smoke-ngl=*)          SMOKE_NGL="${1#*=}"; shift ;;
    --require-smoke-pass)   REQUIRE_SMOKE_PASS=1; SMOKE_TEST=1; shift ;;
    --lock-timeout)         require_value "$1" "$#"; LOCK_TIMEOUT="$2"; shift 2 ;;
    --lock-timeout=*)       LOCK_TIMEOUT="${1#*=}"; shift ;;
    --json)                 JSON_MODE=1; shift ;;
    --json-strict)          JSON_MODE=1; JSON_STRICT=1; shift ;;
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

[[ -n "$MODEL_ID" ]] || die "Missing <model-id>"
is_valid_quant_type "$QUANT_TYPE" || die "Unknown quant type: $QUANT_TYPE (supported: $VALID_QUANT_TYPES)"
case "$CONVERT_TYPE" in
  f16|bf16) ;;
  *) die "--convert-type must be f16 or bf16" ;;
esac

if [[ -n "$THREADS" ]]; then
  [[ "$THREADS" =~ ^[0-9]+$ ]] || die "--threads must be a positive integer"
  (( THREADS > 0 )) || die "--threads must be greater than 0"
fi
[[ "$SMOKE_TOKENS" =~ ^[0-9]+$ ]] || die "--smoke-tokens must be a positive integer"
(( SMOKE_TOKENS > 0 )) || die "--smoke-tokens must be greater than 0"
[[ "$SMOKE_NGL" =~ ^[0-9]+$ ]] || die "--smoke-ngl must be a non-negative integer"
[[ "$LOCK_TIMEOUT" =~ ^-?[0-9]+$ ]] || die "--lock-timeout must be an integer (0, >0, or -1)"
if [[ -n "$IMATRIX" ]]; then
  [[ -f "$IMATRIX" ]] || die "--imatrix file not found: $IMATRIX"
fi

CURRENT_STAGE="preflight"
have convert_hf_to_gguf.py || die "convert_hf_to_gguf.py not found on PATH. Install llama-cpp."
have llama-quantize || die "llama-quantize not found on PATH. Install llama-cpp."
if [[ "$SMOKE_TEST" == "1" ]]; then
  have llama-completion || die "llama-completion not found on PATH (needed for --smoke-test). Install llama-cpp."
fi
python3 -c 'import gguf' 2>/dev/null || die "Python 'gguf' module not found. Run: uv pip install gguf"
if [[ "$ONLINE" == "1" ]]; then
  python3 -c 'import huggingface_hub' 2>/dev/null || die "Python 'huggingface_hub' module not found. Run: uv pip install huggingface_hub"
fi

CONVERTER_BIN="$(command -v convert_hf_to_gguf.py)"
LLAMA_QUANTIZE_BIN="$(command -v llama-quantize)"
CONVERTER_SHA="$(sha256_file "$CONVERTER_BIN")"
LLAMA_QUANTIZE_SHA="$(sha256_file "$LLAMA_QUANTIZE_BIN")"
GGUF_PYTHON_VERSION="$(python_module_version gguf)"

LLAMA_CPP_VERSION=""
if have llama-completion; then
  LLAMA_CPP_VERSION="$(llama-completion --version 2>&1 | sed -nE 's/.*version:[[:space:]]*b?([0-9]+).*/\1/p' | head -1 2>/dev/null || true)"
fi
if [[ -z "$LLAMA_CPP_VERSION" ]]; then
  LLAMA_CPP_VERSION="$(llama-quantize 2>&1 | sed -nE 's/.*build[[:space:]]*=[[:space:]]*([0-9]+).*/\1/p' | head -1 || echo unknown)"
fi
[[ -n "$LLAMA_CPP_VERSION" ]] || LLAMA_CPP_VERSION="unknown"

if [[ -z "$THREADS" ]]; then
  THREADS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi
[[ "$THREADS" =~ ^[0-9]+$ ]] || die "--threads must be a positive integer"
(( THREADS > 0 )) || die "--threads must be greater than 0"

CACHE_DIR="${MODEL_CACHE_DIR:-./models}"
OUTPUT_DIR="${QUANTIZED_OUTPUT_DIR:-$CACHE_DIR}"
WRITE_LAYOUT="${WRITE_LOCAL_REPO_LAYOUT:-0}"
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

if [[ -z "$SUFFIX" ]]; then
  SUFFIX="-GGUF-${QUANT_TYPE}"
fi
OUTPUT_MODEL_ID="${MODEL_ID}${SUFFIX}"
OUTPUT_SLUG="$(slugify "$OUTPUT_MODEL_ID")"
MODEL_SHORT="$(basename "$MODEL_ID")"
GGUF_FILENAME="${MODEL_SHORT}-${QUANT_TYPE}.gguf"

_SRC_CACHE_KEY="$(echo "$MODEL_ID" | sed 's|/|--|g')"
_SRC_REFS="$CACHE_DIR/hub/models--${_SRC_CACHE_KEY}/refs/${REVISION}"
if [[ -f "$_SRC_REFS" ]]; then
  RESOLVED_COMMIT="$(head -1 "$_SRC_REFS" | tr -d '[:space:]')"
elif [[ "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  RESOLVED_COMMIT="$REVISION"
else
  RESOLVED_COMMIT=""
fi

_SRC_SNAP=""
if [[ -n "$RESOLVED_COMMIT" ]]; then
  _SRC_SNAP="$CACHE_DIR/hub/models--${_SRC_CACHE_KEY}/snapshots/${RESOLVED_COMMIT}"
  [[ -d "$_SRC_SNAP" ]] || _SRC_SNAP=""
fi
if [[ -z "$_SRC_SNAP" && "$ONLINE" == "0" ]]; then
  die "Source model not found in cache: $MODEL_ID (revision: $REVISION). Use --online to download."
fi

IMATRIX_SHA256=""
[[ -n "$IMATRIX" ]] && IMATRIX_SHA256="$(sha256_file "$IMATRIX")"
SCRIPT_SHA="$(sha256_file "$0")"

FINGERPRINT_JSON="$(build_fingerprint_json "$MODEL_ID" "$RESOLVED_COMMIT" "$QUANT_TYPE" "$CONVERT_TYPE" "$IMATRIX_SHA256" "$SCRIPT_SHA" "$LLAMA_CPP_VERSION" "$CONVERTER_SHA" "$LLAMA_QUANTIZE_SHA" "$GGUF_PYTHON_VERSION" "$TRUST_REMOTE_CODE")"
SNAP_REV="$(echo "$FINGERPRINT_JSON" | tail -1)"
FINGERPRINT_BODY="$(echo "$FINGERPRINT_JSON" | sed '$d')"

if [[ "$WRITE_LAYOUT" == "1" ]]; then
  MODEL_CACHE_KEY="$(echo "$OUTPUT_MODEL_ID" | sed 's|/|--|g')"
  FINAL_OUT="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/snapshots/${SNAP_REV}"
  REFS_DIR="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/refs"
else
  FINAL_OUT="$OUTPUT_DIR/${OUTPUT_SLUG}/${SNAP_REV}"
  REFS_DIR=""
fi

CURRENT_STAGE="locking"
have flock || die "flock not found. Install util-linux (or equivalent) for safe concurrent use."
LOCK_FILE="$OUTPUT_DIR/.quantize-${OUTPUT_SLUG}.lock"
exec 9>"$LOCK_FILE"
acquire_lock_fd 9 "$LOCK_FILE" "$LOCK_TIMEOUT"
maybe_exit_if_output_exists "$FINAL_OUT" "$FINGERPRINT_BODY"

export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
if [[ "$ONLINE" == "0" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
export PYTHONWARNINGS=ignore

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
log "  converter SHA:  ${CONVERTER_SHA:0:16}"
log "  quantize SHA:   ${LLAMA_QUANTIZE_SHA:0:16}"
log "  gguf module:    $GGUF_PYTHON_VERSION"
if [[ -n "$IMATRIX" ]]; then
  log "  imatrix:        $IMATRIX"
  log "  imatrix SHA:    $IMATRIX_SHA256"
fi
log "  Lock:           $LOCK_FILE"
log "  Lock wait:      $LOCK_TIMEOUT"
log "  F16 caching:    $CACHE_F16"
if [[ "$SMOKE_TEST" == "1" ]]; then
  log "  Smoke NGL:      $SMOKE_NGL"
  if [[ "$REQUIRE_SMOKE_PASS" == "1" ]]; then
    log "  Smoke gate:     required"
  fi
fi
log ""

CURRENT_STAGE="source-resolution"
if [[ -z "$_SRC_SNAP" ]]; then
  if [[ "$ONLINE" == "1" ]]; then
    log "Downloading model (online mode)..."
    python3 - "$MODEL_ID" "$REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download
model_id, revision = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=model_id, revision=revision, local_files_only=False)
PY
    if [[ -f "$_SRC_REFS" ]]; then
      RESOLVED_COMMIT="$(head -1 "$_SRC_REFS" | tr -d '[:space:]')"
    fi
    _SRC_SNAP="$CACHE_DIR/hub/models--${_SRC_CACHE_KEY}/snapshots/${RESOLVED_COMMIT}"
  fi
  [[ -d "$_SRC_SNAP" ]] || die "Source snapshot not found: $_SRC_SNAP"
fi

log "Source snapshot: $_SRC_SNAP"
log ""

if [[ "$(echo "$FINGERPRINT_BODY" | python3 -c 'import json,sys; print(json.load(sys.stdin)["source_commit"])')" != "$RESOLVED_COMMIT" ]]; then
  FINGERPRINT_JSON="$(build_fingerprint_json "$MODEL_ID" "$RESOLVED_COMMIT" "$QUANT_TYPE" "$CONVERT_TYPE" "$IMATRIX_SHA256" "$SCRIPT_SHA" "$LLAMA_CPP_VERSION" "$CONVERTER_SHA" "$LLAMA_QUANTIZE_SHA" "$GGUF_PYTHON_VERSION" "$TRUST_REMOTE_CODE")"
  SNAP_REV="$(echo "$FINGERPRINT_JSON" | tail -1)"
  FINGERPRINT_BODY="$(echo "$FINGERPRINT_JSON" | sed '$d')"
  if [[ "$WRITE_LAYOUT" == "1" ]]; then
    FINAL_OUT="$OUTPUT_DIR/hub/models--${MODEL_CACHE_KEY}/snapshots/${SNAP_REV}"
  else
    FINAL_OUT="$OUTPUT_DIR/${OUTPUT_SLUG}/${SNAP_REV}"
  fi
  maybe_exit_if_output_exists "$FINAL_OUT" "$FINGERPRINT_BODY"
fi

T_START="$(date +%s.%N)"

CURRENT_STAGE="f16-conversion"
F16_CACHE_DIR="${FLOX_ENV_CACHE:-$CACHE_DIR/.gguf-cache}/gguf-staging"
F16_SLUG="$(slugify "$MODEL_ID")"
GGUF_VERSION_SLUG="$(printf '%s' "$GGUF_PYTHON_VERSION" | tr -c 'A-Za-z0-9._-' '_')"
F16_CACHE_KEY="${RESOLVED_COMMIT}-${CONVERT_TYPE}-trc${TRUST_REMOTE_CODE}-cv${CONVERTER_SHA:0:16}-gv${GGUF_VERSION_SLUG}"
F16_CACHE_PATH="$F16_CACHE_DIR/${F16_SLUG}/${F16_CACHE_KEY}"
F16_LOCKS_DIR="$F16_CACHE_DIR/.locks"
F16_LOCK_FILE="$F16_LOCKS_DIR/${F16_SLUG}-${F16_CACHE_KEY}.lock"
F16_GGUF_NAME="${MODEL_SHORT}-${CONVERT_TYPE^^}.gguf"
F16_GGUF=""
F16_BUILD_OUT=""

if [[ "$CACHE_F16" == "1" ]]; then
  mkdir -p "$F16_CACHE_PATH" "$F16_LOCKS_DIR"
  exec 8>"$F16_LOCK_FILE"
  acquire_lock_fd 8 "$F16_LOCK_FILE" "$LOCK_TIMEOUT"

  if [[ -f "$F16_CACHE_PATH/$F16_GGUF_NAME" ]]; then
    if validate_gguf_file "$F16_CACHE_PATH/$F16_GGUF_NAME" 0; then
      log "Using cached ${CONVERT_TYPE^^} GGUF: $F16_CACHE_PATH/$F16_GGUF_NAME"
      F16_GGUF="$F16_CACHE_PATH/$F16_GGUF_NAME"
    else
      log "Ignoring cached ${CONVERT_TYPE^^} GGUF at $F16_CACHE_PATH/$F16_GGUF_NAME: $LAST_VALIDATE_REASON"
      rm -f "$F16_CACHE_PATH/$F16_GGUF_NAME"
      F16_BUILD_OUT="$TMP_OUT/${F16_GGUF_NAME}.cachebuild"
      F16_GGUF="$F16_CACHE_PATH/$F16_GGUF_NAME"
    fi
  else
    F16_BUILD_OUT="$TMP_OUT/${F16_GGUF_NAME}.cachebuild"
    F16_GGUF="$F16_CACHE_PATH/$F16_GGUF_NAME"
  fi
else
  F16_GGUF="$TMP_OUT/$F16_GGUF_NAME"
  F16_BUILD_OUT="$F16_GGUF"
fi

if [[ -n "$F16_BUILD_OUT" ]]; then
  log "Converting HF model to ${CONVERT_TYPE^^} GGUF..."
  CONVERT_ARGS=("$_SRC_SNAP" "--outfile" "$F16_BUILD_OUT" "--outtype" "$CONVERT_TYPE")
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
  [[ -f "$F16_BUILD_OUT" ]] || die "convert_hf_to_gguf.py did not produce expected output: $F16_BUILD_OUT"
  validate_gguf_file "$F16_BUILD_OUT" 0 || die "Invalid ${CONVERT_TYPE^^} GGUF produced by convert_hf_to_gguf.py: $LAST_VALIDATE_REASON"
  if [[ "$CACHE_F16" == "1" ]]; then
    mv "$F16_BUILD_OUT" "$F16_GGUF"
  fi
fi

[[ -f "$F16_GGUF" ]] || die "Expected ${CONVERT_TYPE^^} GGUF not found: $F16_GGUF"
validate_gguf_file "$F16_GGUF" 0 || die "Invalid cached ${CONVERT_TYPE^^} GGUF: $LAST_VALIDATE_REASON"
_f16_size="$(file_size_bytes "$F16_GGUF")"
_f16_size_gb="$(python3 -c "import sys; print(f'{int(sys.argv[1])/(1024**3):.2f}')" "$_f16_size")"
log "  ${CONVERT_TYPE^^} GGUF: $F16_GGUF (${_f16_size_gb} GB)"
log ""
if [[ "$CACHE_F16" == "1" ]]; then
  flock -u 8 || true
  exec 8>&-
fi

CURRENT_STAGE="quantization"
QUANT_OUTPUT="$TMP_OUT/$GGUF_FILENAME"
log "Quantizing ${CONVERT_TYPE^^} → $QUANT_TYPE..."
QUANT_ARGS=()
if [[ -n "$IMATRIX" ]]; then
  QUANT_ARGS+=("--imatrix" "$IMATRIX")
fi
QUANT_ARGS+=("$F16_GGUF" "$QUANT_OUTPUT" "$QUANT_TYPE" "$THREADS")
log "  Running: llama-quantize ${QUANT_ARGS[*]}"
if [[ "$JSON_MODE" == "1" ]]; then
  llama-quantize "${QUANT_ARGS[@]}" >&2
else
  llama-quantize "${QUANT_ARGS[@]}"
fi
[[ -f "$QUANT_OUTPUT" ]] || die "llama-quantize did not produce expected output: $QUANT_OUTPUT"
validate_gguf_file "$QUANT_OUTPUT" 1 || die "Invalid quantized GGUF produced by llama-quantize: $LAST_VALIDATE_REASON"
GGUF_SIZE="$LAST_VALIDATED_GGUF_SIZE"
GGUF_SHA256="$LAST_VALIDATED_GGUF_SHA256"
GGUF_SIZE_GB="$(python3 -c "import sys; print(f'{int(sys.argv[1])/(1024**3):.2f}')" "$GGUF_SIZE")"
log "  Quantized GGUF: $QUANT_OUTPUT (${GGUF_SIZE_GB} GB)"
log "  Artifact SHA:   $GGUF_SHA256"
log ""

CURRENT_STAGE="artifact-validation"
echo "$FINGERPRINT_BODY" > "$TMP_OUT/FINGERPRINT.json"
T_END="$(date +%s.%N)"
ELAPSED="$(python3 -c "import sys; print(round(float(sys.argv[1]) - float(sys.argv[2]), 3))" "$T_END" "$T_START")"
export TMP_OUT
python3 - "$MODEL_ID" "$QUANT_TYPE" "$CONVERT_TYPE" "$GGUF_FILENAME" \
  "$GGUF_SIZE" "$ELAPSED" "$LLAMA_CPP_VERSION" "$RESOLVED_COMMIT" \
  "$IMATRIX" "$IMATRIX_SHA256" "$THREADS" "$SNAP_REV" "$SCRIPT_SHA" \
  "$CONVERTER_SHA" "$LLAMA_QUANTIZE_SHA" "$GGUF_PYTHON_VERSION" "$TRUST_REMOTE_CODE" \
  "$GGUF_SHA256" <<'PY'
import json, os, platform, sys, time
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
snapshot_id = sys.argv[12]
script_sha = sys.argv[13]
converter_sha = sys.argv[14]
llama_quantize_sha = sys.argv[15]
gguf_version = sys.argv[16]
trust_remote_code = sys.argv[17] == "1"
artifact_sha256 = sys.argv[18]

meta = {
    "model_id": model_id,
    "source_commit": source_commit,
    "quant_type": quant_type,
    "convert_type": convert_type,
    "gguf_filename": gguf_filename,
    "gguf_size_bytes": gguf_size,
    "artifact_sha256": artifact_sha256,
    "imatrix": imatrix_path,
    "imatrix_sha256": imatrix_sha,
    "threads": threads,
    "elapsed_seconds": elapsed,
    "artifact_snapshot_id": snapshot_id,
    "script_sha256": script_sha,
    "convert_hf_to_gguf_sha256": converter_sha,
    "llama_quantize_sha256": llama_quantize_sha,
    "gguf_python_version": gguf_version,
    "trust_remote_code": trust_remote_code,
    "llama_cpp_version": llama_ver,
    "platform": platform.platform(),
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(os.path.join(os.environ["TMP_OUT"], "gguf_quantize_info.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)
PY
log "Wrote metadata to $TMP_OUT/gguf_quantize_info.json"
validate_output_dir "$TMP_OUT" "$FINGERPRINT_BODY" || die "Refusing to publish invalid artifact: $LAST_VALIDATE_REASON"
GGUF_SIZE="$LAST_VALIDATED_GGUF_SIZE"
GGUF_SHA256="$LAST_VALIDATED_GGUF_SHA256"
GGUF_SIZE_GB="$(python3 -c "import sys; print(f'{int(sys.argv[1])/(1024**3):.2f}')" "$GGUF_SIZE")"
if [[ "$CACHE_F16" != "1" && -f "$TMP_OUT/$F16_GGUF_NAME" ]]; then
  rm -f "$TMP_OUT/$F16_GGUF_NAME"
fi

CURRENT_STAGE="smoke-test"
SMOKE_RAN=0
SMOKE_OK=0
if [[ "$SMOKE_TEST" == "1" ]]; then
  log "Running smoke test..."
  SMOKE_RAN=1
  if _smoke_out="$(llama-completion -m "$QUANT_OUTPUT" -p "$SMOKE_PROMPT" -n "$SMOKE_TOKENS" -ngl "$SMOKE_NGL" --no-display-prompt 2>/dev/null)"; then
    [[ "$JSON_MODE" != "1" && -n "$_smoke_out" ]] && echo "$_smoke_out"
    SMOKE_OK=1
    log "Smoke test: PASS (ngl=$SMOKE_NGL)"
  else
    if [[ "$REQUIRE_SMOKE_PASS" == "1" ]]; then
      die "Smoke test failed and --require-smoke-pass is set (ngl=$SMOKE_NGL)"
    fi
    log "WARNING: Smoke test failed (non-fatal, ngl=$SMOKE_NGL)"
  fi
  log ""
fi

CURRENT_STAGE="publish"
mkdir -p "$(dirname "$FINAL_OUT")"
if [[ -e "$FINAL_OUT" ]]; then
  log "Replacing existing output: $FINAL_OUT"
  safe_rm_rf "$FINAL_OUT"
fi
mv "$TMP_OUT" "$FINAL_OUT"
TMP_OUT=""
if [[ "$WRITE_LAYOUT" == "1" ]]; then
  mkdir -p "$REFS_DIR"
  write_repo_ref "$REFS_DIR" "$REVISION" "$SNAP_REV"
fi
log "Published: $FINAL_OUT"
log ""

T_FINAL="$(date +%s.%N)"
TOTAL_ELAPSED="$(python3 -c "import sys; print(round(float(sys.argv[1]) - float(sys.argv[2]), 3))" "$T_FINAL" "$T_START")"
CURRENT_STAGE="complete"
if [[ "$JSON_MODE" == "1" ]]; then
  python3 - "$MODEL_ID" "$QUANT_TYPE" "$FINAL_OUT" "$GGUF_FILENAME" \
    "$GGUF_SIZE" "$TOTAL_ELAPSED" "$SMOKE_RAN" "$SMOKE_OK" "$SMOKE_NGL" "$GGUF_SHA256" <<'PY'
import json, sys, time
model_id = sys.argv[1]
quant_type = sys.argv[2]
out_path = sys.argv[3]
gguf_file = sys.argv[4]
gguf_size = int(sys.argv[5])
elapsed = float(sys.argv[6])
smoke_ran = sys.argv[7] == "1"
smoke_ok = sys.argv[8] == "1"
smoke_ngl = int(sys.argv[9])
artifact_sha256 = sys.argv[10]
print(json.dumps({
    "status": "ok",
    "source": model_id,
    "quant_type": quant_type,
    "output_path": out_path,
    "gguf_file": gguf_file,
    "gguf_size_bytes": gguf_size,
    "artifact_sha256": artifact_sha256,
    "elapsed_seconds": elapsed,
    "smoke_test": {"ran": smoke_ran, "ok": smoke_ok, "ngl": smoke_ngl},
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}))
PY
else
  log "=== Done ==="
  log "  Source:       $MODEL_ID"
  log "  Type:         $QUANT_TYPE"
  log "  Output:       $FINAL_OUT"
  log "  GGUF:         $GGUF_FILENAME"
  log "  Size:         ${GGUF_SIZE_GB} GB"
  log "  Artifact SHA: $GGUF_SHA256"
  log "  Elapsed:      ${TOTAL_ELAPSED}s"
  if [[ "$SMOKE_RAN" == "1" ]]; then
    if [[ "$SMOKE_OK" == "1" ]]; then
      log "  Smoke test:   PASS (ngl=${SMOKE_NGL})"
    else
      log "  Smoke test:   FAIL (ngl=${SMOKE_NGL})"
    fi
  fi
fi
