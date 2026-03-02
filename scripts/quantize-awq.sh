#!/usr/bin/env bash
# quantize-awq-prod-hardened.sh — Quantize a Hugging Face model to AWQ with hardened behavior.
#
# Usage:
#   quantize-awq-prod-hardened.sh [--json] [--smoke-test {full|fast|off}] <model-id> [bits] [group-size]
#
# Environment variables:
#   MODEL_CACHE_DIR              Base HF cache dir (default: ./models)
#   QUANTIZED_OUTPUT_DIR         Base output dir (default: ${MODEL_CACHE_DIR}/quantized)
#   MODEL_REVISION               HF revision (tag/branch/commit), default: main
#   HF_OFFLINE                   1 cache-only loads, default: 1
#   TRUST_REMOTE_CODE            1 enables trust_remote_code, default: 0
#   FORCE_REQUANTIZE             1 removes existing output, default: 0
#   SHOW_SIZES                   1 prints size stats, default: 0
#   PYTHON                       Python executable (default: python3)
#
# Execution policy:
#   REQUIRE_CUDA                 1 fails if CUDA is unavailable, default: 1
#   DEVICE_MAP                   auto|cuda0|cpu
#                               default: cuda0 if CUDA available, else cpu
#
# Output structure:
#   WRITE_LOCAL_REPO_LAYOUT      1 writes cache-shaped layout under OUTPUT_DIR/hub/models--... (default: 0)
#                               0 writes plain output under OUTPUT_DIR/<label>/<snapshot-id>
#
# Locking:
#   LOCK_TIMEOUT_SECONDS         seconds to wait for lock (default: 0, fail immediately)
#   LOCK_METHOD                  auto|flock|mkdir  (default: auto)
#   LOCK_STALE_SECONDS           if >0, stale lockdir can be removed when older than this and pid is dead,
#                               or pid was reused (linux /proc starttime mismatch) (default: 0)
#   ALLOW_REMOTE_STALE           0|1 allow stale-lock cleanup across hosts on shared storage (default: 0)
#
# Calibration knobs (optional; passed only if set; script fails if AWQ rejects them):
#   AWQ_CALIB_DATASET            string
#   AWQ_MAX_CALIB_SAMPLES        integer
#   AWQ_MAX_CALIB_SEQ_LEN        integer
#   AWQ_N_PARALLEL_CALIB_SAMPLES integer
#
# Back-compat aliases:
#   AWQ_CALIB_SAMPLES            mapped to AWQ_MAX_CALIB_SAMPLES if set
#   AWQ_CALIB_SEQ_LEN            mapped to AWQ_MAX_CALIB_SEQ_LEN if set
#
# Determinism knobs:
#   QUANT_SEED                   integer seed (default: 1337)
#   DETERMINISTIC                1 requests deterministic algorithms (default: 0)
#   DETERMINISTIC_FAIL_CLOSED    1 makes determinism setup errors fatal, 0 logs warning and continues (default: 1)
#
# Save knobs (fail-closed: script exits if AutoAWQ rejects these kwargs):
#   QUANT_SHARD_SIZE             shard size for save_quantized (default: 5GB)
#   QUANT_SAFETENSORS            0|1 save safetensors (default: 1)
#
# Validation:
#   --smoke-test full            reload saved artifact and run forward + tiny generate
#   --smoke-test fast            forward + tiny generate on in-memory model after quantize
#   --smoke-test off             skip forward/generate (still checks required files exist)
#   SMOKE_TEST_MODE              same as flag (default: full)
#
# Integrity (opt-in):
#   WRITE_CHECKSUMS              0|1 write FILES_SHA256.json (default: 0)
#   WRITE_WEIGHT_CHECKSUMS       0|1 include weight shards in checksums (default: 0)
#
# Fingerprint composition:
#   FINGERPRINT_INCLUDE_VERS     0|1 include torch/transformers/awq versions (default: 1)
#   FINGERPRINT_INCLUDE_SYS      0|1 include host/sys info (default: 0)

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN:  $*" >&2; }

on_err() {
  local rc=$?
  local cmd="${BASH_COMMAND}"
  if (( ${#cmd} > 200 )); then
    echo "ERROR: command failed (rc=$rc) at ${BASH_SOURCE[0]}:${BASH_LINENO[0]}" >&2
  else
    echo "ERROR: command failed (rc=$rc) at ${BASH_SOURCE[0]}:${BASH_LINENO[0]}: ${cmd}" >&2
  fi
  exit "$rc"
}
trap on_err ERR

usage() { sed -n '2,65s/^# \?//p' "$0"; }

# --------------------------
# Parse flags
# --------------------------
JSON_MODE=0
SMOKE_TEST_MODE="${SMOKE_TEST_MODE:-full}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --json) JSON_MODE=1; shift ;;
    --smoke-test)
      [[ $# -ge 2 ]] || die "--smoke-test requires: full|fast|off"
      SMOKE_TEST_MODE="$2"
      shift 2
      ;;
    --smoke-test=*)
      SMOKE_TEST_MODE="${1#*=}"
      shift
      ;;
    --) shift; break ;;
    --*) die "Unknown option: $1" ;;
    *) break ;;
  esac
done

case "${SMOKE_TEST_MODE,,}" in
  full|fast|off) ;;
  *) die "Unknown smoke test mode: $SMOKE_TEST_MODE (use full|fast|off)" ;;
esac

[[ $# -ge 1 ]] || { usage; exit 2; }

MODEL_ID="$1"
BITS="${2:-4}"
GROUP_SIZE="${3:-128}"

[[ "$BITS" =~ ^[0-9]+$ ]]       || die "bits must be an integer (got: $BITS)"
[[ "$GROUP_SIZE" =~ ^[0-9]+$ ]] || die "group-size must be an integer (got: $GROUP_SIZE)"
(( BITS >= 2 && BITS <= 8 ))    || die "bits must be between 2 and 8 (got: $BITS)"
(( GROUP_SIZE > 0 ))            || die "group-size must be > 0 (got: $GROUP_SIZE)"

CACHE_DIR="${MODEL_CACHE_DIR:-./models}"
OUTPUT_DIR="${QUANTIZED_OUTPUT_DIR:-$CACHE_DIR/quantized}"
REVISION="${MODEL_REVISION:-main}"
HF_OFFLINE="${HF_OFFLINE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
FORCE_REQUANTIZE="${FORCE_REQUANTIZE:-0}"
SHOW_SIZES="${SHOW_SIZES:-0}"
PYTHON="${PYTHON:-python3}"

REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
DEVICE_MAP="${DEVICE_MAP:-}"   # resolved later
WRITE_LOCAL_REPO_LAYOUT="${WRITE_LOCAL_REPO_LAYOUT:-0}"

LOCK_TIMEOUT_SECONDS="${LOCK_TIMEOUT_SECONDS:-0}"
LOCK_METHOD="${LOCK_METHOD:-auto}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-0}"
ALLOW_REMOTE_STALE="${ALLOW_REMOTE_STALE:-0}"

QUANT_SEED="${QUANT_SEED:-1337}"
DETERMINISTIC="${DETERMINISTIC:-0}"
DETERMINISTIC_FAIL_CLOSED="${DETERMINISTIC_FAIL_CLOSED:-1}"

QUANT_SHARD_SIZE="${QUANT_SHARD_SIZE:-5GB}"
QUANT_SAFETENSORS="${QUANT_SAFETENSORS:-1}"

WRITE_CHECKSUMS="${WRITE_CHECKSUMS:-0}"
WRITE_WEIGHT_CHECKSUMS="${WRITE_WEIGHT_CHECKSUMS:-0}"

FINGERPRINT_INCLUDE_VERS="${FINGERPRINT_INCLUDE_VERS:-1}"
FINGERPRINT_INCLUDE_SYS="${FINGERPRINT_INCLUDE_SYS:-0}"

# Back-compat calibration aliases
if [[ -n "${AWQ_CALIB_SAMPLES:-}" && -z "${AWQ_MAX_CALIB_SAMPLES:-}" ]]; then
  AWQ_MAX_CALIB_SAMPLES="${AWQ_CALIB_SAMPLES}"
fi
if [[ -n "${AWQ_CALIB_SEQ_LEN:-}" && -z "${AWQ_MAX_CALIB_SEQ_LEN:-}" ]]; then
  AWQ_MAX_CALIB_SEQ_LEN="${AWQ_CALIB_SEQ_LEN}"
fi

# Validate boolean-ish vars
for b in HF_OFFLINE TRUST_REMOTE_CODE FORCE_REQUANTIZE SHOW_SIZES REQUIRE_CUDA WRITE_LOCAL_REPO_LAYOUT \
         DETERMINISTIC DETERMINISTIC_FAIL_CLOSED QUANT_SAFETENSORS WRITE_CHECKSUMS WRITE_WEIGHT_CHECKSUMS \
         FINGERPRINT_INCLUDE_VERS FINGERPRINT_INCLUDE_SYS ALLOW_REMOTE_STALE; do
  v="${!b}"
  [[ "$v" == "0" || "$v" == "1" ]] || die "$b must be 0 or 1"
done
[[ "$LOCK_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || die "LOCK_TIMEOUT_SECONDS must be an integer (got: $LOCK_TIMEOUT_SECONDS)"
[[ "$LOCK_STALE_SECONDS" =~ ^[0-9]+$ ]] || die "LOCK_STALE_SECONDS must be an integer (got: $LOCK_STALE_SECONDS)"
[[ "$LOCK_METHOD" == "auto" || "$LOCK_METHOD" == "flock" || "$LOCK_METHOD" == "mkdir" ]] || die "LOCK_METHOD must be auto|flock|mkdir"
[[ "$QUANT_SEED" =~ ^[0-9]+$ ]] || die "QUANT_SEED must be an integer (got: $QUANT_SEED)"

for v in AWQ_MAX_CALIB_SAMPLES AWQ_MAX_CALIB_SEQ_LEN AWQ_N_PARALLEL_CALIB_SAMPLES; do
  if [[ -n "${!v:-}" && ! "${!v}" =~ ^[0-9]+$ ]]; then
    die "$v must be an integer (got: ${!v})"
  fi
done

command -v "$PYTHON" >/dev/null 2>&1 || die "Cannot find Python executable: $PYTHON"

# Suppress noisy deprecation/future warnings from third-party libraries
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error

# Logging: keep stdout clean in --json mode
log() {
  if [[ "$JSON_MODE" == "1" ]]; then
    echo "$*" >&2
  else
    echo "$*"
  fi
}

# Script sha256
SCRIPT_PATH="$("$PYTHON" - <<'PY' "$0"
import os,sys
print(os.path.realpath(sys.argv[1]))
PY
)"
SCRIPT_SHA256="$(
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$SCRIPT_PATH" | awk '{print $1}'
  else
    "$PYTHON" - "$SCRIPT_PATH" <<'PY'
import hashlib, sys
p=sys.argv[1]
h=hashlib.sha256()
with open(p,'rb') as f:
    for chunk in iter(lambda: f.read(1024*1024), b''):
        h.update(chunk)
print(h.hexdigest())
PY
  fi
)"

MODEL_ID_ESC="${MODEL_ID//\//--}"
OUTPUT_MODEL_ID="${MODEL_ID}-AWQ"
OUTPUT_MODEL_ID_ESC="${OUTPUT_MODEL_ID//\//--}"

SOURCE_CACHE="$CACHE_DIR/hub/models--${MODEL_ID_ESC}"
SOURCE_REFS_DIR="$SOURCE_CACHE/refs"
SOURCE_SNAP_DIR="$SOURCE_CACHE/snapshots"

# Offline preflight: repo folder must exist.
if [[ "$HF_OFFLINE" == "1" && ! -d "$SOURCE_CACHE" ]]; then
  die "Source model not found in cache: $SOURCE_CACHE (set HF_OFFLINE=0 to allow downloads)"
fi

# Offline preflight for symbolic revision: require refs/<revision>
if [[ "$HF_OFFLINE" == "1" && ! "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  if [[ ! -f "$SOURCE_REFS_DIR/$REVISION" ]]; then
    die "Offline mode missing ref file: $SOURCE_REFS_DIR/$REVISION. Use a commit hash revision or prefetch refs for '$REVISION'."
  fi
fi

# Offline preflight: snapshot exists for resolved commit.
if [[ "$HF_OFFLINE" == "1" ]]; then
  if [[ "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then
    OFFLINE_COMMIT="$REVISION"
  else
    OFFLINE_COMMIT="$(cat -- "$SOURCE_REFS_DIR/$REVISION" 2>/dev/null || true)"
    [[ "$OFFLINE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "Offline ref file did not contain a 40-hex commit: $SOURCE_REFS_DIR/$REVISION"
  fi
  if [[ ! -d "$SOURCE_SNAP_DIR/$OFFLINE_COMMIT" ]]; then
    die "Offline mode missing snapshot payload: $SOURCE_SNAP_DIR/$OFFLINE_COMMIT (ref '$REVISION' -> $OFFLINE_COMMIT). Prefetch the model snapshot."
  fi
fi

# Resolve default device_map
RESOLVED_DEVICE_MAP="$DEVICE_MAP"
if [[ -z "$RESOLVED_DEVICE_MAP" ]]; then
  RESOLVED_DEVICE_MAP="$("$PYTHON" - <<'PY'
import torch
print("cuda0" if torch.cuda.is_available() else "cpu")
PY
)"
fi
case "${RESOLVED_DEVICE_MAP,,}" in
  auto|cuda0|cpu) ;;
  *) die "Unknown DEVICE_MAP='$RESOLVED_DEVICE_MAP'. Use auto|cuda0|cpu." ;;
esac

# VRAM check: if targeting GPU, verify enough free memory before proceeding.
if [[ "${RESOLVED_DEVICE_MAP,,}" != "cpu" ]]; then
  "$PYTHON" -c "
import sys, torch
if torch.cuda.is_available():
    try:
        free, total = torch.cuda.mem_get_info(0)
        free_mb, total_mb = free >> 20, total >> 20
        if free_mb < 512:
            print(f'GPU has only {free_mb} MB free of {total_mb} MB. '
                  f'Another process may be using the GPU. '
                  f'Free GPU memory or re-run with DEVICE_MAP=cpu.', file=sys.stderr)
            sys.exit(1)
    except SystemExit:
        raise
    except Exception:
        print('Failed to query GPU memory — VRAM may be fully consumed '
              'by another process. Free GPU memory or re-run with DEVICE_MAP=cpu.', file=sys.stderr)
        sys.exit(1)
" || exit 1
fi

# Dependency check:
# - Always: torch, transformers, awq
# - huggingface_hub: only when online + symbolic revision
# - accelerate: when DEVICE_MAP=auto
"$PYTHON" - "$HF_OFFLINE" "$REVISION" "${RESOLVED_DEVICE_MAP,,}" <<'PY'
import importlib.util, re, sys
offline = sys.argv[1] == "1"
revision = sys.argv[2]
device_map = sys.argv[3].strip().lower()
need_hub = (not offline) and (not re.fullmatch(r"[0-9a-f]{40}", revision))
base = ["torch", "transformers", "awq"] + (["huggingface_hub"] if need_hub else [])
missing = [m for m in base if importlib.util.find_spec(m) is None]
if missing:
    print("Missing Python modules:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
if device_map == "auto" and importlib.util.find_spec("accelerate") is None:
    print("Missing Python module: accelerate (required for DEVICE_MAP=auto).", file=sys.stderr)
    sys.exit(1)
PY

# Fingerprint config + resolve commit + compute snapshot id.
FINGERPRINT_TMP="$(mktemp -t awq-fingerprint.XXXXXX.json)"
IFS=' ' read -r SOURCE_COMMIT DTYPE_POLICY OUTPUT_SNAPSHOT_ID < <(
  "$PYTHON" - \
    "$MODEL_ID" "$REVISION" "$CACHE_DIR" "$HF_OFFLINE" \
    "$BITS" "$GROUP_SIZE" \
    "${RESOLVED_DEVICE_MAP,,}" "$TRUST_REMOTE_CODE" "$REQUIRE_CUDA" \
    "${AWQ_CALIB_DATASET:-}" "${AWQ_MAX_CALIB_SAMPLES:-}" "${AWQ_MAX_CALIB_SEQ_LEN:-}" "${AWQ_N_PARALLEL_CALIB_SAMPLES:-}" \
    "$SCRIPT_SHA256" \
    "$QUANT_SEED" "$DETERMINISTIC" \
    "$QUANT_SAFETENSORS" "$QUANT_SHARD_SIZE" \
    "$FINGERPRINT_INCLUDE_VERS" "$FINGERPRINT_INCLUDE_SYS" \
    "$FINGERPRINT_TMP" \
  <<'PY'
import hashlib, json, logging, os, platform, re, sys, warnings
logging.getLogger("torchao").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

(
  model_id, revision, cache_base, offline_s,
  bits_s, group_s,
  device_map, trust_s, require_cuda_s,
  calib_dataset, max_calib_samples_s, max_calib_seq_len_s, n_parallel_calib_samples_s,
  script_sha,
  seed_s, deterministic_s,
  safetensors_s, shard_size,
  include_vers_s, include_sys_s,
  out_path
) = sys.argv[1:23]

offline = offline_s == "1"
bits = int(bits_s)
group = int(group_s)
trust_remote_code = trust_s == "1"
require_cuda = require_cuda_s == "1"
seed = int(seed_s)
deterministic = deterministic_s == "1"
save_safetensors = safetensors_s == "1"
include_vers = include_vers_s == "1"
include_sys = include_sys_s == "1"

def repo_folder_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")

def read_ref_commit_from_cache() -> str | None:
    refs_dir = os.path.join(cache_base, "hub", repo_folder_name(model_id), "refs")
    ref_path = os.path.join(refs_dir, revision)
    if os.path.isfile(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            v = f.read().strip()
            return v or None
    return None

def resolve_commit() -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    cached = read_ref_commit_from_cache()
    if cached and re.fullmatch(r"[0-9a-f]{40}", cached):
        return cached
    if offline:
        raise SystemExit(
            f"Offline mode could not resolve a commit for revision '{revision}'. "
            f"Use a commit hash revision, or prefetch refs/{revision}."
        )
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(model_id, revision=revision)
    sha = getattr(info, "sha", None)
    if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"Could not resolve commit sha for {model_id}@{revision}")
    return sha

def safe_ver(modname: str):
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", None)
    except Exception:
        return None

commit = resolve_commit()

max_calib_samples = int(max_calib_samples_s) if max_calib_samples_s else None
max_calib_seq_len = int(max_calib_seq_len_s) if max_calib_seq_len_s else None
n_parallel_calib_samples = int(n_parallel_calib_samples_s) if n_parallel_calib_samples_s else None

import torch
cuda_ok = torch.cuda.is_available()
if require_cuda and not cuda_ok:
    raise SystemExit("CUDA is not available but REQUIRE_CUDA=1.")

device_map = device_map.strip().lower()
if device_map == "cpu":
    dtype_policy = "cpu"
else:
    if cuda_ok:
        bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        dtype_policy = "bf16" if bf16_ok else "fp16"
    else:
        dtype_policy = "cpu"

versions = None
sysinfo = None
if include_vers:
    versions = {
        "torch": safe_ver("torch"),
        "transformers": safe_ver("transformers"),
        "awq": safe_ver("awq"),
    }
if include_sys:
    sysinfo = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": cuda_ok,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if cuda_ok else None,
        "gpu_name": (torch.cuda.get_device_name(0) if cuda_ok else None),
    }

finger_cfg = {
    "model_id": model_id,
    "source_commit": commit,
    "quant": {
        "scheme": "awq",
        "w_bit": bits,
        "q_group_size": group,
        "zero_point": True,
        "version": "GEMM",
    },
    "calibration": {
        "calib_data": calib_dataset or None,
        "max_calib_samples": max_calib_samples,
        "max_calib_seq_len": max_calib_seq_len,
        "n_parallel_calib_samples": n_parallel_calib_samples,
    },
    "execution_policy": {
        "device_map": device_map,
        "dtype_policy": dtype_policy,
        "trust_remote_code": trust_remote_code,
        "require_cuda": require_cuda,
    },
    "determinism": {
        "seed": seed,
        "deterministic": deterministic,
    },
    "save": {
        "safetensors": save_safetensors,
        "shard_size": shard_size,
    },
    "build": {
        "script_sha256": script_sha,
        "include_versions": include_vers,
        "include_sysinfo": include_sys,
        "versions": versions,
        "sysinfo": sysinfo,
    },
}

fingerprint = json.dumps(finger_cfg, sort_keys=True, separators=(",", ":"))
snap = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(finger_cfg, f, indent=2, sort_keys=True)

print(commit, dtype_policy, snap)
PY
)

# Layout selection
if [[ "$WRITE_LOCAL_REPO_LAYOUT" == "1" ]]; then
  OUTPUT_ROOT="$OUTPUT_DIR/hub/models--${OUTPUT_MODEL_ID_ESC}"
  OUTPUT_SNAPSHOTS="$OUTPUT_ROOT/snapshots"
  OUTPUT_REFS="$OUTPUT_ROOT/refs"
  OUTPUT_PATH="$OUTPUT_SNAPSHOTS/$OUTPUT_SNAPSHOT_ID"
  REF_PATH="$OUTPUT_REFS/$REVISION"
  LAYOUT_DESC="cache-shaped"
else
  OUTPUT_ROOT="$OUTPUT_DIR/${OUTPUT_MODEL_ID_ESC}"
  OUTPUT_PATH="$OUTPUT_ROOT/$OUTPUT_SNAPSHOT_ID"
  REF_PATH=""
  LAYOUT_DESC="plain-dir"
fi

mkdir -p -- "$OUTPUT_ROOT"

# --------------------------
# Safety helper for rm -rf
# --------------------------
rm_rf_guarded() {
  local p="$1"
  [[ -n "$p" ]] || die "Refusing to rm -rf empty path"
  [[ "$p" != "/" ]] || die "Refusing to rm -rf path '/'"
  rm -rf -- "$p"
}

# --------------------------
# Locking (single cleanup handler releases lock)
# --------------------------
LOCK_HELD=0
LOCK_KIND=""   # "flock" or "mkdir"
LOCKFILE="$OUTPUT_ROOT/.quantize.lock"
LOCKDIR="$OUTPUT_ROOT/.quantize.lockdir"

release_lock() {
  if [[ "$LOCK_HELD" != "1" ]]; then
    return 0
  fi
  if [[ "$LOCK_KIND" == "mkdir" ]]; then
    rm -rf -- "$LOCKDIR" 2>/dev/null || true
  elif [[ "$LOCK_KIND" == "flock" ]]; then
    exec 9>&- 2>/dev/null || true
  fi
  LOCK_HELD=0
  LOCK_KIND=""
}

write_lock_info() {
  local host epoch pid ticks
  host="$(hostname 2>/dev/null || echo unknown)"
  epoch="$(date +%s)"
  pid="$$"
  ticks=""
  if [[ -r "/proc/$pid/stat" ]]; then
    ticks="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
  fi
  {
    echo "host=$host"
    echo "pid=$pid"
    echo "started_epoch=$epoch"
    echo "proc_start_ticks=${ticks}"
  } >"$LOCKDIR/lock.info" 2>/dev/null || true
}

stale_lockdir_should_remove() {
  [[ "$LOCK_STALE_SECONDS" -gt 0 ]] || return 1
  [[ -f "$LOCKDIR/lock.info" ]] || return 1

  local now host pid started proc_ticks age cur_ticks cur_host
  now="$(date +%s)"
  host="$(grep -E '^host=' "$LOCKDIR/lock.info" 2>/dev/null | head -n1 | cut -d= -f2 || true)"
  pid="$(grep -E '^pid=' "$LOCKDIR/lock.info" 2>/dev/null | head -n1 | cut -d= -f2 || true)"
  started="$(grep -E '^started_epoch=' "$LOCKDIR/lock.info" 2>/dev/null | head -n1 | cut -d= -f2 || true)"
  proc_ticks="$(grep -E '^proc_start_ticks=' "$LOCKDIR/lock.info" 2>/dev/null | head -n1 | cut -d= -f2 || true)"

  [[ "$started" =~ ^[0-9]+$ ]] || return 1
  age=$(( now - started ))
  (( age > LOCK_STALE_SECONDS )) || return 1

  # Shared storage protection: only touch same-host locks unless overridden.
  if [[ "$ALLOW_REMOTE_STALE" != "1" ]]; then
    cur_host="$(hostname 2>/dev/null || echo unknown)"
    if [[ -n "$host" && "$host" != "$cur_host" ]]; then
      return 1
    fi
  fi

  if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]]; then
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  if [[ -n "$proc_ticks" && "$proc_ticks" =~ ^[0-9]+$ && -r "/proc/$pid/stat" ]]; then
    cur_ticks="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
    if [[ -n "$cur_ticks" && "$cur_ticks" =~ ^[0-9]+$ && "$cur_ticks" != "$proc_ticks" ]]; then
      return 0
    fi
  fi

  return 1
}

acquire_lock() {
  local started waited
  started="$(date +%s)"

  while true; do
    if [[ "$LOCK_METHOD" == "flock" || "$LOCK_METHOD" == "auto" ]]; then
      if command -v flock >/dev/null 2>&1; then
        exec 9>"$LOCKFILE"
        if [[ "$LOCK_TIMEOUT_SECONDS" -gt 0 ]]; then
          if flock -w "$LOCK_TIMEOUT_SECONDS" 9; then
            LOCK_HELD=1
            LOCK_KIND="flock"
            return 0
          else
            return 1
          fi
        else
          if flock -n 9; then
            LOCK_HELD=1
            LOCK_KIND="flock"
            return 0
          else
            return 1
          fi
        fi
      fi
      if [[ "$LOCK_METHOD" == "flock" ]]; then
        die "LOCK_METHOD=flock but flock is not available"
      fi
    fi

    if mkdir "$LOCKDIR" 2>/dev/null; then
      write_lock_info
      LOCK_HELD=1
      LOCK_KIND="mkdir"
      return 0
    fi

    if stale_lockdir_should_remove; then
      warn "Removing stale lockdir: $LOCKDIR"
      rm -rf -- "$LOCKDIR" 2>/dev/null || true
      continue
    fi

    if [[ "$LOCK_TIMEOUT_SECONDS" -eq 0 ]]; then
      return 1
    fi
    waited=$(( $(date +%s) - started ))
    if (( waited >= LOCK_TIMEOUT_SECONDS )); then
      return 1
    fi
    sleep 1
  done
}

TMP_DIR=""
cleanup() {
  local rc=$?
  if [[ $rc -ne 0 && -n "$TMP_DIR" ]]; then
    rm -rf -- "$TMP_DIR" 2>/dev/null || true
  fi
  rm -f -- "$FINGERPRINT_TMP" 2>/dev/null || true
  release_lock
  exit "$rc"
}
trap cleanup EXIT INT TERM

if ! acquire_lock; then
  die "Lock busy: $OUTPUT_ROOT (timeout_s=${LOCK_TIMEOUT_SECONDS})"
fi

# --------------------------
# Existing output checks (tightened; tokenizer artifacts added)
# --------------------------
output_is_complete() {
  "$PYTHON" - "$1" "$OUTPUT_SNAPSHOT_ID" <<'PY'
import json, os, sys

out_dir = sys.argv[1]
expected_snap = sys.argv[2]

def has_any_weight(root: str) -> bool:
    for base, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".safetensors") or fn.endswith(".bin"):
                return True
    return False

def has_tokenizer(root: str) -> bool:
    # Require tokenizer_config.json and one core tokenizer payload file.
    if not os.path.isfile(os.path.join(root, "tokenizer_config.json")):
        return False
    return (
        os.path.isfile(os.path.join(root, "tokenizer.json")) or
        os.path.isfile(os.path.join(root, "tokenizer.model"))
    )

req = ["config.json", "awq_quantize_meta.json", "FINGERPRINT.json"]
for r in req:
    if not os.path.isfile(os.path.join(out_dir, r)):
        print("0")
        raise SystemExit(0)

if not has_any_weight(out_dir):
    print("0")
    raise SystemExit(0)

if not has_tokenizer(out_dir):
    print("0")
    raise SystemExit(0)

try:
    with open(os.path.join(out_dir, "awq_quantize_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    snap = meta.get("output_snapshot_id")
    if snap != expected_snap:
        print("0")
        raise SystemExit(0)
except Exception:
    print("0")
    raise SystemExit(0)

print("1")
PY
}

if [[ -d "$OUTPUT_PATH" ]]; then
  if [[ "$(output_is_complete "$OUTPUT_PATH")" == "1" ]]; then
    if [[ "$FORCE_REQUANTIZE" == "1" ]]; then
      log "Existing complete output found. FORCE_REQUANTIZE=1 removes it."
      rm_rf_guarded "$OUTPUT_PATH"
    else
      log "Output already exists (complete) at $OUTPUT_PATH"
      if [[ "$JSON_MODE" == "1" ]]; then
        "$PYTHON" - "$MODEL_ID" "$REVISION" "$SOURCE_COMMIT" "$OUTPUT_SNAPSHOT_ID" "$OUTPUT_PATH" "$REF_PATH" "$LAYOUT_DESC" <<'PY'
import json, sys
model_id, revision, source_commit, snap, out_path, ref_path, layout = sys.argv[1:8]
print(json.dumps({
  "status": "exists",
  "model_id": model_id,
  "revision": revision,
  "source_commit": source_commit,
  "output_snapshot_id": snap,
  "layout": layout,
  "output_path": out_path,
  "ref_path": ref_path if ref_path else None,
}))
PY
      fi
      exit 0
    fi
  else
    warn "Output directory exists but looks incomplete; removing: $OUTPUT_PATH"
    rm_rf_guarded "$OUTPUT_PATH"
  fi
fi

# Temp dir on same filesystem as destination parent (atomic mv).
DEST_PARENT="$(dirname -- "$OUTPUT_PATH")"
mkdir -p -- "$DEST_PARENT"
TMP_DIR="$(mktemp -d "$DEST_PARENT/.tmp.${OUTPUT_SNAPSHOT_ID}.XXXXXX")"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="$QUANT_SEED"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"

# --------------------------
# Quantize + save + validate
# --------------------------
"$PYTHON" - \
  "$MODEL_ID" "$REVISION" "$SOURCE_COMMIT" "$OUTPUT_SNAPSHOT_ID" \
  "$TMP_DIR" "$CACHE_DIR" "$HF_OFFLINE" "$TRUST_REMOTE_CODE" \
  "$REQUIRE_CUDA" "${RESOLVED_DEVICE_MAP,,}" "$DTYPE_POLICY" \
  "$BITS" "$GROUP_SIZE" \
  "${AWQ_CALIB_DATASET:-}" "${AWQ_MAX_CALIB_SAMPLES:-}" "${AWQ_MAX_CALIB_SEQ_LEN:-}" "${AWQ_N_PARALLEL_CALIB_SAMPLES:-}" \
  "$FINGERPRINT_TMP" "${SMOKE_TEST_MODE,,}" \
  "$QUANT_SEED" "$DETERMINISTIC" "$DETERMINISTIC_FAIL_CLOSED" \
  "$QUANT_SAFETENSORS" "$QUANT_SHARD_SIZE" \
  "$WRITE_CHECKSUMS" "$WRITE_WEIGHT_CHECKSUMS" \
<<'PY'
import hashlib, json, logging, os, random, re, sys, time, warnings
from datetime import datetime, timezone

logging.getLogger("torchao").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
warnings.filterwarnings("ignore")  # re-apply: AWQ __init__ overrides filters

try:
    import numpy as np
except Exception:
    np = None

(
  model_id, user_revision, source_commit, expected_snapshot_id,
  out_dir, cache_base, offline_s, trust_s,
  require_cuda_s, device_map_s, dtype_policy,
  bits_s, group_s,
  calib_dataset, max_calib_samples_s, max_calib_seq_len_s, n_parallel_calib_samples_s,
  fingerprint_path, smoke_mode,
  seed_s, deterministic_s, deterministic_fail_closed_s,
  safetensors_s, shard_size,
  write_checksums_s, write_weight_checksums_s,
) = sys.argv[1:27]

offline = offline_s == "1"
trust_remote_code = trust_s == "1"
require_cuda = require_cuda_s == "1"
device_map_s = device_map_s.strip().lower()
dtype_policy = dtype_policy.strip().lower()
bits = int(bits_s)
group_size = int(group_s)
smoke_mode = smoke_mode.strip().lower()
seed = int(seed_s)
deterministic = deterministic_s == "1"
deterministic_fail_closed = deterministic_fail_closed_s == "1"
save_safetensors = safetensors_s == "1"
write_checksums = write_checksums_s == "1"
write_weight_checksums = write_weight_checksums_s == "1"

# Cache env
os.environ["HF_HOME"] = cache_base
os.environ["HF_HUB_CACHE"] = os.path.join(cache_base, "hub")
if offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
local_files_only = offline

# Seeds
random.seed(seed)
if np is not None:
    np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if deterministic:
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        if deterministic_fail_closed:
            raise
        print(f"WARN: deterministic algorithms request failed: {e}", file=sys.stderr, flush=True)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

CRITICAL_KW = {"revision", "local_files_only"}  # never drop these

def call_with_kw_fallback(fn, *args, critical=CRITICAL_KW, **kwargs):
    dropped = []
    while True:
        try:
            out = fn(*args, **kwargs)
            if dropped:
                print(
                    f"WARN: dropped kwargs for {getattr(fn, '__qualname__', fn)}: {', '.join(dropped)}",
                    file=sys.stderr,
                    flush=True,
                )
            return out
        except TypeError as e:
            msg = str(e)
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if not m:
                raise
            bad = m.group(1)
            if bad in critical:
                raise SystemExit(
                    f"Dependency stack rejects critical kwarg '{bad}' for {getattr(fn, '__qualname__', fn)}. "
                    f"Upgrade torch/transformers/awq."
                )
            if bad not in kwargs:
                raise
            kwargs.pop(bad, None)
            dropped.append(bad)

cuda_ok = torch.cuda.is_available()
if require_cuda and not cuda_ok:
    raise SystemExit("CUDA is not available but REQUIRE_CUDA=1.")

# dtype from policy
if dtype_policy == "bf16":
    if not cuda_ok:
        raise SystemExit("dtype_policy=bf16 but CUDA is unavailable.")
    dtype = torch.bfloat16
elif dtype_policy == "fp16":
    if not cuda_ok:
        raise SystemExit("dtype_policy=fp16 but CUDA is unavailable.")
    dtype = torch.float16
elif dtype_policy == "cpu":
    dtype = None
else:
    raise SystemExit(f"Unknown dtype_policy: {dtype_policy}")

# device_map for load + smoke reload
if device_map_s == "auto":
    model_device_map = "auto"
elif device_map_s == "cuda0":
    if not cuda_ok:
        raise SystemExit("DEVICE_MAP=cuda0 requested but CUDA is unavailable.")
    model_device_map = {"": 0}
elif device_map_s == "cpu":
    model_device_map = {"": "cpu"}
else:
    raise SystemExit(f"Unknown DEVICE_MAP='{device_map_s}'. Use auto|cuda0|cpu.")

# Load by resolved commit hash when possible
load_rev = source_commit if re.fullmatch(r"[0-9a-f]{40}", source_commit) else user_revision

# When loading offline, resolve to the local snapshot directory so that
# transformers treats it as a local path (_is_local=True).  This avoids
# transformers 4.57+ calling model_info() — an API request — inside
# _patch_mistral_regex even when local_files_only=True.
model_path = model_id
if local_files_only and load_rev:
    _snap = os.path.join(cache_base, "hub", f"models--{model_id.replace('/', '--')}", "snapshots", load_rev)
    if os.path.isdir(_snap):
        model_path = _snap

print(f"Loading tokenizer: {model_path} (offline={offline}) ...", file=sys.stderr, flush=True)
tok_kwargs = dict(
    local_files_only=local_files_only,
    trust_remote_code=trust_remote_code,
)
tokenizer = call_with_kw_fallback(AutoTokenizer.from_pretrained, model_path, **tok_kwargs)

print(f"Loading model: {model_path} ...", file=sys.stderr, flush=True)
model_kwargs = dict(
    local_files_only=local_files_only,
    trust_remote_code=trust_remote_code,
    low_cpu_mem_usage=True,
    device_map=model_device_map,
)
if dtype is not None:
    model_kwargs["torch_dtype"] = dtype

model = call_with_kw_fallback(AutoAWQForCausalLM.from_pretrained, model_path, **model_kwargs)

quant_config = {
    "zero_point": True,
    "q_group_size": group_size,
    "w_bit": bits,
    "version": "GEMM",
}

quantize_kwargs = dict(quant_config=quant_config)
calib = {
    "calib_data": None,
    "max_calib_samples": None,
    "max_calib_seq_len": None,
    "n_parallel_calib_samples": None,
}
if calib_dataset:
    quantize_kwargs["calib_data"] = calib_dataset
    calib["calib_data"] = calib_dataset
if max_calib_samples_s:
    quantize_kwargs["max_calib_samples"] = int(max_calib_samples_s)
    calib["max_calib_samples"] = int(max_calib_samples_s)
if max_calib_seq_len_s:
    quantize_kwargs["max_calib_seq_len"] = int(max_calib_seq_len_s)
    calib["max_calib_seq_len"] = int(max_calib_seq_len_s)
if n_parallel_calib_samples_s:
    quantize_kwargs["n_parallel_calib_samples"] = int(n_parallel_calib_samples_s)
    calib["n_parallel_calib_samples"] = int(n_parallel_calib_samples_s)

def pick_device_for_inputs(m):
    if cuda_ok and device_map_s != "cpu":
        try:
            dev = next(m.parameters()).device
            # Accelerate dispatch can leave params on meta or CPU.
            if dev.type == "meta":
                return torch.device("cuda:0")
            if dev.type != "cuda":
                return torch.device("cuda:0")
            return dev
        except Exception:
            return torch.device("cuda:0")
    return torch.device("cpu")

def forward_and_tiny_generate(m, tok, dev: torch.device):
    text = "Hello"
    inp = tok(text, return_tensors="pt")
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.inference_mode():
        out = m(**inp)
    logits = getattr(out, "logits", None)
    if logits is None or getattr(logits, "ndim", None) != 3:
        raise RuntimeError(f"bad logits: {None if logits is None else tuple(logits.shape)}")
    with torch.inference_mode():
        gen = m.generate(**inp, max_new_tokens=2, do_sample=False)
    if getattr(gen, "ndim", None) != 2:
        raise RuntimeError(f"bad generate shape: {None if gen is None else tuple(gen.shape)}")
    if gen.shape[1] < inp["input_ids"].shape[1]:
        raise RuntimeError("generate output shorter than prompt")
    return {
        "logits_shape": list(logits.shape),
        "generate_shape": list(gen.shape),
        "device": str(dev),
    }

print(f"Quantizing: w_bit={bits}, q_group_size={group_size} ...", file=sys.stderr, flush=True)
t0 = time.time()
with torch.inference_mode():
    model.quantize(tokenizer, **quantize_kwargs)
t1 = time.time()

smoke = {"mode": smoke_mode, "ran": False, "ok": None, "details": None}

if smoke_mode == "fast":
    print("Smoke test (fast): forward + tiny generate on in-memory model ...", file=sys.stderr, flush=True)
    try:
        dev = pick_device_for_inputs(model)
        details = forward_and_tiny_generate(model, tokenizer, dev)
        smoke.update({"ran": True, "ok": True, "details": details})
    except Exception as e:
        smoke.update({"ran": True, "ok": False, "details": {"error": str(e)}})

print(f"Saving quantized model to: {out_dir}", file=sys.stderr, flush=True)
os.makedirs(out_dir, exist_ok=True)

# Fail-closed save: if these args are unsupported, error out.
model.save_quantized(out_dir, safetensors=save_safetensors, shard_size=shard_size)
tokenizer.save_pretrained(out_dir)

# Fingerprint must match expected snapshot id
with open(fingerprint_path, "r", encoding="utf-8") as f:
    finger_cfg = json.load(f)
fp = json.dumps(finger_cfg, sort_keys=True, separators=(",", ":"))
snap = hashlib.sha256(fp.encode("utf-8")).hexdigest()
if snap != expected_snapshot_id:
    raise SystemExit(f"Fingerprint mismatch: expected {expected_snapshot_id}, computed {snap}")

def list_files(root: str):
    out = []
    for base, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(base, fn)
            rel = os.path.relpath(p, root)
            out.append((rel, p))
    out.sort()
    return out

files = list_files(out_dir)
weight_like = [rel for rel, _ in files if rel.endswith(".safetensors") or rel.endswith(".bin")]

# Required outputs
req = ["config.json"]
missing = [x for x in req if not os.path.isfile(os.path.join(out_dir, x))]
if missing:
    raise SystemExit(f"Missing required output files after save: {missing}")
if not weight_like:
    raise SystemExit("Missing weight files (*.safetensors or *.bin) after save.")

# Tokenizer required outputs (same idea as bash completeness check)
if not os.path.isfile(os.path.join(out_dir, "tokenizer_config.json")):
    raise SystemExit("Missing tokenizer_config.json after save.")
if not (os.path.isfile(os.path.join(out_dir, "tokenizer.json")) or os.path.isfile(os.path.join(out_dir, "tokenizer.model"))):
    raise SystemExit("Missing tokenizer.json/tokenizer.model after save.")

with open(os.path.join(out_dir, "FINGERPRINT.json"), "w", encoding="utf-8") as f:
    json.dump(finger_cfg, f, indent=2, sort_keys=True)

# Full reload smoke test (reload policy matches device_map_s)
if smoke_mode == "full":
    print("Smoke test (full): reload saved artifact and run forward + tiny generate ...", file=sys.stderr, flush=True)
    if not hasattr(AutoAWQForCausalLM, "from_quantized"):
        raise SystemExit("Smoke test failed: AutoAWQForCausalLM.from_quantized is unavailable in this AWQ version.")
    try:
        if device_map_s == "auto":
            dm = "auto"
        elif device_map_s == "cuda0":
            dm = {"": 0}
        else:
            dm = {"": "cpu"}

        m2 = AutoAWQForCausalLM.from_quantized(
            out_dir,
            device_map=dm,
            trust_remote_code=trust_remote_code,
        )
        t2 = AutoTokenizer.from_pretrained(out_dir, local_files_only=True, trust_remote_code=trust_remote_code)
        dev = pick_device_for_inputs(m2)
        details = forward_and_tiny_generate(m2, t2, dev)
        smoke.update({"ran": True, "ok": True, "details": details})
    except Exception as e:
        smoke.update({"ran": True, "ok": False, "details": {"error": str(e)}})

elif smoke_mode == "off":
    smoke.update({"ran": False, "ok": None, "details": None})

# Checksums (opt-in; weights opt-in separately)
if write_checksums:
    sha256_manifest = {}
    for rel, path in files:
        is_weight = rel.endswith(".safetensors") or rel.endswith(".bin")
        if is_weight and not write_weight_checksums:
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        sha256_manifest[rel] = h.hexdigest()
    with open(os.path.join(out_dir, "FILES_SHA256.json"), "w", encoding="utf-8") as f:
        json.dump(sha256_manifest, f, indent=2, sort_keys=True)

meta = {
    "source_model": model_id,
    "user_revision": user_revision,
    "load_revision": load_rev,
    "source_commit": source_commit,
    "output_snapshot_id": expected_snapshot_id,
    "quant_config": quant_config,
    "offline": offline,
    "trust_remote_code": trust_remote_code,
    "torch_cuda": cuda_ok,
    "dtype_policy": dtype_policy,
    "device_map": device_map_s,
    "seconds_quantize": round(t1 - t0, 3),
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "calibration": calib,
    "seed": seed,
    "deterministic": deterministic,
    "save": {"safetensors": save_safetensors, "shard_size": shard_size},
    "weights_found": weight_like[:50],
    "smoke_test": smoke,
}

with open(os.path.join(out_dir, "awq_quantize_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)

print("Done.", file=sys.stderr, flush=True)
print(f"  Output: {out_dir}", file=sys.stderr, flush=True)

if smoke_mode in ("fast", "full") and smoke.get("ok") is False:
    raise SystemExit("Smoke test failed.")
PY

# Publish result (guarded rm -rf)
rm_rf_guarded "$OUTPUT_PATH"
mv -- "$TMP_DIR" "$OUTPUT_PATH"
TMP_DIR=""

if [[ "$WRITE_LOCAL_REPO_LAYOUT" == "1" ]]; then
  mkdir -p -- "$OUTPUT_SNAPSHOTS" "$OUTPUT_REFS"
  mkdir -p -- "$(dirname -- "$REF_PATH")"
  printf '%s' "$OUTPUT_SNAPSHOT_ID" >"$REF_PATH"
fi

# Optional size stats
if [[ "$SHOW_SIZES" == "1" ]]; then
  log ""
  log "=== Size stats (approx) ==="
  "$PYTHON" - "$SOURCE_CACHE" "$OUTPUT_PATH" <<'PY'
import os, sys
def dir_size_bytes(root: str) -> int:
    total = 0
    for base, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(base, fn)
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
    return total
src, out = sys.argv[1:3]
src_bytes = dir_size_bytes(src) if os.path.isdir(src) else 0
out_bytes = dir_size_bytes(out) if os.path.isdir(out) else 0
if src_bytes:
    print(f"  Source cache: {src_bytes} bytes")
print(f"  Output:       {out_bytes} bytes")
if src_bytes and out_bytes:
    print(f"  Compression:  {src_bytes / out_bytes:.2f}x")
PY
fi

if [[ "$JSON_MODE" == "1" ]]; then
  "$PYTHON" - "$MODEL_ID" "$REVISION" "$SOURCE_COMMIT" "$OUTPUT_SNAPSHOT_ID" \
    "$OUTPUT_PATH" "$REF_PATH" "$LAYOUT_DESC" "$DTYPE_POLICY" "${RESOLVED_DEVICE_MAP,,}" \
  <<'PY'
import json, os, sys
(
  model_id, revision, source_commit, snap,
  out_path, ref_path, layout, dtype_policy, device_map
) = sys.argv[1:10]

meta_path = os.path.join(out_path, "awq_quantize_meta.json")
meta = None
if os.path.isfile(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

checksums_path = os.path.join(out_path, "FILES_SHA256.json")
print(json.dumps({
  "status": "ok",
  "model_id": model_id,
  "revision": revision,
  "source_commit": source_commit,
  "output_snapshot_id": snap,
  "layout": layout,
  "output_path": out_path,
  "ref_path": ref_path if ref_path else None,
  "dtype_policy": dtype_policy,
  "device_map": device_map,
  "seconds_quantize": (meta.get("seconds_quantize") if meta else None),
  "smoke_test": (meta.get("smoke_test") if meta else None),
  "checksums": (checksums_path if os.path.isfile(checksums_path) else None),
}))
PY
else
  log ""
  log "Success."
  log "  Quantized model path: $OUTPUT_PATH"
  if [[ -f "$OUTPUT_PATH/FILES_SHA256.json" ]]; then
    log "  Checksums:            $OUTPUT_PATH/FILES_SHA256.json"
  fi
fi
