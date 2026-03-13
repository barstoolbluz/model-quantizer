#!/usr/bin/env bash
# quantize-fp8.sh
#
# Convert a Hugging Face causal LM to TorchAO FP8 weight-only (E4M3FN), then save
# a static checkpoint under an HF-cache-like layout.
#
# Fixes vs prior version:
# - Validate CLI-provided numeric flags (export before preflight)
# - Include --max-shard-size in snapshot id derivation (it changes output bytes/layout)
# - Canonical-path delete guard for snapshots/
# - Cross-host lock heartbeat to reduce multi-writer races
# - Validate/normalize cache and output roots (non-empty, absolute, writable)
# - Early disk free-space check (best-effort)
# - Smoke test moves inputs to embedding device (works with device_map='auto')
#
# Usage:
#   quantize-fp8 [--json] <model-id> [options]
#
# Options:
#   -c, --cache-dir DIR           HF cache root (default: $MODEL_CACHE_DIR or ./models)
#   -o, --output-dir DIR          Output root (default: $QUANTIZED_OUTPUT_DIR or cache dir)
#   -r, --revision REV            Revision to load (branch/tag/commit). Default: main
#       --device MODE             auto|cpu|cuda (default: auto)
#       --online                  Allow network access (default: offline)
#       --trust-remote-code       Allow model repo custom code (default: off)
#       --force                   Rebuild even if snapshot exists (publish swaps old->bak only after new is built)
#       --suffix STR              Output model id suffix (default: -FP8-TORCHAO)
#       --format FMT              torch|safetensors (default: torch)
#       --allow-safetensors       Attempt safetensors; implies --format safetensors (experimental)
#       --offline-pick-latest     Offline mode: pick newest cached snapshot when refs are missing (default: off)
#       --no-validate             Skip structural validation
#       --no-validate-quant       Skip quantization coverage validation
#       --quant-min-ratio FLOAT   Minimum fraction of Linear layers expected to be quantized (default: 0.80)
#       --validate-zip-crc        For .bin shards stored as zip, run zip CRC checks (slow; default: off)
#       --max-shard-size STR      Pass through to save_pretrained(max_shard_size=...) (default: unset)
#       --lock-ttl-seconds N      Lock TTL for cross-host stale handling (default: 21600)
#       --smoke-test              Load output and run a tiny generation (default: off)
#       --smoke-prompt STR        Prompt for smoke test (default: "Hello")
#       --smoke-max-new-tokens N  Tokens to generate in smoke test (default: 1)
#       --smoke-temperature F     Temperature for smoke test (default: 0.0)
#       --json                    JSON output mode (progress to stderr, single JSON object on stdout)
#
# Version gates (override via env vars):
#   MIN_TORCH_VERSION          (default: 2.1.0)
#   MIN_TRANSFORMERS_VERSION   (default: 4.40.0)
#   MIN_TORCHAO_VERSION        (default: 0.10.0)

set -euo pipefail
IFS=$'\n\t'

usage() { sed -n '3,46s/^# \?//p' "$0"; }

log() {
  # shellcheck disable=SC2059
  if [[ "$JSON_MODE" == "1" ]]; then
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
  else
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
  fi
}

die() { log "ERROR: $*"; exit 1; }

hf_repo_dir() {
  # HF cache uses models--org--name for org/name
  # shellcheck disable=SC2001
  printf '%s' "$1" | sed 's|/|--|g'
}

MODEL_ID=""
CACHE_DIR="${MODEL_CACHE_DIR:-./models}"
OUTPUT_DIR="${QUANTIZED_OUTPUT_DIR:-$CACHE_DIR}"
REVISION="main"
DEVICE_MODE="auto"
ONLINE=0
TRUST_REMOTE_CODE=0
FORCE=0
OUT_SUFFIX="-FP8-TORCHAO"
OUT_FORMAT="torch"
ALLOW_SAFETENSORS=0
OFFLINE_PICK_LATEST=0

DO_VALIDATE=1
DO_VALIDATE_QUANT=1
QUANT_MIN_RATIO="0.80"
DO_VALIDATE_ZIP_CRC=0
MAX_SHARD_SIZE=""

DO_SMOKE_TEST=0
SMOKE_PROMPT="Hello"
SMOKE_MAX_NEW_TOKENS="1"
SMOKE_TEMPERATURE="0.0"

LOCK_TTL_SECONDS="21600"  # 6h
JSON_MODE=0

MIN_TORCH_VERSION="${MIN_TORCH_VERSION:-2.1.0}"
MIN_TRANSFORMERS_VERSION="${MIN_TRANSFORMERS_VERSION:-4.40.0}"
MIN_TORCHAO_VERSION="${MIN_TORCHAO_VERSION:-0.10.0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--cache-dir) [[ $# -ge 2 ]] || die "--cache-dir requires a value"; [[ -n "${2//[[:space:]]/}" ]] || die "--cache-dir must not be empty"; CACHE_DIR="$2"; shift 2;;
    -o|--output-dir) [[ $# -ge 2 ]] || die "--output-dir requires a value"; [[ -n "${2//[[:space:]]/}" ]] || die "--output-dir must not be empty"; OUTPUT_DIR="$2"; shift 2;;
    -r|--revision) [[ $# -ge 2 ]] || die "--revision requires a value"; REVISION="$2"; shift 2;;
    --device) [[ $# -ge 2 ]] || die "--device requires a value"; DEVICE_MODE="$2"; shift 2;;
    --online) ONLINE=1; shift;;
    --trust-remote-code) TRUST_REMOTE_CODE=1; shift;;
    --force) FORCE=1; shift;;
    --suffix) [[ $# -ge 2 ]] || die "--suffix requires a value"; OUT_SUFFIX="$2"; shift 2;;
    --format) [[ $# -ge 2 ]] || die "--format requires a value"; OUT_FORMAT="$2"; shift 2;;
    --allow-safetensors) ALLOW_SAFETENSORS=1; OUT_FORMAT="safetensors"; shift;;
    --offline-pick-latest) OFFLINE_PICK_LATEST=1; shift;;
    --no-validate) DO_VALIDATE=0; shift;;
    --no-validate-quant) DO_VALIDATE_QUANT=0; shift;;
    --quant-min-ratio) [[ $# -ge 2 ]] || die "--quant-min-ratio requires a value"; QUANT_MIN_RATIO="$2"; shift 2;;
    --validate-zip-crc) DO_VALIDATE_ZIP_CRC=1; shift;;
    --max-shard-size) [[ $# -ge 2 ]] || die "--max-shard-size requires a value"; MAX_SHARD_SIZE="$2"; shift 2;;
    --lock-ttl-seconds) [[ $# -ge 2 ]] || die "--lock-ttl-seconds requires a value"; LOCK_TTL_SECONDS="$2"; shift 2;;
    --smoke-test) DO_SMOKE_TEST=1; shift;;
    --smoke-prompt) [[ $# -ge 2 ]] || die "--smoke-prompt requires a value"; SMOKE_PROMPT="$2"; shift 2;;
    --smoke-max-new-tokens) [[ $# -ge 2 ]] || die "--smoke-max-new-tokens requires a value"; SMOKE_MAX_NEW_TOKENS="$2"; shift 2;;
    --smoke-temperature) [[ $# -ge 2 ]] || die "--smoke-temperature requires a value"; SMOKE_TEMPERATURE="$2"; shift 2;;
    --json) JSON_MODE=1; shift;;
    -h|--help) usage; exit 0;;
    --) shift; break;;
    -*) die "Unknown option: $1";;
    *) if [[ -z "$MODEL_ID" ]]; then MODEL_ID="$1"; shift; else die "Unexpected argument: $1"; fi;;
  esac
done

[[ -n "$MODEL_ID" ]] || { usage; die "Missing <model-id>"; }

case "$DEVICE_MODE" in auto|cpu|cuda) ;; *) die "--device must be one of: auto|cpu|cuda";; esac
case "$OUT_FORMAT" in torch|safetensors) ;; *) die "--format must be one of: torch|safetensors";; esac

# VRAM check: if targeting GPU, verify enough free memory before proceeding.
if [[ "$DEVICE_MODE" != "cpu" ]]; then
  python3 -c "
import sys, torch
if torch.cuda.is_available():
    try:
        free, total = torch.cuda.mem_get_info(0)
        free_mb, total_mb = free >> 20, total >> 20
        if free_mb < 512:
            print(f'GPU has only {free_mb} MB free of {total_mb} MB. '
                  f'Another process may be using the GPU. '
                  f'Free GPU memory or re-run with --device cpu.', file=sys.stderr)
            sys.exit(1)
    except SystemExit:
        raise
    except Exception:
        print('Failed to query GPU memory — VRAM may be fully consumed '
              'by another process. Free GPU memory or re-run with --device cpu.', file=sys.stderr)
        sys.exit(1)
" || exit 1
fi

command -v python3 >/dev/null 2>&1 || die "python3 not found"

canon_path() {
  python3 - "$1" <<'PY'
import os, sys
p = sys.argv[1]
if p is None:
    print("", end="")
    sys.exit(0)
p = os.path.expanduser(p)
p = os.path.realpath(os.path.abspath(p))
print(p)
PY
}

# Normalize and validate dirs (avoid accidental writes under /hub, etc.)
CACHE_DIR="$(canon_path "$CACHE_DIR")"
OUTPUT_DIR="$(canon_path "$OUTPUT_DIR")"
[[ -n "$CACHE_DIR" ]] || die "--cache-dir resolved to empty"
[[ -n "$OUTPUT_DIR" ]] || die "--output-dir resolved to empty"
[[ "$CACHE_DIR" == /* ]] || die "--cache-dir must resolve to an absolute path: $CACHE_DIR"
[[ "$OUTPUT_DIR" == /* ]] || die "--output-dir must resolve to an absolute path: $OUTPUT_DIR"

if [[ "$ONLINE" -eq 0 ]]; then
  [[ -d "$CACHE_DIR" ]] || die "Offline mode: cache dir not found: $CACHE_DIR"
else
  mkdir -p "$CACHE_DIR" || die "Could not create cache dir: $CACHE_DIR"
fi

mkdir -p "$OUTPUT_DIR" || die "Could not create output dir: $OUTPUT_DIR"
_write_probe="$OUTPUT_DIR/.write-probe-$$"
( : > "$_write_probe" ) 2>/dev/null || die "Output dir not writable: $OUTPUT_DIR"
rm -f -- "$_write_probe" 2>/dev/null || true


# Suppress noisy deprecation/future warnings from third-party libraries
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error

# Offline by default
if [[ "$ONLINE" -eq 0 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

# Layout selection
WRITE_LOCAL_REPO_LAYOUT="${WRITE_LOCAL_REPO_LAYOUT:-0}"

SOURCE_ROOT="$CACHE_DIR/hub/models--$(hf_repo_dir "$MODEL_ID")"
OUTPUT_MODEL_ID="${MODEL_ID}${OUT_SUFFIX}"
if [[ "$WRITE_LOCAL_REPO_LAYOUT" == "1" ]]; then
  OUT_ROOT="$OUTPUT_DIR/hub/models--$(hf_repo_dir "$OUTPUT_MODEL_ID")"
else
  OUT_ROOT="$OUTPUT_DIR/$(hf_repo_dir "$OUTPUT_MODEL_ID")"
fi

# Export config so Python preflight validates actual CLI-provided values.
export QUANT_MIN_RATIO SMOKE_TEMPERATURE SMOKE_MAX_NEW_TOKENS LOCK_TTL_SECONDS MAX_SHARD_SIZE

# Preflight checks (fast fail with clear errors)
export JSON_MODE ONLINE REVISION OUT_FORMAT ALLOW_SAFETENSORS MIN_TORCH_VERSION MIN_TRANSFORMERS_VERSION MIN_TORCHAO_VERSION DEVICE_MODE DO_VALIDATE DO_VALIDATE_QUANT DO_SMOKE_TEST
python3 - <<'PY'
import os, sys, re, inspect, logging
logging.getLogger("torchao").setLevel(logging.ERROR)

json_mode = os.environ.get("JSON_MODE", "0") == "1"
out = sys.stderr if json_mode else sys.stdout

def fail(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def ver_ge(found: str, req: str) -> bool:
    try:
        from packaging.version import Version
        return Version(found) >= Version(req)
    except Exception:
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)", found or "")
        n = re.match(r"^(\d+)\.(\d+)\.(\d+)", req or "")
        if not m or not n:
            return False
        return tuple(map(int, m.groups())) >= tuple(map(int, n.groups()))

# Validate numeric flags
try:
    qmr = float(os.environ.get("QUANT_MIN_RATIO", "0.80"))
    if not (0.0 <= qmr <= 1.0):
        fail("QUANT_MIN_RATIO must be in [0.0, 1.0]")
except Exception:
    fail("--quant-min-ratio must parse as a float in [0.0, 1.0] (example: 0.85)")

try:
    float(os.environ.get("SMOKE_TEMPERATURE", "0.0"))
except Exception:
    fail("--smoke-temperature must parse as a float (example: 0.0)")

try:
    mnt = int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "1"))
    if mnt < 0:
        fail("--smoke-max-new-tokens must be a non-negative int")
except Exception:
    fail("--smoke-max-new-tokens must parse as an int (example: 1)")

try:
    ttl = int(os.environ.get("LOCK_TTL_SECONDS", "21600"))
    if ttl < 0:
        fail("--lock-ttl-seconds must be a non-negative int")
except Exception:
    fail("--lock-ttl-seconds must parse as an int (example: 21600)")

mss = os.environ.get("MAX_SHARD_SIZE", "")
if mss and not mss.strip():
    fail("--max-shard-size must not be whitespace-only")

try:
    import torch
except Exception as e:
    fail(f"failed to import torch: {e}")

try:
    import transformers
except Exception as e:
    fail(f"failed to import transformers: {e}")

try:
    import torchao
except Exception as e:
    fail(f"failed to import torchao: {e}")

torch_ver = getattr(torch, "__version__", "0.0.0")
tf_ver = getattr(transformers, "__version__", "0.0.0")
ta_ver = getattr(torchao, "__version__", "0.0.0")

min_torch = os.environ.get("MIN_TORCH_VERSION", "2.1.0")
min_tf = os.environ.get("MIN_TRANSFORMERS_VERSION", "4.40.0")
min_ta = os.environ.get("MIN_TORCHAO_VERSION", "0.10.0")

if not ver_ge(torch_ver, min_torch):
    fail(f"torch>={min_torch} required (found {torch_ver}).")
if not ver_ge(tf_ver, min_tf):
    fail(f"transformers>={min_tf} required (found {tf_ver}).")
if not ver_ge(ta_ver, min_ta):
    fail(f"torchao>={min_ta} required (found {ta_ver}).")

device_mode = os.environ.get("DEVICE_MODE", "auto")
if device_mode in ("auto", "cuda"):
    try:
        import accelerate  # noqa: F401
    except Exception as e:
        fail(f"device mode '{device_mode}' requires accelerate (pip install accelerate): {e}")

if not hasattr(torch, "float8_e4m3fn"):
    fail("torch.float8_e4m3fn is unavailable in this PyTorch build. Install a float8-capable build of PyTorch.")

do_validate = os.environ.get("DO_VALIDATE", "1") == "1"
if do_validate:
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" not in sig.parameters:
            fail("this torch build lacks torch.load(..., weights_only=...). Upgrade torch or run with --no-validate.")
    except (TypeError, ValueError):
        fail("could not inspect torch.load signature; upgrade torch or run with --no-validate.")

out_format = os.environ.get("OUT_FORMAT", "torch")
allow_safetensors = os.environ.get("ALLOW_SAFETENSORS", "0") == "1"
if out_format == "safetensors" and not allow_safetensors:
    fail("--format safetensors requires --allow-safetensors.")

online = os.environ.get("ONLINE", "0") == "1"
rev = os.environ.get("REVISION", "main")
is_commit = bool(re.fullmatch(r"[0-9a-f]{40}", rev))
if online and not is_commit:
    try:
        import huggingface_hub  # noqa: F401
    except Exception as e:
        fail(f"online revision resolution requires huggingface_hub (pip install huggingface_hub): {e}")

print("Preflight OK:", file=out)
print(f"  torch:        {torch_ver}", file=out)
print(f"  transformers: {tf_ver}", file=out)
print(f"  torchao:      {ta_ver}", file=out)
if device_mode in ("auto", "cuda"):
    import accelerate
    print(f"  accelerate:   {getattr(accelerate, '__version__', 'unknown')}", file=out)
PY

# Resolve source snapshot id (strict by default; optional offline fallback)
SOURCE_SNAPSHOT_ID=""
if [[ -f "$SOURCE_ROOT/refs/$REVISION" ]]; then
  SOURCE_SNAPSHOT_ID="$(tr -d '\r\n' < "$SOURCE_ROOT/refs/$REVISION")"
elif [[ -f "$SOURCE_ROOT/refs/main" ]]; then
  SOURCE_SNAPSHOT_ID="$(tr -d '\r\n' < "$SOURCE_ROOT/refs/main")"
elif [[ "$ONLINE" -eq 0 ]]; then
  if [[ -d "$SOURCE_ROOT/snapshots" ]]; then
    mapfile -t _snaps < <(ls -1 "$SOURCE_ROOT/snapshots" 2>/dev/null || true)
    if [[ "${#_snaps[@]}" -eq 1 ]]; then
      SOURCE_SNAPSHOT_ID="${_snaps[0]}"
    elif [[ "${#_snaps[@]}" -gt 1 ]] && [[ "$OFFLINE_PICK_LATEST" -eq 1 ]]; then
      SOURCE_SNAPSHOT_ID="$(
        python3 - <<'PY'
import os, sys
root = sys.argv[1]
cands = []
for name in os.listdir(root):
    p = os.path.join(root, name)
    if os.path.isdir(p):
        try:
            cands.append((os.path.getmtime(p), name))
        except OSError:
            pass
if not cands:
    sys.exit(1)
cands.sort()
print(cands[-1][1])
PY
        "$SOURCE_ROOT/snapshots"
      )" || die "Offline mode: found snapshots but could not select one."
      log "Offline fallback picked newest cached snapshot: $SOURCE_SNAPSHOT_ID"
    else
      die "Offline mode: cannot resolve snapshot id (missing refs/$REVISION and refs/main). Re-run with --offline-pick-latest, or cache a commit and pass --revision <commit>."
    fi
  else
    die "Offline mode: source model cache not found at: $SOURCE_ROOT"
  fi
fi

# Resolve a commit hash for determinism
RESOLVED_COMMIT=""
if [[ -n "$SOURCE_SNAPSHOT_ID" ]] && [[ "$SOURCE_SNAPSHOT_ID" =~ ^[0-9a-f]{40}$ ]]; then
  RESOLVED_COMMIT="$SOURCE_SNAPSHOT_ID"
elif [[ "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  RESOLVED_COMMIT="$REVISION"
elif [[ "$ONLINE" -eq 1 ]]; then
  export MODEL_ID REVISION
  RESOLVED_COMMIT="$(
    python3 - <<'PY'
import os, sys
from huggingface_hub import HfApi
model_id = os.environ["MODEL_ID"]
rev = os.environ.get("REVISION","main")
api = HfApi()
info = api.model_info(model_id, revision=rev)
sha = getattr(info, "sha", None)
if not sha:
    print("ERROR: hub returned no commit sha", file=sys.stderr)
    sys.exit(4)
print(sha)
PY
  )" || die "Could not resolve a commit for --revision '$REVISION'. Use a commit hash, or pre-cache the model and run offline."
else
  die "Offline mode needs a cached commit hash. Cache the model commit and pass --revision <commit>."
fi


# Lock per OUT_ROOT
mkdir -p "$OUT_ROOT"
LOCK_DIR="$OUT_ROOT/.quantize.lock"
LOCK_PID_FILE="$LOCK_DIR/pid"
LOCK_META_FILE="$LOCK_DIR/meta"
LOCK_HEARTBEAT_FILE="$LOCK_DIR/heartbeat_epoch"
THIS_HOST="$(hostname 2>/dev/null || echo unknown)"

now_epoch() { date +%s; }

pid_state() {
  python3 - "$1" <<'PY'
import os, sys, errno
pid = int(sys.argv[1])
try:
    os.kill(pid, 0)
except OSError as e:
    if e.errno == errno.ESRCH:
        sys.exit(1)
    if e.errno == errno.EPERM:
        sys.exit(2)
    sys.exit(3)
else:
    sys.exit(0)
PY
}

lock_meta_get() {
  local key="$1"
  [[ -f "$LOCK_META_FILE" ]] || return 1
  awk -F= -v k="$key" '$1==k{print $2; exit}' "$LOCK_META_FILE" 2>/dev/null || return 1
}

read_heartbeat() {
  [[ -f "$LOCK_HEARTBEAT_FILE" ]] || return 1
  tr -d '\r\n' < "$LOCK_HEARTBEAT_FILE" 2>/dev/null || return 1
}

write_lock_meta() {
  {
    printf 'started_epoch=%s\n' "$(now_epoch)"
    printf 'host=%s\n' "$THIS_HOST"
    printf 'user=%s\n' "${USER:-unknown}"
    printf 'cwd=%s\n' "$(pwd)"
    printf 'cmd=%q ' "$0" "$@"
    printf '\n'
  } > "$LOCK_META_FILE" 2>/dev/null || true
}

write_heartbeat() {
  printf '%s\n' "$(now_epoch)" > "$LOCK_HEARTBEAT_FILE" 2>/dev/null || true
}

HEARTBEAT_PID=""

start_heartbeat() {
  write_heartbeat
  (
    while true; do
      write_heartbeat
      sleep 30
    done
  ) &
  HEARTBEAT_PID="$!"
}

stop_heartbeat() {
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    wait "$HEARTBEAT_PID" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
}

recent_tmp_activity() {
  local snaps
  if [[ "$WRITE_LOCAL_REPO_LAYOUT" == "1" ]]; then
    snaps="$OUT_ROOT/snapshots"
  else
    snaps="$OUT_ROOT"
  fi
  [[ -d "$snaps" ]] || return 1
  # If a tmp dir was touched in the last 15 min, treat as active work.
  find "$snaps" -maxdepth 1 -type d -name '.tmp-*' -mmin -15 -print -quit 2>/dev/null | grep -q .
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_PID_FILE"
    write_lock_meta "$@"
    write_heartbeat
    return 0
  fi

  local ttl pid host started hb state now started_age hb_age
  ttl="$LOCK_TTL_SECONDS"
  now="$(now_epoch)"

  pid=""
  [[ -f "$LOCK_PID_FILE" ]] && pid="$(tr -d '\r\n' < "$LOCK_PID_FILE" || true)"
  host="$(lock_meta_get host || true)"
  started="$(lock_meta_get started_epoch || true)"
  hb="$(read_heartbeat || true)"

  started_age=""
  hb_age=""
  if [[ -n "$started" && "$started" =~ ^[0-9]+$ ]]; then
    started_age=$(( now - started ))
  fi
  if [[ -n "$hb" && "$hb" =~ ^[0-9]+$ ]]; then
    hb_age=$(( now - hb ))
  fi

  # Same-host: PID check first; fall back to heartbeat
  if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ && "$host" == "$THIS_HOST" ]]; then
    state=0
    pid_state "$pid" || state=$?
    case "$state" in
      0) die "Another job is running for: $OUT_ROOT (host=$host pid=$pid)";;
      2) die "Another job is running for: $OUT_ROOT (host=$host pid=$pid; permission denied)";;
      1) ;; # stale pid
      *) die "Lock exists but pid state could not be checked (host=$host pid=$pid state=$state). Remove $LOCK_DIR if you know it is stale.";;
    esac
    if [[ -n "$hb_age" && "$hb_age" -ge 0 && "$hb_age" -lt 120 ]]; then
      die "Lock looks active via heartbeat (host=$host heartbeat_age=${hb_age}s)."
    fi
  fi

  # Cross-host: only take over if (a) started is old, (b) heartbeat is old/missing, and (c) no recent tmp activity.
  if [[ -n "$host" && "$host" != "$THIS_HOST" ]]; then
    [[ -n "$started_age" ]] || die "Lock exists from another host (host=$host) with no start time. Remove $LOCK_DIR if you know it is stale."
    if (( started_age < 0 )); then
      die "Lock time looks invalid (host=$host started_epoch=$started). Remove $LOCK_DIR if you know it is stale."
    fi
    if (( started_age < ttl )); then
      die "Lock exists from another host (host=$host age=${started_age}s < ttl=${ttl}s)."
    fi
    if [[ -n "$hb_age" && "$hb_age" -ge 0 && "$hb_age" -lt ttl ]]; then
      die "Lock from another host still has a recent heartbeat (host=$host heartbeat_age=${hb_age}s < ttl=${ttl}s)."
    fi
    if recent_tmp_activity; then
      die "Lock from another host looks active (recent tmp activity under snapshots)."
    fi
    log "Lock from another host exceeded ttl and looks inactive; taking over. (host=$host age=${started_age}s ttl=${ttl}s heartbeat_age=${hb_age:-missing})"
  fi

  rm -rf "$LOCK_DIR" 2>/dev/null || true
  mkdir "$LOCK_DIR" 2>/dev/null || die "Failed to acquire lock for: $OUT_ROOT"
  printf '%s\n' "$$" > "$LOCK_PID_FILE"
  write_lock_meta "$@"
  write_heartbeat
}

release_lock() { rm -rf "$LOCK_DIR" 2>/dev/null || true; }

acquire_lock "$@"
start_heartbeat

# If we fail before the main cleanup trap is installed, still drop the lock.
early_cleanup() { stop_heartbeat || true; release_lock; }
trap early_cleanup EXIT INT TERM

if [[ "$WRITE_LOCAL_REPO_LAYOUT" == "1" ]]; then
  SNAPSHOTS_DIR="$OUT_ROOT/snapshots"
  REFS_DIR="$OUT_ROOT/refs"
else
  SNAPSHOTS_DIR="$OUT_ROOT"
  REFS_DIR=""
fi
mkdir -p "$SNAPSHOTS_DIR"


disk_space_check() {
  local src="$SOURCE_ROOT/snapshots/$SOURCE_SNAPSHOT_ID"
  local dst="$SNAPSHOTS_DIR"
  [[ -d "$src" ]] || return 0

  local src_kb avail_kb need_kb
  src_kb="$(du -sk "$src" 2>/dev/null | awk '{print $1}')" || return 0
  [[ -n "$src_kb" ]] || return 0

  avail_kb="$(df -Pk "$dst" 2>/dev/null | awk 'NR==2{print $4}')" || return 0
  [[ -n "$avail_kb" ]] || return 0

  # Need ~= source_size + 20% + 1MB.
  need_kb=$(( src_kb + src_kb/5 + 1024 ))

  if (( avail_kb < need_kb )); then
    log "Disk check failed:"
    log "  source snapshot:   $src"
    log "  source size (KB):  $src_kb"
    log "  free space (KB):   $avail_kb"
    log "  needed (KB):       $need_kb"
    return 1
  fi
  return 0
}

if ! disk_space_check; then
  die "Not enough free space under: $SNAPSHOTS_DIR"
fi


safe_rm_rf() {
  local target="$1"
  [[ -n "$target" ]] || { log "ERROR: refusing to delete empty path"; return 2; }

  python3 - "$SNAPSHOTS_DIR" "$target" <<'PY'
import os, sys
snap = os.path.realpath(sys.argv[1])
tgt = os.path.realpath(sys.argv[2])
if tgt == snap or tgt.startswith(snap + os.sep):
    sys.exit(0)
print(f"Refusing to delete outside snapshots: {tgt}", file=sys.stderr)
sys.exit(3)
PY
  local ok=$?
  if [[ "$ok" -ne 0 ]]; then
    return "$ok"
  fi

  rm -rf -- "$target"
}


export REVISION
REVISION_REF_SAFE="$(
python3 - <<'PY'
import os, re
rev = os.environ.get("REVISION","main")
rev = rev.replace("/", "--")
rev = re.sub(r"[^A-Za-z0-9._-]+", "_", rev)
print(rev)
PY
)"

# Deterministic OUT_SNAPSHOT_ID (includes MAX_SHARD_SIZE)
export MODEL_ID REVISION RESOLVED_COMMIT DEVICE_MODE TRUST_REMOTE_CODE OUT_SUFFIX OUT_FORMAT ALLOW_SAFETENSORS MIN_TORCH_VERSION MIN_TRANSFORMERS_VERSION MIN_TORCHAO_VERSION MAX_SHARD_SIZE
OUT_SNAPSHOT_ID="$(
python3 - <<'PY'
import hashlib, json, logging, os, platform
logging.getLogger("torchao").setLevel(logging.ERROR)
import torch, transformers, torchao

mss = os.environ.get("MAX_SHARD_SIZE","").strip() or None

payload = {
    "script_version": 12,
    "model_id": os.environ["MODEL_ID"],
    "revision_requested": os.environ.get("REVISION", "main"),
    "resolved_commit": os.environ.get("RESOLVED_COMMIT", ""),
    "trust_remote_code": os.environ.get("TRUST_REMOTE_CODE", "0") == "1",
    "device_mode": os.environ.get("DEVICE_MODE", "auto"),
    "out_suffix": os.environ.get("OUT_SUFFIX", "-FP8-TORCHAO"),
    "out_format": os.environ.get("OUT_FORMAT", "torch"),
    "allow_safetensors": os.environ.get("ALLOW_SAFETENSORS", "0") == "1",
    "max_shard_size": mss,
    "quant": {
        "scheme": "torchao.Float8WeightOnlyConfig",
        "weight_dtype": "torch.float8_e4m3fn",
        "set_inductor_config": True,
        "version": 2,
    },
    "versions": {
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": getattr(transformers, "__version__", "unknown"),
        "torchao": getattr(torchao, "__version__", "unknown"),
    },
}
blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
print(hashlib.sha1(blob).hexdigest())
PY
)"

FINAL_SNAPSHOT_PATH="$SNAPSHOTS_DIR/$OUT_SNAPSHOT_ID"
TMP_SNAPSHOT_PATH="$SNAPSHOTS_DIR/.tmp-$OUT_SNAPSHOT_ID-$$"
BAK_SNAPSHOT_PATH="$SNAPSHOTS_DIR/.bak-$OUT_SNAPSHOT_ID-$$"

log "TorchAO FP8 weight-only (E4M3FN)"
log "  Source model:       $MODEL_ID"
log "  Source cache:       $CACHE_DIR"
log "  Revision requested: $REVISION"
log "  Resolved commit:    $RESOLVED_COMMIT"
log "  Output model:       $OUTPUT_MODEL_ID"
log "  Output root:        $OUT_ROOT"
log "  Snapshot id:        $OUT_SNAPSHOT_ID"
log "  Device mode:        $DEVICE_MODE"
log "  Network:            $([[ "$ONLINE" -eq 1 ]] && echo enabled || echo disabled)"
log "  Remote code:        $([[ "$TRUST_REMOTE_CODE" -eq 1 ]] && echo allowed || echo blocked)"
log "  Format:             $OUT_FORMAT"
if [[ "$OUT_FORMAT" == "safetensors" ]]; then
  log "  Note:               safetensors may fail with tensor-subclass checkpoints depending on versions"
fi
if [[ "$DO_VALIDATE" -eq 0 ]]; then
  log "  Note:               structural validation disabled"
fi
if [[ "$DO_VALIDATE_QUANT" -eq 0 ]]; then
  log "  Note:               quant coverage validation disabled"
else
  log "  Quant min ratio:    $QUANT_MIN_RATIO"
fi
if [[ "$DO_VALIDATE_ZIP_CRC" -eq 1 ]]; then
  log "  Zip CRC validation: enabled (may be slow)"
fi
if [[ -n "$MAX_SHARD_SIZE" ]]; then
  log "  Max shard size:     $MAX_SHARD_SIZE"
fi
if [[ "$DO_SMOKE_TEST" -eq 1 ]]; then
  log "  Smoke test:         enabled (max_new_tokens=$SMOKE_MAX_NEW_TOKENS, temperature=$SMOKE_TEMPERATURE)"
fi
log ""

PUBLISHED=0
cleanup() {
  local rc=$?
  if [[ -d "$TMP_SNAPSHOT_PATH" ]]; then
    if ! safe_rm_rf "$TMP_SNAPSHOT_PATH" 2>/dev/null; then
      log "WARNING: refused to delete tmp snapshot path: $TMP_SNAPSHOT_PATH"
    fi
  fi
  if [[ "$PUBLISHED" -eq 1 ]]; then
    if [[ -d "$BAK_SNAPSHOT_PATH" ]]; then
      if ! safe_rm_rf "$BAK_SNAPSHOT_PATH" 2>/dev/null; then
        log "WARNING: refused to delete backup snapshot path: $BAK_SNAPSHOT_PATH"
      fi
    fi
  fi
  stop_heartbeat || true
  release_lock
  exit $rc
}
trap cleanup EXIT INT TERM

if [[ -d "$FINAL_SNAPSHOT_PATH" ]] && [[ -f "$FINAL_SNAPSHOT_PATH/config.json" ]] && [[ "$FORCE" -eq 0 ]]; then
  log "Output already exists: $FINAL_SNAPSHOT_PATH"
  log "Pass --force to rebuild."
  if [[ "$JSON_MODE" == "1" ]]; then
    python3 - "$MODEL_ID" "$REVISION" "$RESOLVED_COMMIT" "$OUT_SNAPSHOT_ID" "$FINAL_SNAPSHOT_PATH" "$DEVICE_MODE" "$OUT_FORMAT" <<'PY'
import json, sys
model_id, revision, commit, snap, out_path, device_mode, fmt = sys.argv[1:8]
print(json.dumps({
  "status": "exists",
  "model_id": model_id,
  "revision": revision,
  "resolved_commit": commit,
  "output_snapshot_id": snap,
  "output_path": out_path,
  "device_mode": device_mode,
  "format": fmt,
}))
PY
  fi
  exit 0
fi

if [[ -d "$TMP_SNAPSHOT_PATH" ]]; then
  safe_rm_rf "$TMP_SNAPSHOT_PATH"
fi
mkdir -p "$TMP_SNAPSHOT_PATH"

export OUTPUT_SNAPSHOT_TMP="$TMP_SNAPSHOT_PATH"
export CACHE_DIR MODEL_ID REVISION SOURCE_SNAPSHOT_ID RESOLVED_COMMIT ONLINE TRUST_REMOTE_CODE DEVICE_MODE OUT_FORMAT ALLOW_SAFETENSORS DO_VALIDATE_QUANT QUANT_MIN_RATIO MAX_SHARD_SIZE JSON_MODE
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
QUANT_START_EPOCH=$(date +%s)
python3 - <<'PY'
import json
import logging
import os
import platform
import time
import sys

logging.getLogger("torchao").setLevel(logging.ERROR)

json_mode = os.environ.get("JSON_MODE", "0") == "1"
out = sys.stderr if json_mode else sys.stdout

model_id = os.environ["MODEL_ID"]
cache_dir = os.path.join(os.environ["CACHE_DIR"], "hub")
out_dir = os.environ["OUTPUT_SNAPSHOT_TMP"]
revision = os.environ.get("REVISION", "main")
source_snapshot_id = os.environ.get("SOURCE_SNAPSHOT_ID", "")
resolved_commit = os.environ.get("RESOLVED_COMMIT", "")
local_only = os.environ.get("ONLINE", "0") != "1"
trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "0") == "1"
device_mode = os.environ.get("DEVICE_MODE", "auto")
out_format = os.environ.get("OUT_FORMAT", "torch")
allow_safetensors = os.environ.get("ALLOW_SAFETENSORS", "0") == "1"
do_validate_quant = os.environ.get("DO_VALIDATE_QUANT", "1") == "1"
quant_min_ratio = float(os.environ.get("QUANT_MIN_RATIO", "0.80"))
max_shard_size = os.environ.get("MAX_SHARD_SIZE", "").strip() or None

os.environ["HF_HOME"] = os.environ["CACHE_DIR"]
os.environ["HF_HUB_CACHE"] = cache_dir

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchao
from torchao.quantization import Float8WeightOnlyConfig, quantize_

if not hasattr(torch, "float8_e4m3fn"):
    raise RuntimeError("torch.float8_e4m3fn not available in this torch build")

load_revision = resolved_commit or (source_snapshot_id if local_only and source_snapshot_id else revision)

# When loading offline, resolve to the local snapshot directory so that
# transformers treats it as a local path (_is_local=True).  This avoids
# transformers 4.57+ calling model_info() — an API request — inside
# _patch_mistral_regex even when local_files_only=True.
model_path = model_id
if local_only and load_revision:
    _snap = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}", "snapshots", load_revision)
    if os.path.isdir(_snap):
        model_path = _snap

print("Versions:", file=out)
print(f"  python:       {platform.python_version()}", file=out)
print(f"  torch:        {torch.__version__}", file=out)
print(f"  transformers: {transformers.__version__}", file=out)
print(f"  torchao:      {getattr(torchao, '__version__', 'unknown')}", file=out)
print(f"  load_revision:{load_revision}", file=out)
if torch.cuda.is_available():
    try:
        print(f"  cuda:         {torch.version.cuda}", file=out)
        print(f"  gpu[0]:       {torch.cuda.get_device_name(0)}", file=out)
    except Exception:
        pass
print("", file=out)

print(f"Loading tokenizer: {model_path} (local_only={local_only})", file=out)
tok = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=local_only,
    trust_remote_code=trust_remote_code,
)

quant_cfg = Float8WeightOnlyConfig(
    weight_dtype=torch.float8_e4m3fn,
    set_inductor_config=True,
    version=2,
)

if device_mode == "auto":
    device_map = "auto"
elif device_mode == "cuda":
    device_map = {"": "cuda:0"}
elif device_mode == "cpu":
    device_map = None
else:
    raise ValueError(f"Unexpected device_mode={device_mode}")

print(f"Loading model (bfloat16): {model_path} (device_map={device_map})", file=out)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=local_only,
    trust_remote_code=trust_remote_code,
    dtype=torch.bfloat16,
    device_map=device_map,
    low_cpu_mem_usage=True,
)
model.eval()

if device_mode == "cpu":
    model.to("cpu")

commit_in_config = getattr(getattr(model, "config", None), "_commit_hash", None)

def iter_linear_named_modules(m):
    import torch.nn as nn
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            yield name, mod

def weight_is_quantized(w):
    if w is None:
        return False
    try:
        if getattr(w, "dtype", None) == torch.float8_e4m3fn:
            return True
    except Exception:
        pass
    try:
        mod = type(w).__module__ or ""
        if "torchao" in mod:
            return True
    except Exception:
        pass
    return False

print("Quantizing weights via torchao.quantize_ (Float8WeightOnlyConfig)...", file=out)
t0 = time.time()
with torch.no_grad():
    quantize_(model, quant_cfg)
print(f"Quantization finished in {time.time() - t0:.1f}s", file=out)

quant_stats = {"linear_total": 0, "linear_quantized": 0, "ratio": None, "sample_not_quantized": []}

if do_validate_quant:
    not_q = []
    total = 0
    q = 0
    for name, mod in iter_linear_named_modules(model):
        total += 1
        w = getattr(mod, "weight", None)
        if weight_is_quantized(w):
            q += 1
        else:
            if len(not_q) < 25:
                not_q.append(name)
    quant_stats["linear_total"] = total
    quant_stats["linear_quantized"] = q
    quant_stats["ratio"] = (q / total) if total else None
    quant_stats["sample_not_quantized"] = not_q

    if total == 0:
        print("WARNING: found zero nn.Linear modules; skipping quant coverage check.", file=sys.stderr)
    else:
        ratio = q / total
        print(f"Quant coverage: {q}/{total} Linear layers look quantized (ratio={ratio:.3f})", file=out)
        if ratio < quant_min_ratio:
            msg = (
                f"Quant coverage ratio {ratio:.3f} < min {quant_min_ratio:.3f}. "
                "This can happen if the model uses non-Linear ops for projections."
            )
            if not_q:
                msg += f" Sample not quantized: {not_q[:10]}"
            raise RuntimeError(msg)

safe_serialization = (out_format == "safetensors")
if safe_serialization and (not allow_safetensors):
    raise RuntimeError("--format safetensors requires --allow-safetensors.")

print(f"Saving checkpoint to: {out_dir}", file=out)
os.makedirs(out_dir, exist_ok=True)

# Unwrap TorchAO tensor subclasses (e.g. Float8Tensor) into raw tensors
# so that safetensors can serialize them.  Float8Tensor stores quantized
# data in .qdata (float8_e4m3fn) and per-row scales in .scale (float32).
raw_state_dict = model.state_dict()
unwrapped = {}
_has_fp8 = False
for _k, _v in raw_state_dict.items():
    if hasattr(_v, 'qdata') and hasattr(_v, 'scale'):
        _has_fp8 = True
        unwrapped[_k] = _v.qdata.contiguous()
        unwrapped[_k + "_scale"] = _v.scale.contiguous()
    elif isinstance(_v, torch.Tensor):
        unwrapped[_k] = _v.contiguous()
    else:
        unwrapped[_k] = _v
if _has_fp8:
    print(f"Unwrapped {sum(1 for k in unwrapped if k.endswith('_scale'))} FP8 tensor subclasses", file=out)

save_kwargs = {}
if max_shard_size:
    save_kwargs["max_shard_size"] = max_shard_size

model.save_pretrained(out_dir, state_dict=unwrapped, **save_kwargs)
tok.save_pretrained(out_dir)

cfg_path = os.path.join(out_dir, "config.json")
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
except FileNotFoundError:
    cfg = {}

cfg["quant_method"] = "torchao"
cfg["quant_type"] = {
    "default": {
        "_type": "Float8WeightOnlyConfig",
        "_data": {
            "weight_dtype": "torch.float8_e4m3fn",
            "set_inductor_config": True,
            "version": 2,
        },
    }
}

with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)

meta = {
    "source_model_id": model_id,
    "revision_requested": revision,
    "source_snapshot_id_cache": source_snapshot_id,
    "resolved_commit": resolved_commit,
    "commit_from_transformers_config": commit_in_config,
    "load_revision_used": load_revision,
    "quantization": {
        "scheme": "torchao.Float8WeightOnlyConfig",
        "weight_dtype": "torch.float8_e4m3fn",
        "set_inductor_config": True,
        "version": 2,
    },
    "quant_coverage": quant_stats,
    "runtime": {
        "device_mode": device_mode,
        "local_only": local_only,
        "trust_remote_code": trust_remote_code,
    },
    "output": {
        "format": out_format,
        "safe_serialization": safe_serialization,
        "max_shard_size": max_shard_size,
    },
    "versions": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torchao": getattr(torchao, "__version__", "unknown"),
    },
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    },
    "cuda": {
        "available": bool(torch.cuda.is_available()),
        "version": getattr(torch.version, "cuda", None),
        "gpu0": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    },
}

with open(os.path.join(out_dir, "quantize_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)

print("Save complete.", file=out)
PY
QUANT_ELAPSED=$(( $(date +%s) - QUANT_START_EPOCH ))

if [[ "$DO_VALIDATE" -eq 1 ]]; then
  export CHECKPOINT_DIR="$TMP_SNAPSHOT_PATH"
  export OUT_FORMAT
  export DO_VALIDATE_ZIP_CRC
  python3 - <<'PY'
import json, os, sys, glob, zipfile

json_mode = os.environ.get("JSON_MODE", "0") == "1"
out = sys.stderr if json_mode else sys.stdout

ckpt_dir = os.environ["CHECKPOINT_DIR"]
out_format = os.environ.get("OUT_FORMAT", "torch")
do_zip_crc = os.environ.get("DO_VALIDATE_ZIP_CRC", "0") == "1"

def fail(msg: str):
    print(f"ERROR: validation failed: {msg}", file=sys.stderr)
    sys.exit(2)

cfg_path = os.path.join(ckpt_dir, "config.json")
if not os.path.exists(cfg_path):
    fail(f"missing config.json in {ckpt_dir}")

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

qm = cfg.get("quant_method")
qt = cfg.get("quant_type")
if not (isinstance(qm, str) and "torchao" in qm):
    fail("config.json missing quant_method containing 'torchao'")
if not (isinstance(qt, dict) and "default" in qt and isinstance(qt["default"], dict)):
    fail("config.json quant_type missing 'default' block")

tok_candidates = [
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
if not any(os.path.exists(os.path.join(ckpt_dir, p)) for p in tok_candidates):
    fail("no tokenizer artifacts found")

def file_nonempty(p: str) -> bool:
    try:
        return os.path.getsize(p) > 0
    except OSError:
        return False

if out_format == "safetensors":
    st = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not st:
        fail("no .safetensors files found")
    try:
        from safetensors import safe_open
    except Exception as e:
        fail(f"could not import safetensors to validate headers: {e}")
    for p in st:
        if not file_nonempty(p):
            fail(f"empty safetensors file: {os.path.basename(p)}")
        try:
            with safe_open(p, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if not keys:
                    fail(f"safetensors file has no tensors: {os.path.basename(p)}")
                _ = f.get_tensor(keys[0]).dtype
        except Exception as e:
            fail(f"failed to open safetensors {os.path.basename(p)}: {e}")
    print("Validation OK (safetensors headers).", file=out)
    sys.exit(0)

index_json = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
single_bin = os.path.join(ckpt_dir, "pytorch_model.bin")

# transformers 5.x always saves as safetensors even when out_format="torch",
# so check for safetensors files as a fallback before failing.
st_fallback = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))

if os.path.exists(index_json):
    with open(index_json, "r", encoding="utf-8") as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    if not isinstance(wm, dict) or not wm:
        fail("index file has empty or invalid weight_map")
    shards = [os.path.join(ckpt_dir, p) for p in sorted(set(wm.values()))]
elif os.path.exists(single_bin):
    shards = [single_bin]
elif st_fallback:
    # Saved as safetensors despite out_format="torch" (transformers 5.x behavior)
    for p in st_fallback:
        if not file_nonempty(p):
            fail(f"empty safetensors file: {os.path.basename(p)}")
    print("Validation OK (safetensors fallback for torch format).", file=out)
    sys.exit(0)
else:
    fail("could not find pytorch_model.bin, pytorch_model.bin.index.json, or any .safetensors files")

def looks_like_zip(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        return sig.startswith(b"PK\x03\x04")
    except Exception:
        return False

def looks_like_pickle(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            b = f.read(2)
        return len(b) >= 1 and b[0] == 0x80
    except Exception:
        return False

for p in shards:
    if not os.path.exists(p):
        fail(f"missing shard: {os.path.basename(p)}")
    if not file_nonempty(p):
        fail(f"empty shard: {os.path.basename(p)}")

    if looks_like_zip(p):
        try:
            with zipfile.ZipFile(p, "r") as zf:
                names = zf.namelist()
                if not names:
                    fail(f"zip shard has no entries: {os.path.basename(p)}")
                if do_zip_crc:
                    bad = zf.testzip()
                    if bad is not None:
                        fail(f"zip CRC check failed in {os.path.basename(p)} at entry {bad}")
        except Exception as e:
            fail(f"could not read zip shard {os.path.basename(p)}: {e}")
    else:
        if not looks_like_pickle(p):
            print(f"WARNING: shard {os.path.basename(p)} is not zip and does not look like a standard pickle header; skipping deep checks.", file=sys.stderr)

print("Validation OK (files + index + basic structure).", file=out)
PY
fi

if [[ "$DO_SMOKE_TEST" -eq 1 ]]; then
  export SMOKE_DIR="$TMP_SNAPSHOT_PATH"
  export DEVICE_MODE
  export SMOKE_PROMPT SMOKE_MAX_NEW_TOKENS SMOKE_TEMPERATURE
  python3 - <<'PY'
import os, sys, time
import torch

json_mode = os.environ.get("JSON_MODE", "0") == "1"
out = sys.stderr if json_mode else sys.stdout
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt_dir = os.environ["SMOKE_DIR"]
device_mode = os.environ.get("DEVICE_MODE", "auto")
prompt = os.environ.get("SMOKE_PROMPT", "Hello")
max_new_tokens = int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "1"))
temperature = float(os.environ.get("SMOKE_TEMPERATURE", "0.0"))

if device_mode in ("auto", "cuda") and not torch.cuda.is_available():
    raise RuntimeError("Smoke test needs CUDA for device_mode=auto|cuda, but torch.cuda.is_available() is False.")

if device_mode == "auto":
    device_map = "auto"
elif device_mode == "cuda":
    device_map = {"": "cuda:0"}
else:
    device_map = None

print(f"Smoke test: loading from {ckpt_dir} (device_mode={device_mode})", file=out)
tok = AutoTokenizer.from_pretrained(ckpt_dir)
model = AutoModelForCausalLM.from_pretrained(
    ckpt_dir,
    dtype=torch.bfloat16,
    device_map=device_map,
    low_cpu_mem_usage=True,
)
model.eval()

try:
    emb = model.get_input_embeddings()
    target_device = emb.weight.device
except Exception:
    target_device = next(model.parameters()).device

inputs = tok(prompt, return_tensors="pt")
inputs = {k: v.to(target_device) for k, v in inputs.items()}

gen_kwargs = dict(max_new_tokens=max_new_tokens)
if temperature <= 0.0:
    gen_kwargs["do_sample"] = False
else:
    gen_kwargs["do_sample"] = True
    gen_kwargs["temperature"] = temperature

t0 = time.time()
with torch.no_grad():
    gen_out = model.generate(**inputs, **gen_kwargs)
dt = time.time() - t0
text = tok.decode(gen_out[0], skip_special_tokens=True)
print(f"Smoke test OK in {dt:.2f}s. Output:", file=out)
print(text, file=out)
PY
fi

publish_snapshot() {
  local tmp="$1" final="$2" bak="$3"

  if [[ -d "$final" ]]; then
    if [[ -d "$bak" ]]; then
      safe_rm_rf "$bak"
    fi
    if ! mv "$final" "$bak"; then
      die "Publish failed: could not move existing snapshot to backup: $final -> $bak"
    fi
  fi

  if mv "$tmp" "$final"; then
    return 0
  fi

  if [[ -d "$bak" ]] && [[ ! -d "$final" ]]; then
    mv "$bak" "$final" 2>/dev/null || true
  fi
  die "Publish failed: could not move tmp snapshot into place: $tmp -> $final (rolled back when possible)"
}

publish_snapshot "$TMP_SNAPSHOT_PATH" "$FINAL_SNAPSHOT_PATH" "$BAK_SNAPSHOT_PATH"
PUBLISHED=1

write_ref() {
  local path="$1" value="$2" tmp
  tmp="${path}.tmp.$$"
  printf '%s' "$value" > "$tmp"
  mv "$tmp" "$path"
}
if [[ "$WRITE_LOCAL_REPO_LAYOUT" == "1" ]]; then
  mkdir -p "$REFS_DIR"
  write_ref "$REFS_DIR/main" "$OUT_SNAPSHOT_ID"
  write_ref "$REFS_DIR/$REVISION_REF_SAFE" "$OUT_SNAPSHOT_ID" 2>/dev/null || true
fi

if [[ -d "$BAK_SNAPSHOT_PATH" ]]; then
  safe_rm_rf "$BAK_SNAPSHOT_PATH"
fi

if [[ "$JSON_MODE" == "1" ]]; then
  python3 - "$MODEL_ID" "$REVISION" "$RESOLVED_COMMIT" "$OUT_SNAPSHOT_ID" \
    "$FINAL_SNAPSHOT_PATH" "$DEVICE_MODE" "$OUT_FORMAT" "$QUANT_ELAPSED" \
  <<'PY'
import json, os, sys
model_id, revision, commit, snap, out_path, device_mode, fmt, elapsed_s = sys.argv[1:9]

meta_path = os.path.join(out_path, "quantize_meta.json")
meta = None
if os.path.isfile(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

print(json.dumps({
  "status": "ok",
  "model_id": model_id,
  "revision": revision,
  "resolved_commit": commit,
  "output_snapshot_id": snap,
  "output_path": out_path,
  "device_mode": device_mode,
  "format": fmt,
  "seconds_quantize": int(elapsed_s),
  "smoke_test": None,
  "validation": (meta.get("quant_coverage") if meta else None),
}))
PY
else
  if command -v du >/dev/null 2>&1; then
    log ""
    log "Sizes:"
    if [[ -d "$SOURCE_ROOT" ]]; then
      log "  Source cache:  $(du -sh "$SOURCE_ROOT" | awk '{print $1}')"
    fi
    log "  Output:        $(du -sh "$FINAL_SNAPSHOT_PATH" | awk '{print $1}')"
  fi

  log ""
  log "Done."
  log "  Snapshot path: $FINAL_SNAPSHOT_PATH"
  log "  Model root:    $OUT_ROOT"
  log ""
  log "vLLM examples:"
  log "  vllm serve '$OUT_ROOT' --quantization torchao --dtype bfloat16"
  log "  # or:"
  log "  vllm serve '$FINAL_SNAPSHOT_PATH' --quantization torchao --dtype bfloat16"
fi
