#!/usr/bin/env bash
# quantize-fp8-production.sh
#
# Convert a Hugging Face causal LM to TorchAO FP8 weight-only (E4M3FN), then save
# a static checkpoint under an HF-cache-like layout.
#
# This production variant is designed for machine use and hostile-ish operational
# environments: exact JSON failure behavior, validated idempotent reuse, CPU-only
# reload validation after the original model is released, and a stronger lock path
# based on descriptor locking with a guarded fallback.
#
# Usage:
#   quantize-fp8-production [--json|--json-strict] <model-id> [options]
#
# Options:
#   -c, --cache-dir DIR              HF cache root (default: $MODEL_CACHE_DIR or ./models)
#   -o, --output-dir DIR             Output root (default: $QUANTIZED_OUTPUT_DIR or cache dir)
#   -r, --revision REV               Revision to load (branch/tag/commit). Default: main
#       --device MODE                auto|cpu|cuda (default: auto)
#       --online                     Allow network access (default: offline)
#       --trust-remote-code          Allow model repo custom code (default: off)
#       --force                      Rebuild even if reusable output exists
#       --suffix STR                 Output model id suffix (default: -FP8-TORCHAO)
#       --format FMT                 torch|safetensors (default: torch)
#       --allow-safetensors          Attempt safetensors; implies --format safetensors
#       --offline-pick-latest        Offline mode: pick newest cached snapshot when refs are missing
#       --no-validate                Skip deep validation, checksum walks, and reload validation; reuse still requires a fingerprint match
#       --reload-validate-device DEV cpu|skip (default: cpu)
#       --no-validate-quant          Skip quant coverage validation
#       --quant-min-ratio FLOAT      Minimum fraction of Linear layers expected to be quantized (default: 0.80)
#       --validate-zip-crc           For .bin shards stored as zip, run zip CRC checks (slow; default: off)
#       --max-shard-size STR         Pass through to save_pretrained(max_shard_size=...)
#       --lock-mode MODE             auto|fd|mkdir (default: auto)
#       --lock-ttl-seconds N         Stale lock TTL for mkdir fallback (default: 21600)
#       --lock-timeout N             Lock wait seconds (0=fail-fast, -1=wait without limit, default: 0)
#       --smoke-test                 Run a tiny generation smoke test on the freshly quantized model
#       --smoke-prompt STR           Prompt for smoke test (default: "Hello")
#       --smoke-max-new-tokens N     Tokens to generate in smoke test (default: 1)
#       --smoke-temperature F        Temperature for smoke test (default: 0.0)
#       --json                       JSON success/exists output; logs go to stderr
#       --json-strict                Like --json, and all failures also emit JSON to stdout
#
# Notes:
#   * Idempotence is with respect to resolved inputs. A floating revision such as
#     main may resolve to a different commit on a later run.
#   * --no-validate skips checksum verification and reload validation. Reuse still
#     requires a matching fingerprint so the script does not alias different
#     build configurations onto the same output snapshot.
#
# Version gates (override via env vars):
#   MIN_TORCH_VERSION          (default: 2.1.0)
#   MIN_TRANSFORMERS_VERSION   (default: 4.40.0)
#   MIN_TORCHAO_VERSION        (default: 0.10.0)

set -Eeuo pipefail
IFS=$'
	'

JSON_MODE=0
JSON_STRICT=0
ERROR_EMITTED=0
CURRENT_STAGE="startup"
LAST_VALIDATE_REASON=""
LAST_LOCK_REASON=""
LOCK_HELPER_PID=""
LOCK_HELPER_STATUS=""
LOCK_FALLBACK_MODE=""
LOCK_HEARTBEAT_PID=""

usage() { sed -n '3,52s/^# \?//p' "$0"; }

emit_error_json() {
  local msg="$1"
  local exit_code="${2:-1}"
  local stage="${3:-${CURRENT_STAGE:-unknown}}"
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

log() {
  local ts
  ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if [[ "${JSON_MODE:-0}" == "1" ]]; then
    printf '[%s] %s
' "$ts" "$*" >&2
  else
    printf '[%s] %s
' "$ts" "$*"
  fi
}

die_with_code() {
  local code="$1"
  shift
  local msg="$*"
  if [[ "${JSON_STRICT:-0}" == "1" ]]; then
    emit_error_json "$msg" "$code"
  else
    log "ERROR: $msg"
  fi
  exit "$code"
}

die() {
  die_with_code 1 "$*"
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

if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' 'ERROR: python3 is required' >&2
  exit 127
fi

require_value() {
  local opt="$1"
  local remaining="$2"
  [[ "$remaining" -ge 2 ]] || die_with_code 2 "Missing value for $opt"
}

have() { command -v "$1" >/dev/null 2>&1; }

hf_repo_dir() {
  printf '%s' "$1" | sed 's|/|--|g'
}

fs_token() {
  python3 - "$1" <<'PY'
import base64, sys
raw = sys.argv[1].encode('utf-8')
token = base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')
print(token or '_')
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

canon_path() {
  python3 - "$1" <<'PY'
import os, sys
p = os.path.expanduser(sys.argv[1])
p = os.path.realpath(os.path.abspath(p))
print(p)
PY
}

safe_rm_rf() {
  local target="$1"
  [[ -n "$target" ]] || die "internal: empty path for deletion"
  [[ "$target" != "/" ]] || die "refusing to delete /"
  python3 - "$SNAPSHOTS_DIR" "$target" <<'PY'
import os, sys
root = os.path.realpath(sys.argv[1])
tgt = os.path.realpath(sys.argv[2])
if tgt == root or tgt.startswith(root + os.sep):
    sys.exit(0)
print(f"refusing to delete outside snapshots root: {tgt}", file=sys.stderr)
sys.exit(3)
PY
  rm -rf -- "$target"
}

write_ref() {
  local path="$1"
  local value="$2"
  local tmp
  tmp="${path}.tmp.$$"
  printf '%s
' "$value" > "$tmp"
  mv "$tmp" "$path"
}

emit_exists_json() {
  local out_path="$1"
  python3 - "$MODEL_ID" "$REVISION" "$RESOLVED_COMMIT" "$OUT_SNAPSHOT_ID" "$out_path" "$DEVICE_MODE" "$OUT_FORMAT" "$DO_VALIDATE" <<'PY'
import json, sys, time
model_id, revision, commit, snap, out_path, device_mode, fmt, do_validate = sys.argv[1:9]
print(json.dumps({
    "status": "exists",
    "model_id": model_id,
    "revision": revision,
    "resolved_commit": commit,
    "output_snapshot_id": snap,
    "output_path": out_path,
    "device_mode": device_mode,
    "format": fmt,
    "validated": do_validate == "1",
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}))
PY
}

emit_success_json() {
  local out_path="$1"
  local elapsed_s="$2"
  python3 - "$MODEL_ID" "$REVISION" "$RESOLVED_COMMIT" "$OUT_SNAPSHOT_ID" "$out_path" "$DEVICE_MODE" "$OUT_FORMAT" "$elapsed_s" <<'PY'
import json, os, sys, time
model_id, revision, commit, snap, out_path, device_mode, fmt, elapsed_s = sys.argv[1:9]
meta_path = os.path.join(out_path, "quantize_meta.json")
meta = None
if os.path.isfile(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
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
    "seconds_quantize": int(float(elapsed_s)),
    "validation": (meta.get("validation") if meta else None),
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}))
PY
}

build_fingerprint_json() {
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "$10" "$11" "$12" "$13" "$14" "$15" "$16" "$17" "$18" <<'PY'
import hashlib, json, os, platform, sys
import torch, transformers, torchao
(
    model_id,
    revision,
    resolved_commit,
    device_mode,
    trust_remote_code,
    out_suffix,
    out_format,
    allow_safetensors,
    max_shard_size,
    do_validate,
    reload_validate_device,
    do_validate_quant,
    quant_min_ratio,
    validate_zip_crc,
    smoke_test,
    smoke_prompt,
    smoke_max_new_tokens,
    smoke_temperature,
) = sys.argv[1:19]
payload = {
    "model_id": model_id,
    "revision_requested": revision,
    "resolved_commit": resolved_commit or None,
    "device_mode": device_mode,
    "trust_remote_code": trust_remote_code == "1",
    "out_suffix": out_suffix,
    "output": {
        "format": out_format,
        "allow_safetensors": allow_safetensors == "1",
        "max_shard_size": (max_shard_size or None),
    },
    "quant": {
        "scheme": "torchao.Float8WeightOnlyConfig",
        "weight_dtype": "torch.float8_e4m3fn",
        "set_inductor_config": False,
        "coverage_validation": {
            "enabled": do_validate_quant == "1",
            "min_ratio": quant_min_ratio,
        },
        "version": 4,
    },
    "validation": {
        "enabled": do_validate == "1",
        "reload_validate_device": reload_validate_device,
        "zip_crc": validate_zip_crc == "1",
        "smoke_test": {
            "enabled": smoke_test == "1",
            "prompt": smoke_prompt if smoke_test == "1" else None,
            "max_new_tokens": int(smoke_max_new_tokens) if smoke_test == "1" else None,
            "temperature": float(smoke_temperature) if smoke_test == "1" else None,
        },
    },
    "build": {
        "script_sha256": os.environ.get("SCRIPT_SHA256"),
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": getattr(transformers, "__version__", "unknown"),
        "torchao": getattr(torchao, "__version__", "unknown"),
    },
}
text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
print(text)
print(hashlib.sha256(text.encode("utf-8")).hexdigest())
PY
}

quick_reuse_check() {
  local out_path="$1"
  local expected_fp_sha="$2"

  [[ -d "$out_path" ]] || { LAST_VALIDATE_REASON="missing directory"; return 1; }
  [[ -f "$out_path/config.json" ]] || { LAST_VALIDATE_REASON="missing config.json"; return 1; }

  local model_index=0
  compgen -G "$out_path/*.bin" >/dev/null && model_index=1
  compgen -G "$out_path/*.safetensors" >/dev/null && model_index=1
  [[ -f "$out_path/model.safetensors.index.json" ]] && model_index=1
  [[ -f "$out_path/pytorch_model.bin.index.json" ]] && model_index=1
  [[ "$model_index" == "1" ]] || { LAST_VALIDATE_REASON="missing model shards or index"; return 1; }

  local fp_sha="$out_path/FINGERPRINT.sha256"
  [[ -f "$fp_sha" ]] || { LAST_VALIDATE_REASON="missing fingerprint sha"; return 1; }
  local got_sha
  got_sha="$(tr -d ' 

	' < "$fp_sha")"
  [[ -n "$got_sha" ]] || { LAST_VALIDATE_REASON="empty fingerprint sha"; return 1; }
  [[ "$got_sha" == "$expected_fp_sha" ]] || { LAST_VALIDATE_REASON="fingerprint mismatch"; return 1; }

  LAST_VALIDATE_REASON="ok"
  return 0
}

full_reuse_check() {
  local out_path="$1"
  local expected_fp_sha="$2"
  quick_reuse_check "$out_path" "$expected_fp_sha" || return 1

  local meta="$out_path/quantize_meta.json"
  local reload="$out_path/reload_validation.json"
  local checksums="$out_path/CHECKSUMS.sha256.json"
  [[ -f "$meta" ]] || { LAST_VALIDATE_REASON="missing quantize_meta.json"; return 1; }
  [[ -f "$reload" ]] || { LAST_VALIDATE_REASON="missing reload_validation.json"; return 1; }
  [[ -f "$checksums" ]] || { LAST_VALIDATE_REASON="missing checksum manifest"; return 1; }

  python3 - "$meta" "$reload" "$checksums" "$MODEL_ID" "$REVISION" "$RESOLVED_COMMIT" "$OUT_FORMAT" "$expected_fp_sha" "$out_path" "$DO_VALIDATE" <<'PY'
import hashlib, json, os, sys
meta_path, reload_path, checksums_path, model_id, revision, commit, fmt, fp_sha, out_path, do_validate = sys.argv[1:11]
with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
with open(reload_path, 'r', encoding='utf-8') as f:
    reload_meta = json.load(f)
with open(checksums_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)
errs = []
if meta.get("model_id") != model_id:
    errs.append("model_id mismatch")
if meta.get("revision") != revision:
    errs.append("revision mismatch")
if meta.get("resolved_commit") != commit:
    errs.append("resolved_commit mismatch")
if meta.get("format") != fmt:
    errs.append("format mismatch")
if meta.get("fingerprint_sha256") != fp_sha:
    errs.append("meta fingerprint mismatch")
if do_validate == "1":
    rv = reload_meta
    if rv.get("skipped") is not True and not rv.get("ok", False):
        errs.append("reload validation not marked ok")
entries = manifest.get("files")
if not isinstance(entries, list) or not entries:
    errs.append("empty checksum manifest")
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
if not errs:
    for entry in entries:
        rel = entry.get("relative_path")
        want = entry.get("sha256")
        size = entry.get("size_bytes")
        if not rel or not want:
            errs.append("malformed checksum manifest entry")
            break
        p = os.path.join(out_path, rel)
        if not os.path.isfile(p):
            errs.append(f"missing file in checksum manifest: {rel}")
            break
        if os.path.getsize(p) != size:
            errs.append(f"size mismatch for {rel}")
            break
        got = sha256_file(p)
        if got != want:
            errs.append(f"sha256 mismatch for {rel}")
            break
if errs:
    raise SystemExit("; ".join(errs))
PY

  LAST_VALIDATE_REASON="ok"
  return 0
}

validate_existing_artifact() {
  local out_path="$1"
  local expected_fp_sha="$2"
  if [[ "$DO_VALIDATE" == "1" || "$SMOKE_TEST" == "1" ]]; then
    full_reuse_check "$out_path" "$expected_fp_sha"
  else
    quick_reuse_check "$out_path" "$expected_fp_sha"
  fi
}

acquire_lock_mkdir() {
  local lockdir="$1"
  local ttl="$2"
  local timeout="$3"
  local started now age owner_host owner_pid hb_path own_hb tmp_hb
  started="$(date +%s)"
  own_hb="$lockdir/heartbeat"
  tmp_hb="$lockdir/heartbeat.$$"

  while true; do
    if mkdir "$lockdir" 2>/dev/null; then
      printf '%s
' "$$" > "$lockdir/pid"
      hostname > "$lockdir/host" 2>/dev/null || true
      date -u '+%Y-%m-%dT%H:%M:%SZ' > "$lockdir/acquired_at"
      date +%s > "$tmp_hb"
      mv "$tmp_hb" "$own_hb"
      LAST_LOCK_REASON="acquired"
      LOCK_FALLBACK_MODE="mkdir"
      return 0
    fi

    hb_path="$lockdir/heartbeat"
    now="$(date +%s)"
    if [[ -f "$hb_path" ]]; then
      age=$(( now - $(cat "$hb_path" 2>/dev/null || echo 0) ))
    else
      age=$(( ttl + 1 ))
    fi

    if [[ "$age" -gt "$ttl" ]]; then
      owner_host="$(cat "$lockdir/host" 2>/dev/null || echo unknown)"
      owner_pid="$(cat "$lockdir/pid" 2>/dev/null || echo unknown)"
      log "Stale mkdir lock detected (> ${ttl}s). Taking over: $lockdir (owner ${owner_host}:${owner_pid})"
      rm -rf -- "$lockdir"
      continue
    fi

    if [[ "$timeout" == "0" ]]; then
      LAST_LOCK_REASON="busy"
      return 1
    fi
    if [[ "$timeout" != "-1" && $(( now - started )) -ge "$timeout" ]]; then
      LAST_LOCK_REASON="timeout"
      return 1
    fi
    sleep 2
  done
}

acquire_lock_fd() {
  local lockfile="$1"
  local timeout="$2"
  local status_file status helper_status

  mkdir -p "$(dirname "$lockfile")"
  status_file="$(mktemp "$(dirname "$lockfile")/.fd-lock-status.XXXXXX")"

  python3 - "$lockfile" "$timeout" "$status_file" "$$" <<'PY' &
import ctypes
import ctypes.util
import errno
import fcntl
import os
import signal
import stat
import sys
import time

lockfile, timeout_s, status_path, parent_pid = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])

def write_status(value: str) -> None:
    with open(status_path, "w", encoding="utf-8") as f:
        f.write(value + "\n")

def set_parent_deathsig(expected_parent_pid: int) -> None:
    if not sys.platform.startswith("linux"):
        return
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return
    libc = ctypes.CDLL(libc_name, use_errno=True)
    PR_SET_PDEATHSIG = 1
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    if os.getppid() != expected_parent_pid:
        raise SystemExit(4)

dir_path = os.path.dirname(lockfile) or "."
base_name = os.path.basename(lockfile)
dir_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
if hasattr(os, "O_DIRECTORY"):
    dir_flags |= os.O_DIRECTORY
dir_fd = os.open(dir_path, dir_flags)
fd = None
try:
    file_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    fd = os.open(base_name, file_flags, 0o600, dir_fd=dir_fd)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        write_status("not-regular")
        raise SystemExit(5)

    set_parent_deathsig(parent_pid)

    started = time.monotonic()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if timeout_s == 0:
                write_status("busy")
                raise SystemExit(6)
            if timeout_s != -1 and (time.monotonic() - started) >= timeout_s:
                write_status("timeout")
                raise SystemExit(7)
            time.sleep(0.2)
        except OSError as e:
            if e.errno in (errno.EOPNOTSUPP, errno.ENOTSUP):
                write_status("unsupported")
                raise SystemExit(8)
            raise

    write_status("acquired")

    def _exit(*_args):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)

    while True:
        if os.getppid() != parent_pid:
            raise SystemExit(0)
        time.sleep(5)
finally:
    try:
        os.close(dir_fd)
    except Exception:
        pass
    if fd is not None:
        try:
            os.close(fd)
        except Exception:
            pass
PY
  LOCK_HELPER_PID=$!

  while true; do
    if [[ -s "$status_file" ]]; then
      status="$(tr -d ' \n\r\t' < "$status_file" 2>/dev/null || true)"
      rm -f -- "$status_file"
      case "$status" in
        acquired)
          LAST_LOCK_REASON="acquired"
          return 0
          ;;
        busy|timeout)
          LAST_LOCK_REASON="$status"
          wait "$LOCK_HELPER_PID" >/dev/null 2>&1 || true
          LOCK_HELPER_PID=""
          return 1
          ;;
        unsupported|not-regular)
          LAST_LOCK_REASON="$status"
          wait "$LOCK_HELPER_PID" >/dev/null 2>&1 || true
          LOCK_HELPER_PID=""
          return 1
          ;;
        *)
          LAST_LOCK_REASON="fd-helper-failed"
          wait "$LOCK_HELPER_PID" >/dev/null 2>&1 || true
          LOCK_HELPER_PID=""
          return 1
          ;;
      esac
    fi

    if ! kill -0 "$LOCK_HELPER_PID" >/dev/null 2>&1; then
      rm -f -- "$status_file"
      wait "$LOCK_HELPER_PID" >/dev/null 2>&1 || true
      LOCK_HELPER_PID=""
      LAST_LOCK_REASON="fd-helper-failed"
      return 1
    fi
    sleep 0.1
  done
}

release_lock_fd() {
  if [[ -n "${LOCK_HELPER_PID:-}" ]]; then
    kill "$LOCK_HELPER_PID" >/dev/null 2>&1 || true
    wait "$LOCK_HELPER_PID" >/dev/null 2>&1 || true
    LOCK_HELPER_PID=""
  fi
}

start_mkdir_heartbeat() {
  local lockdir="$1"
  [[ "${LOCK_FALLBACK_MODE:-}" == "mkdir" ]] || return 0
  (
    while true; do
      date +%s > "$lockdir/heartbeat.$$" 2>/dev/null || exit 0
      mv "$lockdir/heartbeat.$$" "$lockdir/heartbeat" 2>/dev/null || exit 0
      sleep 5 || exit 0
    done
  ) &
  LOCK_HEARTBEAT_PID=$!
}

stop_mkdir_heartbeat() {
  if [[ -n "${LOCK_HEARTBEAT_PID:-}" ]]; then
    kill "$LOCK_HEARTBEAT_PID" >/dev/null 2>&1 || true
    wait "$LOCK_HEARTBEAT_PID" >/dev/null 2>&1 || true
  fi
}

acquire_lock() {
  local lockdir="$1"
  local ttl="$2"
  local timeout="$3"
  local mode="$4"

  LOCK_FALLBACK_MODE=""

  case "$mode" in
    fd)
      if acquire_lock_fd "${lockdir}.fd" "$timeout"; then
        LOCK_FALLBACK_MODE="fd"
        return 0
      fi
      return 1
      ;;
    mkdir)
      if acquire_lock_mkdir "$lockdir" "$ttl" "$timeout"; then
        LOCK_FALLBACK_MODE="mkdir"
        return 0
      fi
      return 1
      ;;
    auto)
      if acquire_lock_fd "${lockdir}.fd" "$timeout"; then
        LOCK_FALLBACK_MODE="fd"
        return 0
      fi
      case "$LAST_LOCK_REASON" in
        unsupported|fd-helper-failed|not-regular)
          log "FD lock path unavailable (${LAST_LOCK_REASON}); falling back to mkdir lock"
          if acquire_lock_mkdir "$lockdir" "$ttl" "$timeout"; then
            LOCK_FALLBACK_MODE="mkdir"
            return 0
          fi
          return 1
          ;;
        *)
          return 1
          ;;
      esac
      ;;
    *)
      LAST_LOCK_REASON="invalid-lock-mode"
      return 1
      ;;
  esac
}

cleanup_lock() {
  release_lock_fd
  stop_mkdir_heartbeat
  if [[ "${LOCK_FALLBACK_MODE:-}" == "mkdir" ]]; then
    rm -rf -- "${LOCK_BASE_PATH}" >/dev/null 2>&1 || true
  fi
}

compare_versions_ge() {
  python3 - "$1" "$2" <<'PY'
from packaging.version import Version
import sys
print(1 if Version(sys.argv[1]) >= Version(sys.argv[2]) else 0)
PY
}

validate_revision_string() {
  local rev="$1"
  # Reject: path traversal, absolute paths, shell metacharacters
  if [[ "$rev" =~ \.\. ]] || [[ "$rev" =~ ^/ ]] || [[ "$rev" =~ /$ ]]; then
    return 1  # Path traversal or absolute path attempts
  fi
  # Reject shell metacharacters and command substitution
  if [[ "$rev" =~ [\$\`\(\)\{\}\[\]\*\?\\\'\"\<\>\|] ]]; then
    return 1  # Shell metacharacters
  fi
  # Allow only: alphanumeric, dash, underscore, forward slash, period
  if [[ ! "$rev" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
    return 1  # Contains other invalid characters
  fi
  # Additional safety: max length and no double slashes
  if [[ "${#rev}" -gt 256 ]] || [[ "$rev" =~ // ]]; then
    return 1
  fi
  return 0
}

CURRENT_STAGE="parse-args"
START_EPOCH="$(date +%s)"

CACHE_DIR="${MODEL_CACHE_DIR:-./models}"
OUTPUT_DIR="${QUANTIZED_OUTPUT_DIR:-}"
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
RELOAD_VALIDATE_DEVICE="cpu"
DO_VALIDATE_QUANT=1
QUANT_MIN_RATIO="0.80"
VALIDATE_ZIP_CRC=0
MAX_SHARD_SIZE=""
LOCK_MODE="auto"
LOCK_TTL_SECONDS="21600"
LOCK_TIMEOUT="0"
SMOKE_TEST=0
SMOKE_PROMPT="Hello"
SMOKE_MAX_NEW_TOKENS="1"
SMOKE_TEMPERATURE="0.0"
MODEL_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--cache-dir)
      require_value "$1" "$#"; CACHE_DIR="$2"; shift 2 ;;
    -o|--output-dir)
      require_value "$1" "$#"; OUTPUT_DIR="$2"; shift 2 ;;
    -r|--revision)
      require_value "$1" "$#"; REVISION="$2"; shift 2 ;;
    --device)
      require_value "$1" "$#"; DEVICE_MODE="$2"; shift 2 ;;
    --online)
      ONLINE=1; shift ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=1; shift ;;
    --force)
      FORCE=1; shift ;;
    --suffix)
      require_value "$1" "$#"; OUT_SUFFIX="$2"; shift 2 ;;
    --format)
      require_value "$1" "$#"; OUT_FORMAT="$2"; shift 2 ;;
    --allow-safetensors)
      ALLOW_SAFETENSORS=1; OUT_FORMAT="safetensors"; shift ;;
    --offline-pick-latest)
      OFFLINE_PICK_LATEST=1; shift ;;
    --no-validate)
      DO_VALIDATE=0; RELOAD_VALIDATE_DEVICE="skip"; shift ;;
    --reload-validate-device)
      require_value "$1" "$#"; RELOAD_VALIDATE_DEVICE="$2"; shift 2 ;;
    --no-validate-quant)
      DO_VALIDATE_QUANT=0; shift ;;
    --quant-min-ratio)
      require_value "$1" "$#"; QUANT_MIN_RATIO="$2"; shift 2 ;;
    --validate-zip-crc)
      VALIDATE_ZIP_CRC=1; shift ;;
    --max-shard-size)
      require_value "$1" "$#"; MAX_SHARD_SIZE="$2"; shift 2 ;;
    --lock-mode)
      require_value "$1" "$#"; LOCK_MODE="$2"; shift 2 ;;
    --lock-ttl-seconds)
      require_value "$1" "$#"; LOCK_TTL_SECONDS="$2"; shift 2 ;;
    --lock-timeout)
      require_value "$1" "$#"; LOCK_TIMEOUT="$2"; shift 2 ;;
    --smoke-test)
      SMOKE_TEST=1; shift ;;
    --smoke-prompt)
      require_value "$1" "$#"; SMOKE_PROMPT="$2"; shift 2 ;;
    --smoke-max-new-tokens)
      require_value "$1" "$#"; SMOKE_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --smoke-temperature)
      require_value "$1" "$#"; SMOKE_TEMPERATURE="$2"; shift 2 ;;
    --json)
      JSON_MODE=1; shift ;;
    --json-strict)
      JSON_MODE=1; JSON_STRICT=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; break ;;
    -*)
      die_with_code 2 "Unknown option: $1" ;;
    *)
      if [[ -z "$MODEL_ID" ]]; then
        MODEL_ID="$1"
      else
        die_with_code 2 "Unexpected positional argument: $1"
      fi
      shift ;;
  esac
done

[[ -n "$MODEL_ID" ]] || die_with_code 2 "Missing required model id"

# Validate revision parameter
if ! validate_revision_string "$REVISION"; then
  die_with_code 2 "Invalid revision format. Must contain only alphanumeric, dash, underscore, period, and forward slash (no path traversal)"
fi

case "$DEVICE_MODE" in
  auto|cpu|cuda) ;;
  *) die_with_code 2 "--device must be one of: auto, cpu, cuda" ;;
esac

case "$OUT_FORMAT" in
  torch|safetensors) ;;
  *) die_with_code 2 "--format must be torch or safetensors" ;;
esac

case "$RELOAD_VALIDATE_DEVICE" in
  cpu|skip) ;;
  *) die_with_code 2 "--reload-validate-device must be cpu or skip" ;;
esac

case "$LOCK_MODE" in
  auto|fd|mkdir) ;;
  *) die_with_code 2 "--lock-mode must be auto, fd, or mkdir" ;;
esac

python3 - "$QUANT_MIN_RATIO" "$LOCK_TTL_SECONDS" "$LOCK_TIMEOUT" "$SMOKE_MAX_NEW_TOKENS" "$SMOKE_TEMPERATURE" <<'PY'
import sys
from decimal import Decimal
ratio = Decimal(sys.argv[1])
if ratio < 0 or ratio > 1:
    raise SystemExit("--quant-min-ratio must be between 0 and 1")
for idx, name in [(2, "--lock-ttl-seconds"), (3, "--lock-timeout"), (4, "--smoke-max-new-tokens")]:
    v = int(sys.argv[idx])
    if name == "--lock-timeout":
        if v < -1:
            raise SystemExit(f"{name} must be >= -1")
    elif v < 0:
        raise SystemExit(f"{name} must be >= 0")
float(sys.argv[5])
PY

if [[ "$DO_VALIDATE" != "1" && "$RELOAD_VALIDATE_DEVICE" != "skip" ]]; then
  die_with_code 2 "--reload-validate-device cannot be used unless validation is enabled"
fi

CURRENT_STAGE="preflight"
SCRIPT_PATH="$(canon_path "$0")"
SCRIPT_SHA256="$(sha256_file "$SCRIPT_PATH")"
export SCRIPT_SHA256

CACHE_DIR="$(canon_path "$CACHE_DIR")"
if [[ -n "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(canon_path "$OUTPUT_DIR")"
else
  OUTPUT_DIR="$CACHE_DIR"
fi

REPO_DIR_NAME="models--$(hf_repo_dir "$MODEL_ID")"
HF_MODEL_ROOT="$CACHE_DIR/$REPO_DIR_NAME"
OUT_REPO_DIR_NAME="models--$(hf_repo_dir "${MODEL_ID}${OUT_SUFFIX}")"
OUT_ROOT="$OUTPUT_DIR/$OUT_REPO_DIR_NAME"
SNAPSHOTS_DIR="$OUT_ROOT/snapshots"
REFS_DIR="$OUT_ROOT/refs"
LOCKS_DIR="$OUT_ROOT/.locks"
REVISION_KEY="$(fs_token "$REVISION")"
REVISION_REF_PATH="$REFS_DIR/$REVISION_KEY"
REVISION_TMP_PATH="$REFS_DIR/.${REVISION_KEY}.tmp.$$"
LOCK_BASE_PATH=""

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$SNAPSHOTS_DIR" "$REFS_DIR" "$LOCKS_DIR"
chmod 700 "$LOCKS_DIR" 2>/dev/null || true

have python3 || die "python3 is required"
have git || log "git not found; remote commit resolution may be limited"

MIN_TORCH_VERSION="${MIN_TORCH_VERSION:-2.1.0}"
MIN_TRANSFORMERS_VERSION="${MIN_TRANSFORMERS_VERSION:-4.40.0}"
MIN_TORCHAO_VERSION="${MIN_TORCHAO_VERSION:-0.10.0}"

PY_INFO="$({ python3 - <<'PY'
import importlib, json
mods = {}
for name in ["torch", "transformers", "torchao", "packaging"]:
    try:
        m = importlib.import_module(name)
        mods[name] = getattr(m, "__version__", "unknown")
    except Exception as e:
        mods[name] = f"ERROR:{e}"
print(json.dumps(mods))
PY
} )"

TORCH_VERSION="$(python3 - "$PY_INFO" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["torch"])
PY
)"
TRANSFORMERS_VERSION="$(python3 - "$PY_INFO" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["transformers"])
PY
)"
TORCHAO_VERSION="$(python3 - "$PY_INFO" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["torchao"])
PY
)"
PACKAGING_VERSION="$(python3 - "$PY_INFO" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["packaging"])
PY
)"

[[ "$PACKAGING_VERSION" != ERROR:* ]] || die "python package 'packaging' is required"
[[ "$TORCH_VERSION" != ERROR:* ]] || die "python package 'torch' is required"
[[ "$TRANSFORMERS_VERSION" != ERROR:* ]] || die "python package 'transformers' is required"
[[ "$TORCHAO_VERSION" != ERROR:* ]] || die "python package 'torchao' is required"

[[ "$(compare_versions_ge "$TORCH_VERSION" "$MIN_TORCH_VERSION")" == "1" ]] || die "torch >= $MIN_TORCH_VERSION required (found $TORCH_VERSION)"
[[ "$(compare_versions_ge "$TRANSFORMERS_VERSION" "$MIN_TRANSFORMERS_VERSION")" == "1" ]] || die "transformers >= $MIN_TRANSFORMERS_VERSION required (found $TRANSFORMERS_VERSION)"
[[ "$(compare_versions_ge "$TORCHAO_VERSION" "$MIN_TORCHAO_VERSION")" == "1" ]] || die "torchao >= $MIN_TORCHAO_VERSION required (found $TORCHAO_VERSION)"

if [[ "$DEVICE_MODE" == "cuda" || "$DEVICE_MODE" == "auto" ]]; then
  CUDA_OK="$({ python3 - <<'PY'
import json, torch
ok = bool(torch.cuda.is_available())
print(json.dumps({"cuda_available": ok, "count": torch.cuda.device_count() if ok else 0}))
PY
} )"
  CUDA_AVAILABLE="$(python3 - "$CUDA_OK" <<'PY'
import json, sys
print("1" if json.loads(sys.argv[1])["cuda_available"] else "0")
PY
)"
  if [[ "$DEVICE_MODE" == "cuda" && "$CUDA_AVAILABLE" != "1" ]]; then
    die "--device cuda requested, but torch.cuda.is_available() is false"
  fi
fi

CURRENT_STAGE="resolve-source"

looks_like_full_commit_sha() {
  [[ "$1" =~ ^[0-9a-fA-F]{40}$ ]]
}

resolve_commit_remote() {
  local model_id="$1"
  local rev="$2"
  if looks_like_full_commit_sha "$rev"; then
    printf '%s\n' "${rev,,}"
    return 0
  fi
  # For annotated tags, use ^{} to peel to the commit
  local peeled_result
  peeled_result="$(git ls-remote "https://huggingface.co/${model_id}" "refs/tags/${rev}^{}" 2>/dev/null | awk 'NR==1{print $1}')"
  if [[ -n "$peeled_result" ]]; then
    echo "$peeled_result"
    return 0
  fi
  # Fall back to direct resolution for branches and lightweight tags
  git ls-remote "https://huggingface.co/${model_id}" "refs/heads/${rev}" "refs/tags/${rev}" "$rev" 2>/dev/null | awk 'NR==1{print $1}' || true
}

resolve_commit_local_ref() {
  local repo_root="$1"
  local rev="$2"

  # Validate the revision string first
  if ! validate_revision_string "$rev"; then
    return 1
  fi

  # Use Python for safe path construction
  python3 - "$repo_root" "$rev" <<'PY'
import os, sys
repo_root, rev = sys.argv[1:3]
# Sanitize revision - remove any path separators except forward slash
# and ensure it doesn't escape the refs directory
rev_parts = rev.split('/')
safe_parts = [part for part in rev_parts if part and part not in ('.', '..')]
safe_rev = '/'.join(safe_parts)
ref_file = os.path.join(repo_root, 'refs', safe_rev)
# Verify the path is within repo_root/refs
real_repo = os.path.realpath(repo_root)
expected_prefix = os.path.join(real_repo, 'refs') + os.sep
try:
    real_ref = os.path.realpath(ref_file)
    if not real_ref.startswith(expected_prefix):
        sys.exit(1)
    if os.path.isfile(real_ref):
        with open(real_ref, 'r') as f:
            commit = f.read().strip()
            if commit:
                print(commit)
                sys.exit(0)
except:
    pass
sys.exit(1)
PY
}

resolve_latest_snapshot_local() {
  local repo_root="$1"
  local snapshots="$repo_root/snapshots"
  [[ -d "$snapshots" ]] || return 1
  python3 - "$snapshots" <<'PY'
import os, sys
root = sys.argv[1]
items = []
for name in os.listdir(root):
    p = os.path.join(root, name)
    if os.path.isdir(p):
        try:
            m = os.path.getmtime(p)
        except OSError:
            continue
        items.append((m, name))
if not items:
    raise SystemExit(1)
items.sort(reverse=True)
print(items[0][1])
PY
}

RESOLVED_COMMIT=""
if [[ "$ONLINE" == "1" ]]; then
  log "Resolving revision via remote: $MODEL_ID @ $REVISION"
  RESOLVED_COMMIT="$(resolve_commit_remote "$MODEL_ID" "$REVISION")"
  [[ -n "$RESOLVED_COMMIT" ]] || die "Failed to resolve remote revision '$REVISION' for '$MODEL_ID'"
else
  # Check if revision looks like a full commit SHA
  if looks_like_full_commit_sha "$REVISION"; then
    # For full SHAs, check if the snapshot exists directly
    sha_lower="${REVISION,,}"
    if [[ -d "$HF_MODEL_ROOT/snapshots/$sha_lower" ]]; then
      RESOLVED_COMMIT="$sha_lower"
      log "Using direct commit SHA: $RESOLVED_COMMIT"
    else
      die "Snapshot for commit SHA '$REVISION' not found in cache"
    fi
  else
    # Try to resolve through refs
    RESOLVED_COMMIT="$(resolve_commit_local_ref "$HF_MODEL_ROOT" "$REVISION" || true)"
    if [[ -z "$RESOLVED_COMMIT" && "$OFFLINE_PICK_LATEST" == "1" ]]; then
      log "Offline refs missing; picking newest cached source snapshot"
      RESOLVED_COMMIT="$(resolve_latest_snapshot_local "$HF_MODEL_ROOT" || true)"
    fi
  fi
  [[ -n "$RESOLVED_COMMIT" ]] || die "Offline resolution failed for '$MODEL_ID' revision '$REVISION'"
fi

SRC_SNAPSHOT_PATH="$HF_MODEL_ROOT/snapshots/$RESOLVED_COMMIT"
[[ -d "$SRC_SNAPSHOT_PATH" ]] || die "Resolved source snapshot does not exist: $SRC_SNAPSHOT_PATH"
[[ -f "$SRC_SNAPSHOT_PATH/config.json" ]] || die "Source snapshot missing config.json: $SRC_SNAPSHOT_PATH"

CURRENT_STAGE="fingerprint"
readarray -t FP_LINES < <(build_fingerprint_json \
  "$MODEL_ID" "$REVISION" "$RESOLVED_COMMIT" "$DEVICE_MODE" "$TRUST_REMOTE_CODE" \
  "$OUT_SUFFIX" "$OUT_FORMAT" "$ALLOW_SAFETENSORS" "$MAX_SHARD_SIZE" "$DO_VALIDATE" "$RELOAD_VALIDATE_DEVICE" \
  "$DO_VALIDATE_QUANT" "$QUANT_MIN_RATIO" "$VALIDATE_ZIP_CRC" "$SMOKE_TEST" "$SMOKE_PROMPT" \
  "$SMOKE_MAX_NEW_TOKENS" "$SMOKE_TEMPERATURE")
FINGERPRINT_JSON="${FP_LINES[0]}"
FINGERPRINT_SHA256="${FP_LINES[1]}"
[[ -n "$FINGERPRINT_JSON" && -n "$FINGERPRINT_SHA256" ]] || die "Failed to build fingerprint"

OUT_SNAPSHOT_ID="${RESOLVED_COMMIT}-$(echo "$FINGERPRINT_SHA256" | cut -c1-12)"
FINAL_SNAPSHOT_PATH="$SNAPSHOTS_DIR/$OUT_SNAPSHOT_ID"
LOCK_BASE_PATH="$LOCKS_DIR/$OUT_SNAPSHOT_ID"

if [[ "$FORCE" != "1" ]]; then
  CURRENT_STAGE="reuse-check"
  if validate_existing_artifact "$FINAL_SNAPSHOT_PATH" "$FINGERPRINT_SHA256"; then
    log "Reusable output already exists: $FINAL_SNAPSHOT_PATH"
    write_ref "$REVISION_TMP_PATH" "$OUT_SNAPSHOT_ID"
    mv "$REVISION_TMP_PATH" "$REVISION_REF_PATH"
    if [[ "$JSON_MODE" == "1" ]]; then
      emit_exists_json "$FINAL_SNAPSHOT_PATH"
    else
      log "READY: $FINAL_SNAPSHOT_PATH"
    fi
    exit 0
  fi
  log "No reusable artifact found for $FINAL_SNAPSHOT_PATH (${LAST_VALIDATE_REASON:-unknown})"
fi

CURRENT_STAGE="lock"
if ! acquire_lock "$LOCK_BASE_PATH" "$LOCK_TTL_SECONDS" "$LOCK_TIMEOUT" "$LOCK_MODE"; then
  case "$LAST_LOCK_REASON" in
    busy) die "Another quantization appears to be in progress for $MODEL_ID @ $REVISION" ;;
    timeout) die "Timed out waiting for lock for $MODEL_ID @ $REVISION" ;;
    unsupported|not-regular) die "FD lock unavailable (${LAST_LOCK_REASON}) and mkdir lock failed" ;;
    fd-helper-failed) die "FD lock helper failed" ;;
    invalid-lock-mode) die "Invalid lock mode specified" ;;
    *) die "Failed to acquire lock" ;;
  esac
fi
if [[ "$LOCK_FALLBACK_MODE" == "mkdir" ]]; then
  start_mkdir_heartbeat "${LOCK_BASE_PATH}"
fi
cleanup() {
  local status=$?
  cleanup_lock
  if [[ "${PUBLISH_IN_PROGRESS:-0}" == "1" ]]; then
    if [[ -n "${TMP_PUBLISH_PATH:-}" && -e "${TMP_PUBLISH_PATH:-}" ]]; then
      safe_rm_rf "$TMP_PUBLISH_PATH" >/dev/null 2>&1 || true
    fi
    if [[ -n "${BACKUP_PATH:-}" && -e "${BACKUP_PATH:-}" ]]; then
      if [[ -e "${FINAL_SNAPSHOT_PATH:-}" ]]; then
        safe_rm_rf "$FINAL_SNAPSHOT_PATH" >/dev/null 2>&1 || true
      fi
      mv -- "$BACKUP_PATH" "$FINAL_SNAPSHOT_PATH" >/dev/null 2>&1 || true
    fi
  fi
  if [[ -n "${WORKDIR:-}" && -d "${WORKDIR:-}" ]]; then
    rm -rf -- "$WORKDIR" >/dev/null 2>&1 || true
  fi
  return "$status"
}
trap cleanup EXIT

if [[ "$FORCE" != "1" ]]; then
  CURRENT_STAGE="reuse-check-after-lock"
  if validate_existing_artifact "$FINAL_SNAPSHOT_PATH" "$FINGERPRINT_SHA256"; then
    log "Reusable output already exists after lock acquisition: $FINAL_SNAPSHOT_PATH"
    write_ref "$REVISION_TMP_PATH" "$OUT_SNAPSHOT_ID"
    mv "$REVISION_TMP_PATH" "$REVISION_REF_PATH"
    if [[ "$JSON_MODE" == "1" ]]; then
      emit_exists_json "$FINAL_SNAPSHOT_PATH"
    else
      log "READY: $FINAL_SNAPSHOT_PATH"
    fi
    exit 0
  fi
fi

CURRENT_STAGE="prepare-workdir"
WORKDIR="$SNAPSHOTS_DIR/.staging-${OUT_SNAPSHOT_ID}.$$"
mkdir -p "$WORKDIR"

cat > "$WORKDIR/quantize_fp8.py" <<'PY'
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Float8WeightOnlyConfig, quantize_

MODEL_ID = os.environ["Q_MODEL_ID"]
REVISION = os.environ["Q_REVISION"]
RESOLVED_COMMIT = os.environ["Q_RESOLVED_COMMIT"]
SRC_SNAPSHOT_PATH = os.environ["Q_SRC_SNAPSHOT_PATH"]
OUT_DIR = os.environ["Q_OUT_DIR"]
DEVICE_MODE = os.environ["Q_DEVICE_MODE"]
ONLINE = os.environ["Q_ONLINE"] == "1"
TRUST_REMOTE_CODE = os.environ["Q_TRUST_REMOTE_CODE"] == "1"
OUT_FORMAT = os.environ["Q_OUT_FORMAT"]
DO_VALIDATE = os.environ["Q_DO_VALIDATE"] == "1"
RELOAD_VALIDATE_DEVICE = os.environ["Q_RELOAD_VALIDATE_DEVICE"]
DO_VALIDATE_QUANT = os.environ["Q_DO_VALIDATE_QUANT"] == "1"
QUANT_MIN_RATIO = float(os.environ["Q_QUANT_MIN_RATIO"])
VALIDATE_ZIP_CRC = os.environ["Q_VALIDATE_ZIP_CRC"] == "1"
MAX_SHARD_SIZE = os.environ.get("Q_MAX_SHARD_SIZE") or None
SMOKE_TEST = os.environ["Q_SMOKE_TEST"] == "1"
SMOKE_PROMPT = os.environ["Q_SMOKE_PROMPT"]
SMOKE_MAX_NEW_TOKENS = int(os.environ["Q_SMOKE_MAX_NEW_TOKENS"])
SMOKE_TEMPERATURE = float(os.environ["Q_SMOKE_TEMPERATURE"])
FINGERPRINT_JSON = os.environ["Q_FINGERPRINT_JSON"]
FINGERPRINT_SHA256 = os.environ["Q_FINGERPRINT_SHA256"]
SCRIPT_SHA256 = os.environ.get("SCRIPT_SHA256")


def jlog(**payload):
    print(json.dumps(payload), file=sys.stderr, flush=True)


def pick_device():
    if DEVICE_MODE == "cpu":
        return "cpu"
    if DEVICE_MODE == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path, device: str, trust_remote_code: bool):
    local_files_only = not ONLINE
    cfg = AutoConfig.from_pretrained(
        path,
        revision=REVISION,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    kwargs = dict(
        revision=REVISION,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
        kwargs["device_map"] = None
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(
            path,
            revision=REVISION,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except Exception as e:
        jlog(level="warn", event="tokenizer_load_failed", error=str(e))
    return cfg, model, tok


def count_linear_modules(model):
    total = 0
    quantized = 0
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            total += 1
            w = getattr(mod, "weight", None)
            if w is not None and getattr(w, "dtype", None) == torch.float8_e4m3fn:
                quantized += 1
    return total, quantized


def quantize_model(model):
    cfg = Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn)
    quantize_(model, cfg)
    return cfg


def get_model_input_device(model):
    """Get the appropriate device for model inputs, handling sharded models correctly."""
    try:
        # First try to get the embedding layer's device (best for sharded models)
        emb = model.get_input_embeddings()
        if emb is not None:
            w = getattr(emb, "weight", None)
            if w is not None and hasattr(w, "device"):
                return w.device
    except Exception:
        pass

    try:
        # Fall back to first non-meta parameter
        for p in model.parameters():
            if hasattr(p, "device") and p.device.type != "meta":
                return p.device
    except Exception:
        pass

    # Default to CPU if nothing else works
    return torch.device("cpu")


def run_generation_smoke(model, tok):
    if tok is None:
        raise RuntimeError("Smoke test requested but tokenizer could not be loaded")
    # Use embedding layer device for proper sharded model support
    target_device = get_model_input_device(model)
    enc = tok(SMOKE_PROMPT, return_tensors="pt")
    enc = {k: v.to(target_device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=SMOKE_MAX_NEW_TOKENS,
            do_sample=(SMOKE_TEMPERATURE > 0),
            temperature=SMOKE_TEMPERATURE,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def build_reload_inputs(model, tok):
    if tok is not None:
        enc = tok(SMOKE_PROMPT, return_tensors="pt")
        if "input_ids" in enc:
            return {k: v.to("cpu") for k, v in enc.items() if isinstance(v, torch.Tensor)}
    cfg = getattr(model, "config", None)
    token_id = None
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(cfg, attr, None) if cfg is not None else None
        if value is not None and int(value) >= 0:
            token_id = int(value)
            break
    if token_id is None:
        vocab_size = int(getattr(cfg, "vocab_size", 0) or 0) if cfg is not None else 0
        if vocab_size > 0:
            token_id = 0
    if token_id is None:
        raise RuntimeError("Could not build reload validation input")
    return {"input_ids": torch.tensor([[token_id]], dtype=torch.long)}


def validate_saved_artifacts(out_dir: Path):
    if not (out_dir / "config.json").is_file():
        raise RuntimeError("Saved output missing config.json")
    shard_candidates = list(out_dir.glob("*.bin")) + list(out_dir.glob("*.safetensors"))
    has_index = (out_dir / "pytorch_model.bin.index.json").is_file() or (out_dir / "model.safetensors.index.json").is_file()
    if not shard_candidates and not has_index:
        raise RuntimeError("Saved output missing model shards or index")
    if VALIDATE_ZIP_CRC:
        import zipfile
        for p in out_dir.glob("*.bin"):
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as zf:
                    bad = zf.testzip()
                    if bad is not None:
                        raise RuntimeError(f"ZIP CRC failure in {p.name}: {bad}")


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_checksum_manifest(out_dir: Path):
    files = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "CHECKSUMS.sha256.json":
            continue
        rel = p.relative_to(out_dir).as_posix()
        files.append({
            "relative_path": rel,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    return {"algorithm": "sha256", "files": files}


def cleanup_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def reload_validate(out_dir: Path):
    if RELOAD_VALIDATE_DEVICE == "skip" and not SMOKE_TEST:
        return {
            "ok": True,
            "skipped": True,
            "device": "skip",
        }
    result = {
        "ok": False,
        "device": RELOAD_VALIDATE_DEVICE,
        "quant_ratio": None,
        "forward_pass_ok": False,
        "smoke_text_preview": None,
        "error": None,
    }
    _, model2, tok2 = load_model(str(out_dir), RELOAD_VALIDATE_DEVICE, TRUST_REMOTE_CODE)
    try:
        total, quantized = count_linear_modules(model2)
        ratio = (quantized / total) if total else 1.0
        result["quant_ratio"] = ratio
        if DO_VALIDATE_QUANT and total and ratio < QUANT_MIN_RATIO:
            raise RuntimeError(
                f"Reload validation quant coverage below threshold: got {ratio:.4f}, required >= {QUANT_MIN_RATIO:.4f}"
            )
        inputs = build_reload_inputs(model2, tok2)
        model2.eval()
        with torch.no_grad():
            outputs = model2(**inputs)
        if outputs is None:
            raise RuntimeError("Reload validation forward pass returned no outputs")
        result["forward_pass_ok"] = True
        if SMOKE_TEST:
            smoke_text = run_generation_smoke(model2, tok2)
            result["smoke_text_preview"] = smoke_text[:500]
        result["ok"] = True
        return result
    except Exception as e:
        result["error"] = str(e)
        raise
    finally:
        model2 = None
        tok2 = None
        cleanup_memory()


def main():
    t0 = time.time()
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_device = pick_device()
    jlog(level="info", event="load_begin", model_id=MODEL_ID, revision=REVISION, source=SRC_SNAPSHOT_PATH, device=load_device)
    cfg, model, tok = load_model(SRC_SNAPSHOT_PATH, load_device, TRUST_REMOTE_CODE)
    total_before, quant_before = count_linear_modules(model)
    jlog(level="info", event="load_done", total_linear=total_before, already_quantized=quant_before)

    qcfg = quantize_model(model)
    total_after, quant_after = count_linear_modules(model)
    ratio = (quant_after / total_after) if total_after else 1.0
    jlog(level="info", event="quant_done", total_linear=total_after, quantized_linear=quant_after, quant_ratio=ratio)

    if DO_VALIDATE_QUANT and total_after and ratio < QUANT_MIN_RATIO:
        raise RuntimeError(
            f"Quant coverage below threshold: got {ratio:.4f}, required >= {QUANT_MIN_RATIO:.4f} "
            f"({quant_after}/{total_after} Linear layers quantized)"
        )

    save_kwargs = {}
    if MAX_SHARD_SIZE:
        save_kwargs["max_shard_size"] = MAX_SHARD_SIZE
    save_kwargs["safe_serialization"] = (OUT_FORMAT == "safetensors")

    jlog(level="info", event="save_begin", out_dir=str(out_dir), format=OUT_FORMAT)
    model.save_pretrained(out_dir, **save_kwargs)
    if tok is not None:
        tok.save_pretrained(out_dir)
    cfg.save_pretrained(out_dir)

    post_save_checks = DO_VALIDATE or SMOKE_TEST
    if post_save_checks:
        validate_saved_artifacts(out_dir)

    # Properly release the model and tokenizer before reload validation
    model = None
    tok = None
    cleanup_memory()

    reload_result = None
    if post_save_checks:
        jlog(level="info", event="reload_validate_begin", device=("cpu" if (DO_VALIDATE or SMOKE_TEST) else RELOAD_VALIDATE_DEVICE))
        reload_result = reload_validate(out_dir)
        if SMOKE_TEST:
            jlog(level="info", event="smoke_done", text_preview=(reload_result.get("smoke_text_preview") or "")[:200])
        (out_dir / "reload_validation.json").write_text(json.dumps(reload_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        jlog(level="info", event="reload_validate_done", ok=reload_result.get("ok"), quant_ratio=reload_result.get("quant_ratio"), forward_pass_ok=reload_result.get("forward_pass_ok"))

    smoke_text = (reload_result or {}).get("smoke_text_preview")

    meta = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_commit": RESOLVED_COMMIT,
        "format": OUT_FORMAT,
        "device_mode": DEVICE_MODE,
        "load_device": load_device,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "source_snapshot": SRC_SNAPSHOT_PATH,
        "fingerprint_sha256": FINGERPRINT_SHA256,
        "quant_coverage": {
            "total_linear": total_after,
            "quantized_linear": quant_after,
            "ratio": ratio,
            "min_ratio_required": QUANT_MIN_RATIO,
        },
        "quantization": {
            "scheme": "torchao.Float8WeightOnlyConfig",
            "weight_dtype": "torch.float8_e4m3fn",
            "set_inductor_config": False,
            "repr": repr(qcfg),
        },
        "validation": {
            "enabled": DO_VALIDATE,
            "reload_validate_device": RELOAD_VALIDATE_DEVICE,
            "reload_validation": reload_result,
        },
        "build": {
            "script_sha256": SCRIPT_SHA256,
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "unknown"),
        },
        "timing": {
            "seconds_total": int(round(time.time() - t0)),
        },
        "smoke_test": {
            "enabled": SMOKE_TEST,
            "prompt": SMOKE_PROMPT if SMOKE_TEST else None,
            "max_new_tokens": SMOKE_MAX_NEW_TOKENS if SMOKE_TEST else None,
            "temperature": SMOKE_TEMPERATURE if SMOKE_TEST else None,
            "text_preview": (smoke_text[:500] if smoke_text else None),
        },
    }

    manifest = None
    if post_save_checks:
        manifest = build_checksum_manifest(out_dir)
        meta["manifest_file_count"] = len(manifest["files"])

    (out_dir / "quantize_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "FINGERPRINT.json").write_text(FINGERPRINT_JSON + "\n", encoding="utf-8")
    (out_dir / "FINGERPRINT.sha256").write_text(FINGERPRINT_SHA256 + "\n", encoding="utf-8")

    if post_save_checks and manifest:
        (out_dir / "CHECKSUMS.sha256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_saved_artifacts(out_dir)

    jlog(level="info", event="done", elapsed_s=int(round(time.time() - t0)))


if __name__ == "__main__":
    main()
PY

CURRENT_STAGE="quantize"
export Q_MODEL_ID="$MODEL_ID"
export Q_REVISION="$REVISION"
export Q_RESOLVED_COMMIT="$RESOLVED_COMMIT"
export Q_SRC_SNAPSHOT_PATH="$SRC_SNAPSHOT_PATH"
export Q_OUT_DIR="$WORKDIR/out"
export Q_DEVICE_MODE="$DEVICE_MODE"
export Q_ONLINE="$ONLINE"
export Q_TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE"
export Q_OUT_FORMAT="$OUT_FORMAT"
export Q_DO_VALIDATE="$DO_VALIDATE"
export Q_RELOAD_VALIDATE_DEVICE="$RELOAD_VALIDATE_DEVICE"
export Q_DO_VALIDATE_QUANT="$DO_VALIDATE_QUANT"
export Q_QUANT_MIN_RATIO="$QUANT_MIN_RATIO"
export Q_VALIDATE_ZIP_CRC="$VALIDATE_ZIP_CRC"
export Q_MAX_SHARD_SIZE="$MAX_SHARD_SIZE"
export Q_SMOKE_TEST="$SMOKE_TEST"
export Q_SMOKE_PROMPT="$SMOKE_PROMPT"
export Q_SMOKE_MAX_NEW_TOKENS="$SMOKE_MAX_NEW_TOKENS"
export Q_SMOKE_TEMPERATURE="$SMOKE_TEMPERATURE"
export Q_FINGERPRINT_JSON="$FINGERPRINT_JSON"
export Q_FINGERPRINT_SHA256="$FINGERPRINT_SHA256"

python3 "$WORKDIR/quantize_fp8.py"

CURRENT_STAGE="publish"
STAGED_OUT="$WORKDIR/out"
[[ -d "$STAGED_OUT" ]] || die "staged output directory missing after quantization: $STAGED_OUT"
if [[ "$DO_VALIDATE" == "1" || "$SMOKE_TEST" == "1" ]]; then
  full_reuse_check "$STAGED_OUT" "$FINGERPRINT_SHA256" || die "staged output failed validation: ${LAST_VALIDATE_REASON:-unknown}"
fi

BACKUP_PATH=""
TMP_PUBLISH_PATH="$SNAPSHOTS_DIR/.publish-${OUT_SNAPSHOT_ID}.$$"
PUBLISH_IN_PROGRESS=1
mv "$STAGED_OUT" "$TMP_PUBLISH_PATH"
if [[ -e "$FINAL_SNAPSHOT_PATH" ]]; then
  BACKUP_PATH="$SNAPSHOTS_DIR/.backup-${OUT_SNAPSHOT_ID}.$$"
  mv "$FINAL_SNAPSHOT_PATH" "$BACKUP_PATH"
fi
mv "$TMP_PUBLISH_PATH" "$FINAL_SNAPSHOT_PATH"
write_ref "$REVISION_TMP_PATH" "$OUT_SNAPSHOT_ID"
mv "$REVISION_TMP_PATH" "$REVISION_REF_PATH"
if [[ -n "$BACKUP_PATH" && -e "$BACKUP_PATH" ]]; then
  safe_rm_rf "$BACKUP_PATH"
fi
PUBLISH_IN_PROGRESS=0
TMP_PUBLISH_PATH=""

CURRENT_STAGE="success"
ELAPSED="$(( $(date +%s) - START_EPOCH ))"
if [[ "$JSON_MODE" == "1" ]]; then
  emit_success_json "$FINAL_SNAPSHOT_PATH" "$ELAPSED"
else
  log "READY: $FINAL_SNAPSHOT_PATH"
  log "Model ID: ${MODEL_ID}${OUT_SUFFIX}"
  log "Revision ref updated: $REVISION_REF_PATH -> $OUT_SNAPSHOT_ID"
  log "Elapsed: ${ELAPSED}s"
  log ""
  log "Transformers load example:"
  log "  AutoModelForCausalLM.from_pretrained('$FINAL_SNAPSHOT_PATH', local_files_only=True)"
  log ""
  log "vLLM examples:"
  log "  vllm serve '$OUT_ROOT' --quantization torchao --dtype bfloat16"
  log "  vllm serve '$FINAL_SNAPSHOT_PATH' --quantization torchao --dtype bfloat16"
fi
