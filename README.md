# Model Quantizer

Flox environment for quantizing HuggingFace models for offline vLLM inference. Three quantization backends cover the full precision-compression spectrum: AWQ 4-bit, FP8 via torchao, and LLM Compressor (FP8, GPTQ, W8A8, NVFP4). Output is written in HuggingFace hub cache layout so vLLM discovers quantized checkpoints with `HF_HUB_OFFLINE=1` and no extra configuration.

Python 3.13 | PyTorch 2.9.1 (CUDA) | x86\_64-linux, aarch64-linux


## How It Works

This repository is a [Flox](https://flox.dev) environment. Flox is a package manager built on Nix that defines your entire development toolchain — system packages, Python runtime, CUDA libraries — in a single declarative manifest (`.flox/env/manifest.toml`). The `.flox/` directory travels with the repo, so anyone who clones it gets an identical environment without installing anything manually beyond Flox itself.

Running `flox activate` does the following:

1. Provides Python 3.13 and PyTorch 2.9.1 with CUDA support from the Flox catalog (no pip/conda)
2. Creates a Python venv (first run only) and installs PyPI packages: torchao, transformers, accelerate, safetensors, huggingface-hub, autoawq, llmcompressor
3. Removes the PyPI torch so Python falls through to the Flox-provided CUDA-enabled build via `--system-site-packages`
4. Applies compatibility patches for AutoAWQ (see [AutoAWQ Compatibility Patches](#autoawq-compatibility-patches))
5. Provides `quantize-awq`, `quantize-fp8`, `quantize-llmc`, and `list-models` commands (from the `model-quantizer` package)

No Docker, no conda, no manual virtualenv management. Clone the repo, install Flox (<70MB), activate, quantize.


## Setup

1. **Install Flox** (one-time): follow the instructions at [flox.dev/docs](https://flox.dev/docs/install-flox/install/) for your platform (apt, rpm, nix, or Docker).
2. **Clone and activate**:

```bash
git clone <this-repo>
cd model-quantizer
flox activate
```

The first activation provisions the Python venv and installs PyPI packages. Subsequent activations are instant.


## Quick Start

```bash
flox activate

# AWQ 4-bit (best compression, ~3.5x smaller)
quantize-awq Qwen/Qwen3-8B

# FP8 via torchao (native Hopper/Blackwell, ~2x smaller)
quantize-fp8 Qwen/Qwen3-8B

# LLM Compressor -- FP8 dynamic (data-free, compressed-tensors for vLLM)
quantize-llmc Qwen/Qwen3-8B

# LLM Compressor -- W4A16 GPTQ (calibration-based)
quantize-llmc Qwen/Qwen3-8B gptq --online

# List cached source and quantized models
list-models
```

All scripts default to offline mode, so models must already be in the cache directory. Pass `--online` (FP8, LLMC) or set `HF_OFFLINE=0` (AWQ) to download models on the fly. If `HF_TOKEN` is set in your shell environment, the HuggingFace libraries will use it automatically for gated model access.

The cache directory defaults to `$FLOX_ENV_PROJECT` (models live under `$FLOX_ENV_PROJECT/hub/models--*/`) but can be pointed anywhere — either by overriding `MODEL_CACHE_DIR` at activation time or by editing the default in `.flox/env/manifest.toml`:

```bash
# Override at activation time
MODEL_CACHE_DIR=/data/models flox activate

# Or change the default permanently in the manifest [vars] section:
#   MODEL_CACHE_DIR = "/data/models"
```


## Features

- Three quantization backends: Legacy AWQ (4-bit INT) (now deprecated; use with older models only); FP8 torchao (E4M3); LLM Compressor (FP8, GPTQ, W8A8, NVFP4)
- Offline-first: all scripts default to cache-only model loading
- HF hub cache output layout: quantized models appear as siblings of source models, ready for vLLM
- Content-addressed output: each run produces a deterministic snapshot ID derived from a full configuration fingerprint; identical parameters always map to the same output path
- Idempotent: re-running with the same parameters skips quantization if output already exists
- Concurrent-safe: file-level locking prevents races when multiple quantization jobs target the same output directory
- Smoke tests: optional forward pass and token generation on quantized output to verify correctness before publishing
- JSON output mode: machine-readable status for pipeline integration (`--json` on all three scripts)
- Checksum manifests (AWQ): opt-in SHA-256 checksums for all output files
- CI-ready: same Flox environment in local dev and CI, with GitHub Actions support (see [CI / Pipeline Usage](#ci--pipeline-usage))
- Auto-provisioned Python venv with PyPI packages on first activation


## Quantization Methods

### AWQ 4-bit (`quantize-awq`)

Uses [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) for activation-aware 4-bit integer weight quantization. Produces the best compression ratio (~3.5x) with good quality retention. Works on all CUDA GPUs.

```bash
# Default: 4-bit, group_size=128
quantize-awq Qwen/Qwen3-8B

# Custom bit-width and group size
quantize-awq Qwen/Qwen3-8B 4 64

# Force re-quantize an existing output
FORCE_REQUANTIZE=1 quantize-awq Qwen/Qwen3-8B

# JSON output for scripting
quantize-awq --json Qwen/Qwen3-8B

# Skip smoke test for faster turnaround
quantize-awq --smoke-test off Qwen/Qwen3-8B
```

Output model is saved as `<model-id>-AWQ` in the cache directory.

### FP8 via torchao (`quantize-fp8`)

Uses [torchao](https://github.com/pytorch/ao) to convert BF16 weights to FP8 E4M3 (weight-only, data-free). Native hardware support on Hopper (SM90) and Blackwell (SM120). ~2x compression with near-zero quality loss.

```bash
# Default: offline, auto device selection
quantize-fp8 Qwen/Qwen3-8B

# Force rebuild, run smoke test after
quantize-fp8 --force --smoke-test Qwen/Qwen3-8B

# Safetensors output (experimental)
quantize-fp8 --allow-safetensors Qwen/Qwen3-8B

# Custom shard size for large models
quantize-fp8 --max-shard-size 4GB Qwen/Qwen3-8B

# JSON output for scripting
quantize-fp8 --json Qwen/Qwen3-8B
```

Output model is saved as `<model-id>-FP8-TORCHAO` in the cache directory.

### LLM Compressor (`quantize-llmc`)

Uses [llm-compressor](https://github.com/vllm-project/llm-compressor) (vLLM project) for unified quantization. Produces `compressed-tensors` format loaded natively by vLLM without format conversion.

| Scheme | Command | Calibration | Output Suffix |
|--------|---------|-------------|---------------|
| FP8 dynamic | `quantize-llmc <model>` | No (data-free) | `-FP8-DYNAMIC` |
| FP8 block | `quantize-llmc <model> fp8 --fp8-scheme block` | No (data-free) | `-FP8-BLOCK` |
| W4A16 GPTQ | `quantize-llmc <model> gptq` | Yes | `-W4A16-GPTQ` |
| W8A8 SmoothQuant | `quantize-llmc <model> w8a8` | Yes | `-W8A8-SMOOTHQUANT` |
| NVFP4 | `quantize-llmc <model> nvfp4` | Yes | `-NVFP4` |

```bash
# FP8 dynamic (default, data-free, no network needed)
quantize-llmc Qwen/Qwen3-8B

# W4A16 GPTQ with custom calibration parameters
quantize-llmc Qwen/Qwen3-8B gptq --online --num-samples 1024 --seq-length 4096

# W8A8 SmoothQuant
quantize-llmc Qwen/Qwen3-8B w8a8 --online

# NVFP4 (Blackwell-native)
quantize-llmc Qwen/Qwen3-8B nvfp4 --online

# Validate output by loading in vLLM and running generation
quantize-llmc Qwen/Qwen3-8B --validate --validate-prompt "Explain gravity."

# Use a local calibration dataset
quantize-llmc Qwen/Qwen3-8B gptq --dataset-path ./calibration.jsonl --text-column content

# Sequential pipeline for large models that exceed GPU memory
quantize-llmc Qwen/Qwen3-32B gptq --online --pipeline sequential

# JSON output for scripting
quantize-llmc --json Qwen/Qwen3-8B
```

Schemes that require calibration (`gptq`, `w8a8`, `nvfp4`) need a dataset. The default is `open_platypus`. Pass `--online` if the dataset is not already cached.


## When to Use What

| Method | Compression | Quality | GPU Support | Best For |
|--------|-------------|---------|-------------|----------|
| AWQ 4-bit | ~3.5x | Good | All CUDA | Fitting larger models in limited VRAM |
| FP8 torchao | ~2x | Excellent | SM90+ | Quick FP8 checkpoint, data-free |
| FP8 dynamic/block (llmc) | ~2x | Excellent | SM90+ | compressed-tensors for vLLM |
| W4A16 GPTQ (llmc) | ~3.5x | Good | All CUDA | 4-bit with vLLM-native format |
| W8A8 SmoothQuant (llmc) | ~2x | Excellent | All CUDA | Production throughput, 8-bit |
| NVFP4 (llmc) | ~4x | Good | SM120 | Native Blackwell 4-bit float |

### Model Sizing Reference (32 GB VRAM)

| Model | BF16 | AWQ/GPTQ 4-bit | FP8 | NVFP4 |
|-------|------|----------------|-----|-------|
| 7-8B | 16 GB | 4.5 GB | 8 GB | ~4 GB |
| 14B | 28 GB | 8 GB | 14 GB | ~7 GB |
| 32B | 64 GB | 18 GB | 32 GB | ~16 GB |
| 70B | 140 GB | 40 GB | 70 GB | ~35 GB |


## Output Layout

When `WRITE_LOCAL_REPO_LAYOUT=1` (the default), quantized models are written in HuggingFace hub cache structure:

```
$QUANTIZED_OUTPUT_DIR/
  hub/
    models--Qwen--Qwen3-8B/              # source model (read-only)
      refs/main
      snapshots/<commit>/
        config.json, model.safetensors, ...
    models--Qwen--Qwen3-8B-AWQ/          # quantized output (AWQ example)
      refs/main
      snapshots/<snapshot-id>/
        config.json
        model.safetensors
        tokenizer.json
        tokenizer_config.json
        FINGERPRINT.json                  # AWQ and FP8 only
        awq_quantize_meta.json            # AWQ only
```

The `<snapshot-id>` is a hash of the full quantization fingerprint (model ID, source commit, quant parameters, seed, etc.) — SHA-256 for AWQ, SHA-1 for FP8 and LLMC. This makes each output content-addressed: changing any parameter produces a new snapshot directory, while identical parameters reuse the existing one.

When `WRITE_LOCAL_REPO_LAYOUT=0`, output is written to a flat directory under `$QUANTIZED_OUTPUT_DIR/<model-id-suffix>/<snapshot-id>/`.


## Configuration

### Global Environment Variables

These are set in the `[vars]` section of `.flox/env/manifest.toml` and apply to all scripts. Change the defaults directly in the manifest, or override per-session at activation time (e.g., `MODEL_CACHE_DIR=/data/models flox activate`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_DIR` | `$FLOX_ENV_PROJECT` | HuggingFace cache root (scripts append `hub/models--*/`) |
| `QUANTIZED_OUTPUT_DIR` | `$FLOX_ENV_PROJECT` | Output root for quantized models |
| `WRITE_LOCAL_REPO_LAYOUT` | `1` | Write HF hub cache layout for vLLM auto-discovery |

### AWQ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_REVISION` | `main` | HF revision (branch, tag, or 40-char commit hash) |
| `HF_OFFLINE` | `1` | Cache-only model loading (no network) |
| `TRUST_REMOTE_CODE` | `0` | Allow execution of model-repo custom code |
| `FORCE_REQUANTIZE` | `0` | Remove and rebuild existing complete output |
| `SHOW_SIZES` | `0` | Print source/output size comparison after quantization |
| `REQUIRE_CUDA` | `1` | Fail if CUDA is unavailable |
| `DEVICE_MAP` | (dynamic) | `auto`, `cuda0`, or `cpu`. Default: `cuda0` if CUDA available, else `cpu` |
| `QUANT_SEED` | `1337` | RNG seed for reproducibility |
| `DETERMINISTIC` | `0` | Request deterministic CUDA algorithms |
| `QUANT_SHARD_SIZE` | `5GB` | Weight shard size |
| `QUANT_SAFETENSORS` | `1` | Save in safetensors format |
| `SMOKE_TEST_MODE` | `full` | `full` (reload + generate), `fast` (in-memory), or `off` |
| `WRITE_CHECKSUMS` | `0` | Write `FILES_SHA256.json` manifest |
| `LOCK_TIMEOUT_SECONDS` | `0` | Seconds to wait for output lock (0 = fail immediately) |
| `LOCK_METHOD` | `auto` | `auto`, `flock`, or `mkdir` |
| `LOCK_STALE_SECONDS` | `0` | Remove stale locks older than this (0 = disabled) |
| `AWQ_CALIB_DATASET` | (unset) | Override calibration dataset |
| `AWQ_MAX_CALIB_SAMPLES` | (unset) | Max calibration samples |
| `AWQ_MAX_CALIB_SEQ_LEN` | (unset) | Max calibration sequence length |
| `AWQ_N_PARALLEL_CALIB_SAMPLES` | (unset) | Parallel calibration samples |
| `DETERMINISTIC_FAIL_CLOSED` | `1` | Make determinism setup errors fatal (0 = warn and continue) |
| `WRITE_WEIGHT_CHECKSUMS` | `0` | Include weight shards in `FILES_SHA256.json` |
| `FINGERPRINT_INCLUDE_VERS` | `1` | Include torch/transformers/awq versions in fingerprint |
| `FINGERPRINT_INCLUDE_SYS` | `0` | Include host/system info in fingerprint |
| `ALLOW_REMOTE_STALE` | `0` | Allow stale-lock cleanup across hosts on shared storage |
| `PYTHON` | `python3` | Python executable |

### FP8 (torchao) Options

The FP8 script uses CLI flags for most configuration. It also respects `MODEL_CACHE_DIR` and `QUANTIZED_OUTPUT_DIR` (see Global Environment Variables above) and three version-gate env vars: `MIN_TORCH_VERSION` (default: `2.1.0`), `MIN_TRANSFORMERS_VERSION` (default: `4.40.0`), `MIN_TORCHAO_VERSION` (default: `0.10.0`).

```
quantize-fp8 <model-id> [options]

  -c, --cache-dir DIR           HF cache root
  -o, --output-dir DIR          Output root
  -r, --revision REV            HF revision (default: main)
      --device MODE             auto|cpu|cuda (default: auto)
      --online                  Allow network access
      --trust-remote-code       Allow model repo custom code
      --force                   Rebuild even if output exists
      --suffix STR              Output suffix (default: -FP8-TORCHAO)
      --format FMT              torch|safetensors (default: torch)
      --allow-safetensors       Attempt safetensors format
      --max-shard-size STR      Weight shard size (e.g. 4GB)
      --offline-pick-latest     Pick newest cached snapshot when refs are missing
      --lock-ttl-seconds N      Lock TTL for cross-host stale handling (default: 21600)
      --smoke-test              Run generation on output
      --smoke-prompt STR        Prompt for smoke test (default: "Hello")
      --smoke-max-new-tokens N  Tokens to generate (default: 1)
      --smoke-temperature F     Temperature (default: 0.0)
      --no-validate             Skip structural validation
      --no-validate-quant       Skip quantization coverage check
      --validate-zip-crc        Run zip CRC checks on .bin shards (slow)
      --quant-min-ratio FLOAT   Min fraction of quantized layers (default: 0.80)
      --json                    JSON output on stdout (logs to stderr)
```

### LLM Compressor Options

```
quantize-llmc <model-id> [scheme] [options]

Schemes: fp8 (default), gptq, w8a8, nvfp4

Quantization:
  --ignore LIST            Extra layer ignore patterns (comma-separated; repeatable)

FP8-specific:
  --fp8-scheme NAME        dynamic|block (default: dynamic)
  --fp8-pathway NAME       oneshot|model_free (default: oneshot)
  --model-free-device STR  Device for model_free_ptq (default: cuda:0)
  --model-free-workers N   Worker count for model_free_ptq (default: 8)

Calibration:
  --num-samples N          Calibration samples (default: 512)
  --seq-length N           Max sequence length (default: 2048)
  --batch-size N           Batch size (default: 1)
  --dataset NAME           HF dataset ID (default: open_platypus)
  --dataset-config NAME    HF dataset config name (optional)
  --dataset-path PATH      Local dataset (.json, .jsonl, .csv, .parquet, or directory)
  --text-column KEY        Text column name (default: text)
  --no-shuffle             Do not shuffle calibration samples
  --seed N                 RNG seed (default: 1234)
  --streaming              Stream dataset (hub ID or dvc:// path; online only)

Pipeline:
  --pipeline NAME          basic|datafree|sequential|independent
  --sequential-targets L   Decoder layer class names (comma-separated)
  --sequential-offload D   Offload device between layers (default: cpu)
  --no-qac                 Disable quantization-aware calibration
  --splits SPEC            Split percentages spec passed to oneshot
  --preprocessing-workers N  Dataset preprocessing workers
  --dataloader-workers N     DataLoader workers (default: 0)

Validation:
  --validate               Load output in vLLM and run checks
  --validate-prompt TEXT   Smoke test prompt (default: "Hello!")
  --validate-suite PATH    JSONL or txt regression suite
  --validate-seed N        Seed for vLLM sampling (default: 1)
  --validate-max-tokens N  Max tokens per prompt (default: 64)
  --validate-min-chars N   Min chars in output to count as pass (default: 1)

General:
  --model-revision REV     HF revision (default: main)
  --online                 Allow network access
  --trust-remote-code      Allow model repo custom code
  --use-auth-token         Use HuggingFace auth for private repos
  --force                  Overwrite existing output
  --suffix STR             Override output suffix
  --lock-timeout SECONDS   Lock wait time (0=fail, -1=unlimited)
  --log-dir PATH           llmcompressor log directory
  --json                   JSON output on stdout (logs to stderr)
```


## Advanced Features

### Locking

All three scripts implement file-level locking on the output directory to prevent concurrent quantization jobs from corrupting each other. The AWQ and FP8 scripts support both `flock` (preferred) and `mkdir`-based locks with configurable timeout and stale lock detection. The LLMC script uses the same pattern with `--lock-timeout`.

### Fingerprinting

All three scripts compute content-addressed snapshot IDs by hashing a fingerprint of all configuration inputs. For AWQ this includes: model ID, source commit, quantization parameters (bits, group size, zero point), calibration settings, device map, dtype policy, seed, determinism flag, save format, shard size, and optionally library versions and system info. FP8 includes a similar set plus script version and output format. LLMC hashes the model ID, resolved commit, revision, scheme, FP8 options, calibration parameters, and pipeline options. Identical configurations always produce the same output path.

### Determinism

Set `DETERMINISTIC=1` (AWQ) or `--seed N` (LLMC) to improve reproducibility. The AWQ script sets Python, NumPy, and PyTorch seeds, and optionally enables `torch.use_deterministic_algorithms(True)`. Full bitwise reproducibility across runs is not guaranteed due to CUDA non-determinism, but output quality is consistent.

### Smoke Tests

- **AWQ**: `--smoke-test full` (default) reloads the saved checkpoint and runs a forward pass plus short generation. `fast` tests the in-memory model immediately after quantization. `off` skips generation but still validates output structure.
- **FP8**: `--smoke-test` loads the output checkpoint and generates one token.
- **LLMC**: `--validate` spawns a separate vLLM process to load the checkpoint and run generation, verifying the output is a valid vLLM-loadable model.

### JSON Mode

All three scripts support `--json` for machine-readable output on stdout (logs go to stderr). Each emits a JSON object with `status` (`ok` or `exists`), source model, and output path. AWQ and FP8 also include source commit, snapshot ID, revision, and smoke test results; FP8 adds device mode, format, and validation coverage. LLMC includes scheme, validation status, output size, and timestamp. Successful runs (`ok`) include timing.


## CI / Pipeline Usage

The scripts are designed for unattended operation. With `--json`, all human-readable logs go to stderr and a single JSON object is written to stdout on completion, making it straightforward to parse results in a pipeline.

### Running in CI

Flox provides GitHub Actions for CI integration. The environment travels with the repo, so CI gets the same toolchain as local development.

The commands (`quantize-fp8`, `quantize-llmc`, etc.) are binaries from the `model-quantizer` package and work in all contexts — interactive sessions and CI alike:

```yaml
# .github/workflows/quantize.yml
jobs:
  quantize:
    runs-on: [self-hosted, gpu]  # needs NVIDIA GPU
    steps:
      - uses: actions/checkout@v4
      - uses: flox/install-flox-action@v2
      - uses: flox/activate-action@v1
        with:
          command: |
            quantize-fp8 --online --json Qwen/Qwen3-8B > result.json
            cat result.json
```

For non-GitHub CI (GitLab, CircleCI, Jenkins), install Flox on the runner and use `flox activate --`:

```bash
flox activate -- quantize-fp8 --online --json Qwen/Qwen3-8B > result.json
```

### Key behaviors for automation

- **Exit codes**: 0 on success or if output already exists, 1 on error
- **Idempotent**: re-running with the same parameters detects existing output and exits immediately (JSON reports `"status": "exists"`)
- **`--force`**: bypasses the exists check and re-quantizes from scratch
- **Locking**: concurrent jobs targeting the same output serialize automatically via file locks; the second job waits or skips
- **`--json`**: structured output for parsing; logs on stderr won't pollute stdout
- **`HF_TOKEN`**: set in CI secrets for gated model access — the HuggingFace libraries pick it up automatically

### Batch quantization example

```bash
#!/usr/bin/env bash
# Quantize a list of models, collect results
# Run inside flox activate, or use: flox activate -- bash batch-quantize.sh
models=(Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct google/gemma-3-4b-it)

for model in "${models[@]}"; do
  echo "--- $model ---" >&2
  quantize-llmc --online --json "$model" >> results.jsonl
done
```

Each line in `results.jsonl` is a self-contained JSON object with status, output path, timing, and validation.


## AutoAWQ Compatibility Patches

AutoAWQ 0.2.9 is the last release and is no longer maintained. The environment applies three patches on first activation to maintain compatibility with transformers 4.52+:

1. **GELUTanh rename**: `PytorchGELUTanh` was renamed to `GELUTanh` in transformers. Patched in `awq/quantize/scale.py`.
2. **Catcher attribute proxy**: The `Catcher` wrapper class in `awq/quantize/quantizer.py` does not proxy attribute access to the wrapped decoder layer. Transformers 4.57+ accesses `attention_type` on Qwen2/Qwen3 layers, crashing calibration. Patched by adding `__getattr__` fallback.
3. **Deprecation noise**: AWQ's `__init__.py` overrides Python's warning filters and emits a deprecation notice on every import. Patched to remove the `simplefilter` override and `warnings.warn` call.

These patches are applied automatically during venv provisioning and do not require manual intervention.


## Packages

From Flox (declarative, pinned):

| Package | Version | Description |
|---------|---------|-------------|
| `flox-cuda/python3Packages.torch` | 2.9.1 | PyTorch with CUDA support |
| `uv` | latest | Python package installer |

From PyPI via uv (auto-provisioned on first `flox activate`):

| Package | Description |
|---------|-------------|
| `torchao` | PyTorch native quantization (FP8 E4M3) |
| `transformers` | HuggingFace Transformers |
| `accelerate` | Model parallelism and device mapping |
| `safetensors` | Safe tensor serialization |
| `huggingface-hub` | HuggingFace Hub client |
| `autoawq` | AWQ 4-bit quantization |
| `llmcompressor` | vLLM's unified quantization library |

PyPI torch is automatically removed after installation so Python falls through to the Flox-provided CUDA-enabled torch via `--system-site-packages`.


## System Requirements

- **OS**: Linux (x86\_64 or aarch64)
- **GPU**: NVIDIA GPU with CUDA support. FP8 methods require SM90+ (Hopper: H100, H200, L40S) or SM120+ (Blackwell: RTX 5090, B200). AWQ and GPTQ work on all CUDA GPUs.
- **Driver**: NVIDIA driver compatible with CUDA 12.x (driver 525+)
- **VRAM**: Depends on model size. 7-8B models need ~16 GB for loading + quantization workspace. AWQ and GPTQ 4-bit outputs fit larger models in less VRAM at inference time.
- **Disk**: Source model + quantized output. Budget 1.5-2x the source model size for the quantization workspace.
- **Flox**: must be installed (see [Setup](#setup))


## Troubleshooting

### "Source model not found in cache"

The model has not been downloaded. Pass `--online` to allow the script to fetch it, or download the model separately before running in offline mode.

### "Lock busy"

A previous run was interrupted and left a stale lock. Remove it manually:

```bash
# AWQ locks (inside the output model directory)
rm -rf $QUANTIZED_OUTPUT_DIR/hub/models--<org>--<model>-<suffix>/.quantize.lockdir
rm -f  $QUANTIZED_OUTPUT_DIR/hub/models--<org>--<model>-<suffix>/.quantize.lock

# FP8 locks (inside the output model directory)
rm -rf $QUANTIZED_OUTPUT_DIR/hub/models--<org>--<model>-<suffix>/.quantize.lock

# LLMC locks (in the output root)
rm -f  $QUANTIZED_OUTPUT_DIR/.quantize-<model-slug>.lock
```

For AWQ, set `LOCK_STALE_SECONDS=300` to automatically clean stale locks older than 5 minutes.

### CUDA out of memory

The model is too large for available VRAM. Options:

- Close other GPU-using processes (`nvidia-smi` to check)
- Use `DEVICE_MAP=auto` or `--device auto` for automatic multi-GPU or CPU offloading
- For LLMC, use `--pipeline sequential` to quantize one decoder layer at a time
- Quantize a smaller model or use a machine with more VRAM

### "CUDA is not available"

PyTorch cannot see the GPU. Check that `nvidia-smi` works and that the Flox-provided torch has CUDA support:

```bash
flox activate
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### Venv issues after environment changes

If the Python venv gets into a bad state (version mismatches, broken patches), delete it and re-activate:

```bash
rm -rf .flox/cache/venv
flox activate  # recreates venv from scratch
```


## Resources

- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) -- AWQ quantization library
- [torchao](https://github.com/pytorch/ao) -- PyTorch native quantization
- [llm-compressor](https://github.com/vllm-project/llm-compressor) -- vLLM's unified quantization
- [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/index.html) -- Loading quantized models in vLLM
- [Flox](https://flox.dev) -- Reproducible development environments
