# NOOCR

[简体中文](README.md) | [English](README_EN.md) | [繁體中文](README_TW.md)

A full-featured OCR system built on ONNX. One core, multiple backends, CPU and GPU.

```bash
pip install -r requirements.txt      # one command, works for both CPU and GPU
python -m noocr models --get ppocrv6-tiny   # fetch weights (~6MB)
python -m noocr invoice.jpg                   # recognize
python -m noocr serve                        # Web UI at http://127.0.0.1:8000
```

On an NVIDIA GPU add `-d cuda` (or `serve --device cuda`) for a **20-40x** end-to-end speedup.

---

## Features

| | |
|---|---|
| **One dependency set** | CPU and NVIDIA machines install the same `requirements.txt`, no rebuild needed |
| **CPU / GPU dual mode** | Same code, `--device cpu` or `--device cuda`, with auto-detection |
| **Three backend tiers** | PP-OCRv6 tiny (6MB) / small (32MB, default) / v5 (22MB) |
| **Many input formats** | Images, PDF, Word, Excel, PPT, URL |
| **Reproducible output** | Recognition results are independent of `rec_batch_size` |
| **Four delivery forms** | Library / CLI / REST API / Web UI |
| **Unified contract** | Every backend returns `OCRResult`, so callers never need branching |

## Installation

Requires Python >= 3.10.

```bash
git clone https://github.com/ITerydh/NOOCR.git
cd noocr
python -m venv .venv && . .venv/Scripts/activate    # Windows
# python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### Why there is only one dependency file

`requirements.txt` installs `onnxruntime-gpu` rather than `onnxruntime`, because the former is a **superset** of the latter: it ships the CPU, CUDA and TensorRT execution providers in a single package, so one install covers both kinds of machine.

| Package | Available EPs | Size |
|---|---|---|
| `onnxruntime` | CPU | ~15 MB |
| `onnxruntime-gpu` | CPU / CUDA / TensorRT | ~700 MB |

Machines without an NVIDIA GPU can **install and run the GPU build just fine** — the GPU portion simply never activates. There is no need to distinguish a "CPU environment" from a "GPU environment", and no need to rebuild the virtual environment for GPU.

The cost is download size. If you only run on CPU and care about install time or disk usage, use the lightweight file instead:

```bash
pip install -r requirements-cpu.txt     # ~15MB ORT, identical recognition results
```

When you later want GPU, run `pip install -r requirements.txt` instead (the GPU build overwrites the CPU one).

### System-level dependencies (GPU only)

pip handles Python packages only. CUDA and cuDNN are **system-level runtimes** and must be installed separately:

1. NVIDIA driver (>= 525)
2. CUDA 12.x
3. **cuDNN 9** (note the 9.x — ORT 1.20+ depends on `cudnn64_9.dll`)

After installing, tell the project where the DLLs live (required on Windows, usually unnecessary on Linux/macOS):

```bash
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin            # Windows
export NOOCR_GPU_LIB_DIR=/usr/local/cudnn/lib       # Linux
```

The project also searches common locations automatically, including the project root and `site-packages/cudnn/` inside virtual environments.

## GPU acceleration

On GPU the end-to-end speedup over CPU is **20-40x** (see [Performance](#performance)). The Python dependency is already in `requirements.txt`, so no extra package install is needed — only the system-level CUDA 12 + cuDNN 9:

```bash
python -m noocr invoice.jpg -d cuda
python -m noocr serve --device cuda
```

`--device` accepts `auto` (default, uses CUDA when present) / `cpu` / `cuda`.

### Switching at runtime from the Web UI

After `serve` starts, **the device badge in the top bar is clickable** — no restart required:

- The badge shows the device **actually in effect** (the GPU name, highlighted green, when the GPU is active), not the value requested at startup;
- The dropdown offers `Auto / GPU / CPU`; when the machine has no usable GPU environment, "GPU" is greyed out with the reason stated;
- Switching automatically unloads the other model set and returns GPU memory, at a cost of roughly 0.4-0.6s; if a recognition result is already on screen it re-runs automatically, so you can compare speeds directly;
- When the target device is unavailable the API returns 400 and **keeps the current device unchanged**, so a working service never turns into a broken one.

You can also drive it through the API:

```bash
curl http://127.0.0.1:8000/api/device# current device
curl -X POST -F "device=cuda" .../api/device              # switch to GPU
```

### cuDNN 9 is mandatory

The CUDA EP in ONNX Runtime 1.20+ depends on **cuDNN 9** (`cudnn64_9.dll`). A wrong version fails in a particularly subtle way: ORT **does not raise an error**, it quietly hands the operators to the CPU, so you get pure-CPU performance while believing you are on the GPU.

This project defends against that in three ways:

1. At startup it **actually tests** whether the CUDA EP can initialize, using a tiny ONNX model, rather than merely checking whether it was compiled in;
2. After a session is created it **verifies the execution providers actually in effect** and raises an error on mismatch;
3. The Web UI keeps a **device badge** in the top bar that shows CPU when the GPU is not active.

Installing cuDNN 9 (extract, then place the DLLs from `bin` into any directory and tell the project where):

```bash
# Download cuDNN 9 for CUDA 12 from https://developer.nvidia.com/cudnn-downloads
# Windows needs the DLL search path registered explicitly; Linux/macOS just need permissions
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin
```

The project automatically looks in the project root and in `site-packages/cudnn/` of sibling virtual environments such as `noocr-gpu/`, so no manual configuration is needed.

> **If you keep the GPU and CPU environments separate** (recommended, to avoid ORT DLLs overwriting each other), put the cuDNN DLLs inside the GPU environment:
> `<gpu-venv>/Lib/site-packages/cudnn/`.

### One gotcha worth remembering

`cudnn_conv_algo_search` **must be set as a provider-level option**. Writing `"DEFAULT"` makes all 228 convolutions of PP-OCRv6-rec fall back to their CPU implementations, degrading a single inference from 5.9ms to 90.2ms (15x slower). Provider-level configuration also takes **priority over** SessionOptions, so writing it in the wrong place gets silently overridden. This project pins it to `"EXHAUSTIVE"` in `build_providers()`.

## Getting the weights

All 17 weight files (95.8MB) are hosted on ModelScope: **[iterhui/noocr-onnx](https://www.modelscope.cn/models/iterhui/noocr-onnx)**. Download per backend as needed:

```bash
python -m noocr models                     # show status per backend
python -m noocr models --get ppocrv6-tiny  # 6.9MB, fastest
python -m noocr models --get ppocrv6-small # 32MB, default
python -m noocr models --get ppocrv5       # 22MB, previous generation
```

Or fetch everything at once with the SDK:

```bash
pip install modelscope
modelscope download --model iterhui/noocr-onnx --local_dir ./models
```

Or download individual files directly (paths match the table below):

```
https://www.modelscope.cn/models/iterhui/noocr-onnx/resolve/master/ppocrv6/det/PP-OCRv6_det_small.onnx
```

### Where to put them

After downloading, the directory structure must look like this (`models.py` resolves paths this way):

```
<project root>/
├─ noocr/
└─ models/
   ├─ ppocrv6/
   │  ├─ det/
   │  │  ├─ PP-OCRv6_det_tiny.onnx      # tiny tier
   │  │  └─ PP-OCRv6_det_small.onnx     # small tier
   │  ├─ rec/
   │  │  ├─ PP-OCRv6_rec_tiny.onnx
   │  │  └─ PP-OCRv6_rec_small.onnx
   │  ├─ cls/
   │  │  └─ cls.onnx                    # orientation classifier (180° correction)
   │  ├─ ppocrv6_tiny_dict.txt          # tiny dictionary (6906 classes)
   │  └─ ppocrv6_dict.txt               # small dictionary (18710 classes)
   └─ ppocrv5/
      ├─ det/det.onnx
      ├─ rec/rec.onnx
      ├─ cls/cls.onnx
      └─ ppocrv5_dict.txt               # 6623 classes
```

The optional backends (`--get plate` / `orientation` / `layout` / `table`) map to:

```
models/license_plate/{car_plate_detect.onnx, plate_rec.onnx}
models/orientation/rapid_orientation.onnx
models/layout/{layout_cdla.onnx, layout_publaynet.onnx}
models/table/slanet-plus.onnx
```

When placing files manually, `python -m noocr models` marks anything missing as `缺失` (missing).

To store them elsewhere, set the environment variable:

```bash
export NOOCR_MODELS_DIR=/data/ocr/models    # Linux/macOS
set NOOCR_MODELS_DIR=D:\ocr\models          # Windows
```

## Command line

```bash
noocr <file>                     # print text
noocr <file> -o out.json         # structured JSON (with coordinates and confidence)
noocr paper.pdf -o paper.md -f md # PDF to Markdown
noocr bench exam.jpg             # per-stage timing breakdown
noocr backends                   # list backends
noocr models                     # weight status
noocr serve --port 8000          # Web UI + API
```

Common options: `-b/--backend` selects the backend, `-f/--format` selects `json|text|md`, `-d/--device` selects `auto|cpu|cuda`, `--batch` tunes the batch size, `--no-cls` disables 180° correction.

## As a library

```python
from noocr import ocr

result = ocr("scan.jpg")
print(result.text)
print(result.to_markdown())

for line in result.all_lines:
    print(f"{line.confidence:.2f}  {line.text}  {line.box.points}")
```

Choosing a backend and parameters:

```python
# Per-call override: use CPU / switch backend for this call only
r = ocr("scan.jpg", device="cpu")        # force CPU
r = ocr("scan.jpg", backend="ppocrv5")   # temporary backend switch
r = ocr("scan.jpg", dpi=300, max_pages=5)

# Fixed configuration: a long-running service should hold its own instance,
# so the model is loaded only once
from noocr import OCRPipeline

pipe = OCRPipeline(backend="ppocrv6-tiny", device="cuda", rec_batch_size=8)
for path in paths:
    print(pipe(path).text)
```

You can also grab a backend directly:

```python
from noocr.backends import get_backend

backend = get_backend("ppocrv6-tiny", rec_batch_size=8, device="cuda")
backend.load()
page = backend.recognize_image(image)      # image is a BGR ndarray
print(page.debug["stage_ms"])              # per-stage timing, to locate the slow step
```

## Web UI and API

```bash
noocr serve --host 0.0.0.0 --port 8000 --device cuda
```

The UI lives at `http://127.0.0.1:8000/` — drop in a file and it recognizes, with text / Markdown / side-by-side comparison views, the image and the detail list linked in two columns.
API docs at `http://127.0.0.1:8000/docs`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/backends` | List available backends |
| `GET` | `/api/device` | Device preference and the device **actually in effect** |
| `POST` | `/api/ocr` | Recognize an uploaded file (multipart) |
| `POST` | `/api/ocr/path?path=...` | Recognize a path on the server |
| `GET` | `/api/page/{doc_id}/{i}` | Fetch the rendered image of page i (for paging) |
| `POST` | `/api/warmup` | Preload a backend |

```bash
curl -X POST http://127.0.0.1:8000/api/ocr \
  -F "file=@invoice.jpg" -F "backend=ppocrv6-tiny"
```

## Performance

RTX 4070 Ti SUPER + ppocrv6-small, median over the sample images:

| Image | Lines | CPU | GPU | Speedup |
|---|---|---|---|---|
| Ticket | 19 | 15537 ms | **594 ms** | 26x |
| Bank statement | 30 | 13040 ms | **577 ms** | 23x |
| Lab report | 69 | 10396 ms | **567 ms** | 18x |
| ID card | 11 | 13317 ms | **400 ms** | 33x |
| Bank branch | 4 | 9414 ms | **249 ms** | 38x |

The first GPU request includes roughly 2.4s of model loading and CUDA kernel compilation; after that it settles at 0.25-0.6s.

Reproduce:

```bash
python scripts/perf/bench_device.py cpu     # CPU baseline
python scripts/perf/bench_device.py cuda    # GPU baseline
```

Other tools:

```bash
python scripts/perf/bench_buckets.py cuda   # buckets and sequential call counts
python scripts/perf/ab_tiers.py cuda        # tier count A/B (with output consistency check)
python scripts/perf/ab_cudnn.py             # cuDNN algorithm search A/B
python scripts/perf/ab_arena.py             # memory arena strategy A/B
python scripts/perf/prof_rec.py cuda        # ORT profiler, per-operator timing
```

### Memory arena and dynamic shapes

ONNX Runtime turns off the memory arena whenever `dynamic_shape=True`. That
is mandatory for models whose **width keeps changing** (rec's input width
grows with text length; forcing the arena on degrades it to seconds). But for
models whose **shape set is finite and discrete** it simply wastes the arena.

PP-OCRv6's det falls in the latter category: the short side is always 736,
the long side is capped and aligned to 32. Interleaved A/B over 5 sample
images of differing aspect ratios, 12 rounds each:

| det configuration | Median per image |
|---|---|
| `dynamic_shape=True` (arena off) | 233.3 ms |
| `dynamic_shape=False` (arena on) | **157.6 ms** (-32.4%) |

The spread ranges (A 227.7-239.2, B 149.8-168.5) do not overlap at all, so
the difference is real. End-to-end went from 1541 ms to 1430 ms.

Reproduce with `python scripts/perf/ab_arena.py`. When the two spreads
overlap, the script reports "not significant" instead of forcing a
conclusion — which is exactly what should happen on a busy machine.

## Backend comparison

| Backend | Size | Relative speed | Weighted confidence | Best for |
|---|---|---|---|---|
| `ppocrv6-tiny` | 6.1MB | **0.58x** | 0.961 | Edge devices, batch processing |
| `ppocrv6-small` | 30.5MB | 1.73x | **0.971** | Default |
| `ppocrv5` | 15.6MB | 1.00x | 0.936 | Compatibility with older projects |

## Project structure

```
noocr/
├─ types.py              unified data models (Pydantic)
├─ models.py             weight manifest and download
├─ cli.py                command line
├─ engine/               inference engine
│  ├─ session.py           session cache, device detection, GPU runtime loading
│  ├─ imageops.py          image geometry and preprocessing
│  └─ base.py             backend abstraction
├─ backends/             OCR backends
│  ├─ ppocr.py             PP-OCRv5
│  ├─ ppocrv6.py           PP-OCRv6
│  ├─ postprocess.py       DB detection post-processing
│  └─ decode.py            CTC decoding
├─ inputs/loader.py      multi-format input
├─ document/             document structuring
├─ pipeline/             orchestration
└─ web/                  Web UI and API
    ├─ app.py
    └─ templates/index.html
tests/                   tests and benchmarks
scripts/
├─ publish_weights.py      weight publishing script
├─ perf/                   performance benchmarks and A/B scripts
└─ models_repo_card.md     model card for the weight repository
```

## Troubleshooting

**`git push` fails with `Permission to <repo> denied to <another user>`**

Some Git distributions set `credential.helper = helper-selector` in the
**system-level** `gitconfig`. It takes precedence over your `~/.gitconfig`
and returns whichever account is stored in the Windows Credential Manager
— so the push is rejected even though your token is correct.

Add a repository-level override; no need to touch global settings:

```bash
git config --local credential.helper ""
git config --local credential.helper store
```

Clearing first matters: the second command alone *appends* rather than
replaces, and the empty entry is what suppresses the upper-level helper.

Verify which credential is actually used (the password is never echoed):

```bash
printf 'protocol=https\nhost=github.com\n\n' \
  | git -c credential.helper= -c credential.helper=store credential fill
```

If `username` is not the repository owner, the credential is still not
in effect.

**`UnicodeEncodeError` when printing Chinese text on Windows**

The console defaults to cp1252. The project already guards against this in
three places (package-level `logging_config`, per-script `reconfigure`, and
`PYTHONIOENCODING` in CI). If you hit it in your own script, set:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

## License

Code is Apache-2.0. Weights originate from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) and are used under the original license terms.