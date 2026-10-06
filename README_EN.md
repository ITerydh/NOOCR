# NOOCR

[简体中文](README.md) | [English](README_EN.md) | [繁體中文](README_TW.md)

ONNX-based full-featured OCR system. One kernel, multiple backends, CPU / GPU.

```bash
pip install -r requirements.txt              # one command, works for CPU and GPU
python -m noocr models --get ppocrv6-tiny    # fetch weights (~6MB)
python -m noocr receipt.jpg                  # recognize
python -m noocr serve                        # Web UI at http://127.0.0.1:8000
```

With an NVIDIA GPU add `-d cuda` (or `serve --device cuda`) for a **20-40x** speedup.

For problems, see [Q&A.md](Q&A.md).

## Features

| | |
|---|---|
| **One dependency file** | CPU and NVIDIA machines install the same `requirements.txt` |
| **CPU / GPU** | Same code, `--device cpu` or `--device cuda`, auto-detected by default |
| **Three backends** | PP-OCRv6 tiny (6MB) / small (32MB, default) / v5 (16MB) |
| **Multi-format input** | Images, PDF, Word, Excel, PPT, URL |
| **Reproducible** | Output is independent of `rec_batch_size` |
| **Four delivery forms** | Library / CLI / REST API / Web UI |
| **Unified contract** | Every backend returns `OCRResult`; callers need no branching |

## Installation

Requires Python >= 3.10.

```bash
git clone https://github.com/ITerydh/NOOCR.git
cd noocr
python -m venv .venv && . .venv/Scripts/activate    # Windows
# python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

Machines without an NVIDIA GPU install the same file and run normally; the GPU part simply never activates. If you only need CPU and care about install size, use the lightweight version (identical results):

```bash
pip install -r requirements-cpu.txt     # about 15MB
```

### Extra system libraries for GPU

CUDA and cuDNN are system-level runtime libraries that pip cannot install. Install NVIDIA driver (>= 525), CUDA 12.x, and **cuDNN 9** separately.

Then tell the project where the DLLs are (required on Windows, usually unnecessary on Linux/macOS):

```bash
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin            # Windows
export NOOCR_GPU_LIB_DIR=/usr/local/cudnn/lib       # Linux
```

The project also searches common locations such as the project root and `site-packages/cudnn/` in virtual environments. See [Q&A.md](Q&A.md#gpu-not-working) when it still does not work.

## Getting the weights

All 17 files (95.8MB) are hosted on ModelScope: **[iterhui/noocr-onnx](https://www.modelscope.cn/models/iterhui/noocr-onnx)**. Fetch per backend:

```bash
python -m noocr models                     # show status per backend
python -m noocr models --get ppocrv6-tiny  # 6.1MB, fastest
python -m noocr models --get ppocrv6-small # 30.5MB, default
python -m noocr models --get ppocrv5       # 15.6MB
```

Or fetch everything at once:

```bash
pip install modelscope
modelscope download --model iterhui/noocr-onnx --local_dir ./models
```

The directory layout must be exactly as follows (`models.py` resolves paths this way):

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
   │  │  └─ cls.onnx                    # direction classifier (180 deg)
   │  ├─ ppocrv6_tiny_dict.txt          # tiny dict (6906 classes)
   │  └─ ppocrv6_dict.txt               # small dict (18710 classes)
   └─ ppocrv5/
      ├─ det/det.onnx
      ├─ rec/rec.onnx
      ├─ cls/cls.onnx
      └─ ppocrv5_dict.txt               # 6623 classes
```

Optional backends (`--get plate` / `orientation` / `layout` / `table`) map to:

```
models/license_plate/{car_plate_detect.onnx, plate_rec.onnx}
models/orientation/rapid_orientation.onnx
models/layout/{layout_cdla.onnx, layout_publaynet.onnx}
models/table/slanet-plus.onnx
```

When placing files manually, `python -m noocr models` marks anything missing as `缺失`. To use another location, set `NOOCR_MODELS_DIR`.

## Command line

```bash
noocr <file>                     # print text
noocr <file> -o out.json        # structured JSON (with boxes and confidence)
noocr paper.pdf -o paper.md -f md # PDF to Markdown
noocr bench exam.jpg             # per-stage timing breakdown
noocr backends                   # list backends
noocr models                     # weight status
noocr serve --port 8000          # Web UI + API
```

Common options: `-b/--backend`, `-f/--format` (`json|text|md`), `-d/--device` (`auto|cpu|cuda`), `--batch`, `--no-cls` to disable 180-degree correction.

## As a library

```python
from noocr import ocr

result = ocr("scan.jpg")
print(result.text)
print(result.to_markdown())

for line in result.all_lines:
    print(f"{line.confidence:.2f}  {line.text}  {line.box.points}")
```

Backend and parameter overrides:

```python
# Per-call override: use CPU / switch backend for this call only
r = ocr("scan.jpg", device="cpu")        # force CPU
r = ocr("scan.jpg", backend="ppocrv5")   # temporary backend
r = ocr("scan.jpg", dpi=300, max_pages=5)

# Fixed configuration: a long-running service should hold its own instance,
# so the model is loaded only once
from noocr import OCRPipeline

pipe = OCRPipeline(backend="ppocrv6-tiny", device="cuda", rec_batch_size=8)
for path in paths:
    print(pipe(path).text)
```

## Web UI and API

```bash
noocr serve --host 0.0.0.0 --port 8000 --device cuda
```

The UI at `http://127.0.0.1:8000/` — upload and parameters on the left, image/text side-by-side with linked highlighting on the right, history and samples below. The device badge in the header switches CPU / GPU without restarting. API docs at `http://127.0.0.1:8000/docs`.

<details open>
<summary><b>UI screenshots</b> (click to expand)</summary>

**Side-by-side image and text** — parameters on the left; the annotated image and the recognized text side by side on the right, with history and samples below. The header badge shows the device actually in use.

![Web UI: side-by-side image and text](docs/images/demo-overview.jpg)

**Linked highlighting** — click any row in the detail table and its box turns blue on the image while the rest stay green.

![Web UI: click a row to highlight it on the image](docs/images/webui-detail.jpg)

</details>

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/backends` | Available backends |
| `GET` | `/api/device` | Preferred and **actually active** device |
| `POST` | `/api/device` | Switch device (`auto`/`cpu`/`cuda`) |
| `POST` | `/api/ocr` | Recognize an uploaded file (multipart) |
| `POST` | `/api/ocr/path?path=...` | Recognize a path on the server |
| `GET` | `/api/page/{doc_id}/{i}` | Rendered image of page i (for paging) |
| `POST` | `/api/warmup` | Preload a backend |

```bash
curl -X POST http://127.0.0.1:8000/api/ocr \
  -F "file=@receipt.jpg" -F "backend=ppocrv6-tiny"
```

## Performance

RTX 4070 Ti SUPER + ppocrv6-small, median over the sample images:

| Image | Lines | CPU | GPU | Speedup |
|---|---|---|---|---|
| Receipt | 19 | 15537 ms | **594 ms** | 26x |
| Bank statement | 30 | 13040 ms | **577 ms** | 23x |
| Lab report | 69 | 10396 ms | **567 ms** | 18x |
| ID card | 11 | 13317 ms | **400 ms** | 33x |
| Bank branch | 4 | 9414 ms | **249 ms** | 38x |

The first GPU request includes about 2.4s of model loading and CUDA kernel compilation; afterwards it stabilizes at 0.25-0.6s. Reproduce with:

```bash
python scripts/perf/bench_device.py cpu     # CPU baseline
python scripts/perf/bench_device.py cuda    # GPU baseline
```

## Backend comparison

| Backend | Size | Relative speed | Weighted confidence | Use for |
|---|---|---|---|---|
| `ppocrv6-tiny` | 6.1MB | **0.58x** | 0.961 | Edge devices, batch work |
| `ppocrv6-small` | 30.5MB | 1.73x | **0.971** | Default |
| `ppocrv5` | 15.6MB | 1.00x | 0.936 | Legacy compatibility |

## Project structure

```
noocr/
├─ types.py              unified data models (Pydantic)
├─ models.py             weight manifest and download
├─ cli.py                command line
├─ engine/               inference engine
│  ├─ session.py           session cache, device detection, GPU runtime loading
│  ├─ imageops.py          image geometry and preprocessing
│  └─ base.py              backend abstraction
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
└─ models_repo_card.md     model card for the weight repo
```

## Acknowledgements

This project is built on the work of the following open-source projects. Thanks to their authors and communities:

- **[OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)** — the ONNX inference path, model export and document structuring ideas that became the starting point of this project.
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** — the PP-OCR model family and algorithm design; all three backend weight sets come from this project.
- **[DeepSeek-AI](https://github.com/deepseek-ai/DeepSeek-OCR)** — reference for OCR accuracy optimization approaches.

If these projects helped your own work, please consider supporting them.

## License

Code is Apache-2.0. Weights originate from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) and are used under its original terms.
