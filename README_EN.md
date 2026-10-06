<img src="docs/images/title.jpg" alt="NewOnnxOCR" width="100%">

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
| **Four backends** | PP-OCRv6 tiny (6.6MB) / small (30.3MB, default) / medium (132.8MB) / v5 (21.1MB) |
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

All 19 files (223.6MB) are hosted on ModelScope: **[iterhui/noocr-onnx](https://www.modelscope.cn/models/iterhui/noocr-onnx)**. Fetch per backend:

```bash
python -m noocr models                      # show status per backend
python -m noocr models --get ppocrv6-tiny   # 6.6MB, fastest
python -m noocr models --get ppocrv6-small  # 30.3MB, default
python -m noocr models --get ppocrv6-medium # 132.8MB, server tier, most accurate
python -m noocr models --get ppocrv5        # 21.1MB
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

RTX 4070 Ti SUPER + `ppocrv6-small`, the same 8 sample images as the comparison table above, image decoding included, each device in its own process, 1 warm-up round then the mean of 3 rounds:

| Image | Lines | CPU | GPU | Speedup |
|---|---|---|---|---|
| Comparison table | 73 | 9139 ms | **485 ms** | 19x |
| Primary school exam | 65 | 2000 ms | **759 ms** | 2.6x |
| ID card | 11 | 1649 ms | **266 ms** | 6.2x |
| Lab report | 69 | 1322 ms | **379 ms** | 3.5x |
| Product spec | 16 | 1750 ms | **267 ms** | 6.6x |
| Bank statement | 31 | 1591 ms | **355 ms** | 4.5x |
| Vertical plaque | 2 | 1317 ms | **98 ms** | 13.5x |
| Train ticket | 19 | 1672 ms | **301 ms** | 5.6x |
| **Total** | — | **20440 ms** | **2909 ms** | **7.0x** |

The speedup has little to do with line count — layout complexity dominates. `Comparison table` is dense small text and post-processing alone takes 9 seconds on CPU; `Vertical plaque` has only 2 lines but a large canvas, and needs just 98ms on GPU. Dense layouts actually suffer more on CPU.

The first GPU request includes about 2.4s of model loading and CUDA kernel compilation; afterwards it stabilizes at 0.1-0.8s. Reproduce with:

```bash
python scripts/perf/bench_device.py cpu 3   # CPU baseline
python scripts/perf/bench_device.py cuda 3  # GPU baseline
```

Run the two devices in separate processes — measuring both in one process makes the CPU thread configuration interfere with itself, and the resulting speedup means nothing.

### Measured against OnnxOCR

**Four tiers, both engines reading the very same weight files** (det / rec / dictionary verified byte-identical by MD5), same RTX 4070 Ti SUPER, both on ONNX Runtime 1.23.2, direction classification enabled on both sides, image decoding included, 1 warm-up round then the mean of 10 rounds per image:

<details open>
<summary><b>PP-OCRv5</b>（点击展开）</summary>

| Sample | Resolution | OnnxOCR<br>PP-OCRv5 | NOOCR<br>PP-OCRv5 | Ratio | Lines |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 3691 ms | **453 ms** | **8.2x** | 82 / 79 |
| `exam_chinese_primary` | 1920x2560 | 3259 ms | **1062 ms** | **3.1x** | 68 / 73 |
| `id_card_china` | 1148x672 | 551 ms | **205 ms** | **2.7x** | 9 / 10 |
| `medical_lab_report` | 430x267 | 2512 ms | **309 ms** | **8.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 778 ms | **213 ms** | **3.6x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1376 ms | **287 ms** | **4.8x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 303 ms | **111 ms** | **2.7x** | 2 / 2 |
| `ticket_train` | 670x510 | 961 ms | **522 ms** | **1.8x** | 19 / 17 |
| **Mean** | — | **1679 ms** | **395 ms** | **4.25x** | 296 / 296 |
| **Total** | — | **13432 ms** | **3162 ms** | **4.25x** | — |

</details>
<details>
<summary><b>PP-OCRv6 tiny</b>（点击展开）</summary>

| Sample | Resolution | OnnxOCR<br>PP-OCRv6 tiny | NOOCR<br>PP-OCRv6 tiny | Ratio | Lines |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 1838 ms | **311 ms** | **5.9x** | 73 / 74 |
| `exam_chinese_primary` | 1920x2560 | 2210 ms | **477 ms** | **4.6x** | 70 / 75 |
| `id_card_china` | 1148x672 | 374 ms | **171 ms** | **2.2x** | 9 / 9 |
| `medical_lab_report` | 430x267 | 1729 ms | **264 ms** | **6.5x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 527 ms | **146 ms** | **3.6x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 932 ms | **257 ms** | **3.6x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 191 ms | **64 ms** | **3.0x** | 2 / 2 |
| `ticket_train` | 670x510 | 631 ms | **188 ms** | **3.4x** | 19 / 20 |
| **Mean** | — | **1054 ms** | **235 ms** | **4.49x** | 289 / 295 |
| **Total** | — | **8433 ms** | **1878 ms** | **4.49x** | — |

</details>
<details>
<summary><b>PP-OCRv6 small</b>（点击展开）</summary>

| Sample | Resolution | OnnxOCR<br>PP-OCRv6 small | NOOCR<br>PP-OCRv6 small | Ratio | Lines |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 2786 ms | **457 ms** | **6.1x** | 71 / 73 |
| `exam_chinese_primary` | 1920x2560 | 2964 ms | **921 ms** | **3.2x** | 71 / 65 |
| `id_card_china` | 1148x672 | 555 ms | **257 ms** | **2.2x** | 10 / 11 |
| `medical_lab_report` | 430x267 | 2501 ms | **363 ms** | **6.9x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 768 ms | **266 ms** | **2.9x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1345 ms | **352 ms** | **3.8x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 308 ms | **100 ms** | **3.1x** | 2 / 2 |
| `ticket_train` | 670x510 | 949 ms | **300 ms** | **3.2x** | 19 / 19 |
| **Mean** | — | **1522 ms** | **377 ms** | **4.04x** | 289 / 285 |
| **Total** | — | **12177 ms** | **3016 ms** | **4.04x** | — |

</details>
<details>
<summary><b>PP-OCRv6 medium</b>（点击展开）</summary>

| Sample | Resolution | OnnxOCR<br>PP-OCRv6 medium | NOOCR<br>PP-OCRv6 medium | Ratio | Lines |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 4535 ms | **478 ms** | **9.5x** | 74 / 73 |
| `exam_chinese_primary` | 1920x2560 | 5175 ms | **729 ms** | **7.1x** | 64 / 66 |
| `id_card_china` | 1148x672 | 1003 ms | **245 ms** | **4.1x** | 12 / 11 |
| `medical_lab_report` | 430x267 | 4202 ms | **462 ms** | **9.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 1326 ms | **299 ms** | **4.4x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 2304 ms | **417 ms** | **5.5x** | 32 / 32 |
| `scene_vertical_plaque` | 720x1150 | 634 ms | **115 ms** | **5.5x** | 2 / 2 |
| `ticket_train` | 670x510 | 1669 ms | **366 ms** | **4.6x** | 19 / 19 |
| **Mean** | — | **2606 ms** | **389 ms** | **6.70x** | 288 / 288 |
| **Total** | — | **20847 ms** | **3112 ms** | **6.70x** | — |

</details>

**The larger the model, the larger the gap.** 4.25x on v5, 6.70x on v6 medium — OnnxOCR makes no trade-off on input resolution and memory arena for big models and simply runs everything at maximum configuration; NOOCR picks resolution by shape characteristics and turns off arena replanning, so the bigger the model the more that saves.

Line counts match closely across all four tiers (v5 296/296, medium 288/288), so preprocessing is on equal footing and the gap comes from the implementation, not from "who reads more words". It comes mainly from three places: det input resolution and memory arena policy (the set of shapes is finite and discrete, so disabling `dynamic_shape` lets the arena work and speeds up det by ~32%), rec dictionary trimming, and session reuse.

Reproduce with `scripts/perf/compare_onnxocr.md` (requires a separate OnnxOCR checkout and dependency environment).

## Backend comparison

| Backend | Size | Relative speed | Weighted confidence | Use for |
|---|---|---|---|---|
| `ppocrv6-tiny` | 6.6MB | **1.95x** | 0.956 | Edge devices, batch work |
| `ppocrv6-medium` | 132.8MB | 1.11x | **0.981** | Complex layouts, best accuracy |
| `ppocrv6-small` | 30.3MB | 1.03x | 0.973 | Default |
| `ppocrv5` | 21.1MB | 1.00x | 0.928 | Legacy compatibility |

RTX 4070 Ti SUPER, 8 sample images, 1 warm-up round then median of 5, relative speed based on `ppocrv5`; confidence weighted by character count (`scripts/perf/bench_tiers.py`, each tier measured in its own process).

All four v6 tiers beat v5 on confidence, and the **medium tier is even faster than small** — despite more parameters, its det resolution strategy keeps the time down, 7% below small.

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
│  ├─ bench_device.py      CPU / GPU baseline
│  ├─ bench_tiers.py       per-tier relative speed and confidence
│  ├─ compare_onnxocr.md   how to reproduce the OnnxOCR comparison
│  └─ ab_*.py              arena / cuDNN / tier A/B
├─ shrink_images.py        screenshot compression
└─ models_repo_card.md     model card for the weight repo
```

## Acknowledgements

This project is built on the work of the following open-source projects. Thanks to their authors and communities:

- **[OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)** — the ONNX inference path, model export and document structuring ideas that became the starting point of this project.
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** — the PP-OCR model family and algorithm design; all four backend weight sets come from this project.
- **[DeepSeek-AI](https://github.com/deepseek-ai/DeepSeek-OCR)** — reference for OCR accuracy optimization approaches.

If these projects helped your own work, please consider supporting them.

## License

Code is Apache-2.0. Weights originate from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) and are used under its original terms.
