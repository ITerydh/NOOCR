# NOOCR

[简体中文](README.md) | [English](README_EN.md) | [繁體中文](README_TW.md)

基於 ONNX 的全功能 OCR 系統。一個核心，多檔後端，CPU / GPU 雙模。

```bash
pip install -r requirements.txt      # 一條命令，CPU 與 GPU 通用
python -m noocr models --get ppocrv6-tiny   # 拉權重（約 6MB）
python -m noocr 發票.jpg                     # 識別
python -m noocr serve                        # Web 介面 http://127.0.0.1:8000
```

有 NVIDIA 顯卡時加上 `-d cuda`（或 `serve --device cuda`），端到端快 **20-40 倍**。

---

## 特性

| | |
|---|---|
| **一份依賴** | CPU 與 NVIDIA 機器裝同一個 `requirements.txt`，無需重建環境 |
| **CPU / GPU 雙模** | 同一份程式碼，`--device cpu` 或 `--device cuda`，可自動探測 |
| **三檔後端** | PP-OCRv6 tiny（6MB）/ small（32MB，預設）/ v5（22MB） |
| **多格式輸入** | 圖片、PDF、Word、Excel、PPT、URL |
| **結果可重現** | 識別輸出與 `rec_batch_size` 無關 |
| **四種交付** | 函式庫 / CLI / REST API / Web 介面 |
| **統一契約** | 任何後端都回傳 `OCRResult`，呼叫端無需分支 |

## 安裝

要求 Python ≥ 3.10。

```bash
git clone https://github.com/ITerydh/NOOCR.git
cd noocr
python -m venv .venv && . .venv/Scripts/activate    # Windows
# python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### 為什麼只有一份依賴

`requirements.txt` 裝的是 `onnxruntime-gpu` 而非 `onnxruntime`，因為前者是後者的**超集**——它同時含 CPU、CUDA、TensorRT 三個 EP，裝一個套件就覆蓋了兩種機器：

| 套件 | 可用 EP | 體積 |
|---|---|---|
| `onnxruntime` | CPU | ~15 MB |
| `onnxruntime-gpu` | CPU / CUDA / TensorRT | ~700 MB |

沒有 NVIDIA 顯卡的機器裝 GPU 版**照樣能正常安裝與執行**，只是 GPU 部分永不啟用。所以不需要區分「CPU 環境」和「GPU 環境」，也不需要為 GPU 重建虛擬環境。

代價是體積。若只跑 CPU 且在意安裝速度或磁碟空間，換一份輕量的：

```bash
pip install -r requirements-cpu.txt     # 約 15MB 的 ORT，識別結果完全一致
```

哪天要開 GPU，改為執行 `pip install -r requirements.txt` 即可（GPU 版會覆蓋 CPU 版）。

### 系統層依賴（僅 GPU 需要）

pip 只管 Python 套件，CUDA 與 cuDNN 是**系統層執行庫**，須單獨安裝：

1. NVIDIA 驅動程式（≥ 525）
2. CUDA 12.x
3. **cuDNN 9**（注意是 9.x，ORT 1.20+ 依賴 `cudnn64_9.dll`）

裝完告訴專案 DLL 在哪（Windows 必填，Linux/macOS 一般不需要）：

```bash
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin            # Windows
export NOOCR_GPU_LIB_DIR=/usr/local/cudnn/lib       # Linux
```

專案也會自動在專案根目錄、虛擬環境的 `site-packages/cudnn/` 等常見位置尋找。

## GPU 加速

GPU 上端到端比 CPU 快 **20-40 倍**（見下方[效能](#效能)）。依賴已在 `requirements.txt` 裡，無需額外安裝 Python 套件；只要系統層裝好 CUDA 12 + cuDNN 9 即可：

```bash
python -m noocr 發票.jpg -d cuda
python -m noocr serve --device cuda
```

`--device` 可選 `auto`（預設，有 CUDA 就用）/ `cpu` / `cuda`。

### 在 Web 介面裡隨時切換

`serve` 啟動後，**頂欄裝置徽標可直接點擊切換**，無需重啟服務：

- 徽標顯示的是**實際生效**的裝置（GPU 生效時顯示顯卡名稱並變綠），不是啟動時的請求值；
- 下拉裡有 `自動 / 顯示卡 / 處理器` 三項，本機無 GPU 環境時「顯示卡」會置灰並寫明原因；
- 切換時自動卸載另一套模型並歸還顯示記憶體，代價約 0.4-0.6s；切換後若已有識別結果會自動重跑，方便直接比較速度；
- 目標裝置不可用時回傳 400 並**保持原裝置不變**，不會把能跑的服務弄成不能跑。

也可以走 API：

```bash
curl http://127.0.0.1:8000/api/device                       # 目前裝置
curl -X POST -F "device=cuda" .../api/device              # 切到 GPU
```

### cuDNN 9 是必要的

ONNX Runtime 1.20+ 的 CUDA EP 依賴 **cuDNN 9**（`cudnn64_9.dll`）。裝錯版本的表現非常隱蔽：ORT **不報錯**，只是把運算子悄悄交給 CPU，你會得到純 CPU 的效能卻以為在用顯示卡。

本專案對此做了三重防護：

1. 啟動時用微型 ONNX 模型**實測** CUDA EP 能否初始化，而非只看它是否被編譯進來；
2. session 建立後**核對實際生效的 EP**，與請求不符直接報錯；
3. Web 介面頂欄常駐**裝置徽標**，GPU 未生效時顯示為 CPU。

cuDNN 9 安裝（解壓後把 `bin` 下的 DLL 放到任一目錄並告知專案）：

```bash
# 從 https://developer.nvidia.com/cudnn-downloads 下載 cuDNN 9 for CUDA 12
# Windows 需顯式登記 DLL 搜尋路徑，Linux/macOS 直接給權限即可
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin
```

專案會自動在專案根目錄、`noocr-gpu/` 等同級虛擬環境的 `site-packages/cudnn/` 下尋找，無需手動設定。

> **若 GPU 環境與 CPU 環境分開建**（建議，避免 ORT 的 DLL 互相覆蓋），把 cuDNN 的 DLL 放到 GPU 環境裡：
> `<gpu-venv>/Lib/site-packages/cudnn/`。

### 一個值得記住的坑

`cudnn_conv_algo_search` **必須設在 provider 層選項裡**，寫成 `"DEFAULT"` 會讓 PP-OCRv6-rec 的 228 個卷積集體退回 CPU 實作，單次推論從 5.9ms 劣化到 90.2ms（15 倍）。且 provider 層設定**優先級高於** SessionOptions，寫錯地方會被靜默覆蓋。本專案已在 `build_providers()` 裡固定為 `"EXHAUSTIVE"`。

## 取得權重

全部 17 個權重（95.8MB）託管在 ModelScope：**[iterhui/noocr-onnx](https://www.modelscope.cn/models/iterhui/noocr-onnx)**，按後端按需下載：

```bash
python -m noocr models                     # 查看各後端狀態
python -m noocr models --get ppocrv6-tiny  # 6.9MB，最快
python -m noocr models --get ppocrv6-small # 32MB，預設
python -m noocr models --get ppocrv5       # 22MB，上一代
```

也可SDK 一次性拉全部：

```bash
pip install modelscope
modelscope download --model iterhui/noocr-onnx --local_dir ./models
```

或直接下載單一檔案（路徑與下表一致）：

```
https://www.modelscope.cn/models/iterhui/noocr-onnx/resolve/master/ppocrv6/det/PP-OCRv6_det_small.onnx
```

### 放置位置

下載後目錄結構必須如下（`models.py` 依此路徑解析）：

```
<專案根目錄>/
├─ noocr/
└─ models/
   ├─ ppocrv6/
   │  ├─ det/
   │  │  ├─ PP-OCRv6_det_tiny.onnx      # tiny 檔
   │  │  └─ PP-OCRv6_det_small.onnx     # small 檔
   │  ├─ rec/
   │  │  ├─ PP-OCRv6_rec_tiny.onnx
   │  │  └─ PP-OCRv6_rec_small.onnx
   │  ├─ cls/
   │  │  └─ cls.onnx                    # 方向分類器（180° 修正）
   │  ├─ ppocrv6_tiny_dict.txt          # tiny 字典（6906 類）
   │  └─ ppocrv6_dict.txt               # small 字典（18710 類）
   └─ ppocrv5/
      ├─ det/det.onnx
      ├─ rec/rec.onnx
      ├─ cls/cls.onnx
      └─ ppocrv5_dict.txt               # 6623 類
```

可選後端（`--get plate` / `orientation` / `layout` / `table`）對應：

```
models/license_plate/{car_plate_detect.onnx, plate_rec.onnx}
models/orientation/rapid_orientation.onnx
models/layout/{layout_cdla.onnx, layout_publaynet.onnx}
models/table/slanet-plus.onnx
```

手動放置時，缺哪個檔案 `python -m noocr models` 會用 `缺失` 標出。

想放到別處，設環境變數：

```bash
export NOOCR_MODELS_DIR=/data/ocr/models    # Linux/macOS
set NOOCR_MODELS_DIR=D:\ocr\models          # Windows
```

## 命令列

```bash
noocr <檔案>                     # 印出文字
noocr <檔案> -o out.json        # 結構化 JSON（含座標與信賴度）
noocr 論文.pdf -o 論文.md -f md  # PDF 轉 Markdown
noocr bench 考卷.jpg             # 分階段耗時剖析
noocr backends                   # 列出後端
noocr models                     # 權重狀態
noocr serve --port 8000          # Web 介面 + API
```

常用選項：`-b/--backend` 選後端、`-f/--format` 選 `json|text|md`、`-d/--device` 選 `auto|cpu|cuda`、`--batch` 調整批次大小、`--no-cls` 關閉 180° 修正。

## 當作函式庫使用

```python
from noocr import ocr

result = ocr("掃描件.jpg")
print(result.text)
print(result.to_markdown())

for line in result.all_lines:
    print(f"{line.confidence:.2f}  {line.text}  {line.box.points}")
```

指定後端與參數：

```python
# 按次覆寫：這一次用 CPU / 換後端，不影響其他呼叫
r = ocr("掃描件.jpg", device="cpu")        # 強制 CPU
r = ocr("掃描件.jpg", backend="ppocrv5")   # 暫時換後端
r = ocr("掃描件.jpg", dpi=300, max_pages=5)

# 固定設定：長駐服務應該自己持有一個實例，模型只載入一次
from noocr import OCRPipeline

pipe = OCRPipeline(backend="ppocrv6-tiny", device="cuda", rec_batch_size=8)
for path in paths:
    print(pipe(path).text)
```

也可以直接拿後端：

```python
from noocr.backends import get_backend

backend = get_backend("ppocrv6-tiny", rec_batch_size=8, device="cuda")
backend.load()
page = backend.recognize_image(image)      # image 為 BGR ndarray
print(page.debug["stage_ms"])              # 分階段耗時，方便定位慢在哪一步
```

## Web 介面與 API

```bash
noocr serve --host 0.0.0.0 --port 8000 --device cuda
```

介面 `http://127.0.0.1:8000/` —— 拖入檔案即識別，文字 / Markdown / 圖文對照三檢視，圖文與明細雙欄連動。
API 文件 `http://127.0.0.1:8000/docs`。

| 方法 | 路徑 | 說明 |
|---|---|---|
| `GET` | `/health` | 健康檢查 |
| `GET` | `/api/backends` | 可用後端列表 |
| `GET` | `/api/device` | 裝置偏好與**實際生效**的裝置 |
| `POST` | `/api/ocr` | 上傳檔案識別（multipart） |
| `POST` | `/api/ocr/path?path=...` | 識別伺服器本機路徑 |
| `GET` | `/api/page/{doc_id}/{i}` | 取第 i 頁渲染圖（多頁翻頁用） |
| `POST` | `/api/warmup` | 預載入後端 |

```bash
curl -X POST http://127.0.0.1:8000/api/ocr \
  -F "file=@發票.jpg" -F "backend=ppocrv6-tiny"
```

## 效能

RTX 4070 Ti SUPER + ppocrv6-small，同一批範例圖取中位數：

| 圖片 | 行數 | CPU | GPU | 加速比 |
|---|---|---|---|---|
| 票據 | 19 | 15537 ms | **594 ms** | 26x |
| 銀行流水 | 30 | 13040 ms | **577 ms** | 23x |
| 化驗單 | 69 | 10396 ms | **567 ms** | 18x |
| 身分證 | 11 | 13317 ms | **400 ms** | 33x |
| 銀行網點 | 4 | 9414 ms | **249 ms** | 38x |

GPU 側首次請求含約 2.4s 的模型載入與 CUDA kernel 編譯，之後穩定在 0.25-0.6s。

重現方式：

```bash
python scripts/perf/bench_device.py cpu     # CPU 基線
python scripts/perf/bench_device.py cuda    # GPU 基線
```

其他工具：

```bash
python scripts/perf/bench_buckets.py cuda   # 分桶與串行呼叫次數
python scripts/perf/ab_tiers.py cuda        # 檔位數 A/B（含輸出一致性校驗）
python scripts/perf/ab_cudnn.py             # cuDNN 演算法搜尋 A/B
python scripts/perf/prof_rec.py cuda        # ORT profiler 逐運算子耗時
```

## 後端對比

| 後端 | 體積 | 相對速度 | 加權信賴度 | 適用 |
|---|---|---|---|---|
| `ppocrv6-tiny` | 6.1MB | **0.58x** | 0.961 | 邊緣裝置、批次 |
| `ppocrv6-small` | 30.5MB | 1.73x | **0.971** | 預設 |
| `ppocrv5` | 15.6MB | 1.00x | 0.936 | 相容舊專案 |

## 專案結構

```
noocr/
├─ types.py              統一資料模型（Pydantic）
├─ models.py             權重清單與下載
├─ cli.py                命令列
├─ engine/               推論引擎
│  ├─ session.py           session 快取、裝置探測、GPU 執行庫掛載
│  ├─ imageops.py          影像幾何與預處理
│  └─ base.py              後端抽象
├─ backends/             OCR 後端
│  ├─ ppocr.py             PP-OCRv5
│  ├─ ppocrv6.py           PP-OCRv6
│  ├─ postprocess.py       DB 檢測後處理
│  └─ decode.pyCTC 解碼
├─ inputs/loader.py      多格式輸入
├─ document/             文件結構化
├─ pipeline/             編排
└─ web/                  Web 介面與 API
    ├─ app.py
    └─ templates/index.html
tests/                   測試與基準
scripts/
├─ publish_weights.py      權重發布腳本
├─ perf/                   效能基準與 A/B 腳本
└─ models_repo_card.md     權重倉庫模型卡
```

## 授權

程式碼 Apache-2.0。權重源自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0），依原授權條款使用。