<img src="docs/images/title.jpg" alt="NewOnnxOCR" width="100%">

# NOOCR

[简体中文](README.md) | [English](README_EN.md) | [繁體中文](README_TW.md)

ONNX 全功能 OCR 系統。一個核心，多檔後端，CPU / GPU 雙模。

```bash
pip install -r requirements.txt              # 一條命令，CPU 與 GPU 通用
python -m noocr models --get ppocrv6-tiny    # 拉權重（約 6MB）
python -m noocr 發票.jpg                      # 辨識
python -m noocr serve                         # Web 介面 http://127.0.0.1:8000
```

有 NVIDIA 顯示卡時加 `-d cuda`（或 `serve --device cuda`），端到端快 **20-40 倍**。

遇到問題先看 [Q&A.md](Q&A.md)。

## 特性

| | |
|---|---|
| **一份依賴** | CPU 與 NVIDIA 機器裝同一個 `requirements.txt`，無需重建環境 |
| **CPU / GPU 雙模** | 同一份程式碼，`--device cpu` 或 `--device cuda`，預設自動探測 |
| **四檔後端** | PP-OCRv6 tiny（6.6MB）/ small（30.3MB，預設）/ medium（132.8MB）/ v5（21.1MB） |
| **多格式輸入** | 圖片、PDF、Word、Excel、PPT、URL |
| **結果可重現** | 辨識輸出與 `rec_batch_size` 無關 |
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

無 NVIDIA 顯示卡的機器裝同一份依賴也能正常安裝執行，只是 GPU 部分永不啟用。若只跑 CPU 且在意安裝體積，可換輕量版（辨識結果完全一致）：

```bash
pip install -r requirements-cpu.txt     # 約 15MB
```

### GPU 需要額外的系統函式庫

CUDA 與 cuDNN 是系統層執行庫，pip 裝不了，須分別安裝：NVIDIA 驅動程式（≥ 525）、CUDA 12.x、**cuDNN 9**。

裝完告訴專案 DLL 在哪（Windows 必填，Linux/macOS 一般不需要）：

```bash
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin            # Windows
export NOOCR_GPU_LIB_DIR=/usr/local/cudnn/lib       # Linux
```

專案也會自動在專案根目錄、虛擬環境的 `site-packages/cudnn/` 等常見位置尋找。仍無法啟用時的排查見 [Q&A.md](Q&A.md#gpu-未生效)。

## 取得權重

全部 19 個權重（223.6MB）託管在 ModelScope：**[iterhui/noocr-onnx](https://www.modelscope.cn/models/iterhui/noocr-onnx)**，按後端按需下載：

```bash
python -m noocr models                     # 查看各後端狀態
python -m noocr models --get ppocrv6-tiny   # 6.6MB，最快
python -m noocr models --get ppocrv6-small  # 30.3MB，預設
python -m noocr models --get ppocrv6-medium # 132.8MB，伺服器檔，精度最高
python -m noocr models --get ppocrv5        # 21.1MB
```

或一次拉全部：

```bash
pip install modelscope
modelscope download --model iterhui/noocr-onnx --local_dir ./models
```

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
   │  │  └─ cls.onnx                    # 方向分類器（180° 糾正）
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

手動放置時，缺哪個檔案 `python -m noocr models` 會用 `缺失` 標出。想放到別處，設 `NOOCR_MODELS_DIR`。

## 命令列

```bash
noocr <檔案>                     # 印出文字
noocr <檔案> -o out.json        # 結構化 JSON（含座標與信賴度）
noocr 論文.pdf -o 論文.md -f md  # PDF 轉 Markdown
noocr bench 試卷.jpg             # 分階段耗時剖析
noocr backends                   # 列出後端
noocr models                     # 權重狀態
noocr serve --port 8000          # Web 介面 + API
```

常用選項：`-b/--backend` 選後端、`-f/--format` 選 `json|text|md`、`-d/--device` 選 `auto|cpu|cuda`、`--batch` 調批次大小、`--no-cls` 關閉 180° 糾正。

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

## Web 介面與 API

```bash
noocr serve --host 0.0.0.0 --port 8000 --device cuda
```

介面 `http://127.0.0.1:8000/` —— 左側上傳與參數、右側圖文對照雙欄連動、下方辨識記錄與範例圖。頂欄裝置徽標可直接點擊切換 CPU / GPU，無需重啟。API 文件 `http://127.0.0.1:8000/docs`。

<details open>
<summary><b>介面圖例</b>（點擊展開）</summary>

**圖文對照雙欄** —— 左側上傳與參數，右側原圖帶識別框與識別文字雙欄連動，下方為辨識記錄與範例圖。頂欄裝置徽標顯示目前實際生效的裝置。

![Web 介面：圖文對照雙欄](docs/images/demo-overview.jpg)

**明細連動** —— 點明細表任一行，圖上對應文字框高亮為藍色，其餘保持綠色。

![Web 介面：點明細行圖上高亮連動](docs/images/webui-detail.jpg)

</details>

| 方法 | 路徑 | 說明 |
|---|---|---|
| `GET` | `/health` | 健康檢查 |
| `GET` | `/api/backends` | 可用後端列表 |
| `GET` | `/api/device` | 裝置偏好與**實際生效**的裝置 |
| `POST` | `/api/device` | 切換裝置（`auto`/`cpu`/`cuda`） |
| `POST` | `/api/ocr` | 上傳檔案辨識（multipart） |
| `POST` | `/api/ocr/path?path=...` | 辨識伺服器本機路徑 |
| `GET` | `/api/page/{doc_id}/{i}` | 取第 i 頁算繪圖（多頁翻頁用） |
| `POST` | `/api/warmup` | 預載入後端 |

```bash
curl -X POST http://127.0.0.1:8000/api/ocr \
  -F "file=@發票.jpg" -F "backend=ppocrv6-tiny"
```

## 效能

RTX 4070 Ti SUPER + `ppocrv6-small`，與效能對比表同一批 8 張圖、含圖片解碼，每檔獨立行程、1 輪預熱後取 3 輪均值：

| 圖片 | 行數 | CPU | GPU | 加速比 |
|---|---|---|---|---|
| 對比表格 | 73 | 9139 ms | **485 ms** | 19x |
| 小學試卷 | 65 | 2000 ms | **759 ms** | 2.6x |
| 身分證 | 11 | 1649 ms | **266 ms** | 6.2x |
| 化驗單 | 69 | 1322 ms | **379 ms** | 3.5x |
| 規格書 | 16 | 1750 ms | **267 ms** | 6.6x |
| 銀行流水 | 31 | 1591 ms | **355 ms** | 4.5x |
| 豎式牌匾 | 2 | 1317 ms | **98 ms** | 13.5x |
| 火車票 | 19 | 1672 ms | **301 ms** | 5.6x |
| **合計** | — | **20440 ms** | **2909 ms** | **7.0x** |

加速比與「行數多少」關係不大，主要看版面複雜度——`對比表格` 是密集小字，CPU 上光後處理就佔了 9 秒；`豎式牌匾` 只有 2 行但圖大，GPU 上只要 98ms。CPU 上密集版面反而更吃虧。

GPU 側首次請求含約 2.4s 的模型載入與 CUDA kernel 編譯，之後穩定在 0.1-0.8s。重現：

```bash
python scripts/perf/bench_device.py cpu 3   # CPU 基線
python scripts/perf/bench_device.py cuda 3  # GPU 基線
```

兩個 device 要分開跑——同一行程裡連續跑會讓 CPU 執行緒設定互相干擾，測出來的加速比沒有意義。

### 與 OnnxOCR 實測對比

**四個檔位、兩側跑同一份權重檔案**（det / rec / 字典經 MD5 校驗逐位元組一致）、同一塊 RTX 4070 Ti SUPER、同為 ONNX Runtime 1.23.2，兩側都開啟方向分類、含圖片解碼，每圖 1 輪預熱後取 10 輪均值：

<details open>
<summary><b>PP-OCRv5</b>（点击展开）</summary>

| 範例圖 | 解析度 | OnnxOCR<br>PP-OCRv5 | NOOCR<br>PP-OCRv5 | 倍數 | 文字行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 3691 ms | **453 ms** | **8.2x** | 82 / 79 |
| `exam_chinese_primary` | 1920x2560 | 3259 ms | **1062 ms** | **3.1x** | 68 / 73 |
| `id_card_china` | 1148x672 | 551 ms | **205 ms** | **2.7x** | 9 / 10 |
| `medical_lab_report` | 430x267 | 2512 ms | **309 ms** | **8.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 778 ms | **213 ms** | **3.6x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1376 ms | **287 ms** | **4.8x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 303 ms | **111 ms** | **2.7x** | 2 / 2 |
| `ticket_train` | 670x510 | 961 ms | **522 ms** | **1.8x** | 19 / 17 |
| **均值** | — | **1679 ms** | **395 ms** | **4.25x** | 296 / 296 |
| **合計** | — | **13432 ms** | **3162 ms** | **4.25x** | — |

</details>
<details>
<summary><b>PP-OCRv6 tiny</b>（点击展开）</summary>

| 範例圖 | 解析度 | OnnxOCR<br>PP-OCRv6 tiny | NOOCR<br>PP-OCRv6 tiny | 倍數 | 文字行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 1838 ms | **311 ms** | **5.9x** | 73 / 74 |
| `exam_chinese_primary` | 1920x2560 | 2210 ms | **477 ms** | **4.6x** | 70 / 75 |
| `id_card_china` | 1148x672 | 374 ms | **171 ms** | **2.2x** | 9 / 9 |
| `medical_lab_report` | 430x267 | 1729 ms | **264 ms** | **6.5x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 527 ms | **146 ms** | **3.6x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 932 ms | **257 ms** | **3.6x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 191 ms | **64 ms** | **3.0x** | 2 / 2 |
| `ticket_train` | 670x510 | 631 ms | **188 ms** | **3.4x** | 19 / 20 |
| **均值** | — | **1054 ms** | **235 ms** | **4.49x** | 289 / 295 |
| **合計** | — | **8433 ms** | **1878 ms** | **4.49x** | — |

</details>
<details>
<summary><b>PP-OCRv6 small</b>（点击展开）</summary>

| 範例圖 | 解析度 | OnnxOCR<br>PP-OCRv6 small | NOOCR<br>PP-OCRv6 small | 倍數 | 文字行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 2786 ms | **457 ms** | **6.1x** | 71 / 73 |
| `exam_chinese_primary` | 1920x2560 | 2964 ms | **921 ms** | **3.2x** | 71 / 65 |
| `id_card_china` | 1148x672 | 555 ms | **257 ms** | **2.2x** | 10 / 11 |
| `medical_lab_report` | 430x267 | 2501 ms | **363 ms** | **6.9x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 768 ms | **266 ms** | **2.9x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1345 ms | **352 ms** | **3.8x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 308 ms | **100 ms** | **3.1x** | 2 / 2 |
| `ticket_train` | 670x510 | 949 ms | **300 ms** | **3.2x** | 19 / 19 |
| **均值** | — | **1522 ms** | **377 ms** | **4.04x** | 289 / 285 |
| **合計** | — | **12177 ms** | **3016 ms** | **4.04x** | — |

</details>
<details>
<summary><b>PP-OCRv6 medium</b>（点击展开）</summary>

| 範例圖 | 解析度 | OnnxOCR<br>PP-OCRv6 medium | NOOCR<br>PP-OCRv6 medium | 倍數 | 文字行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 4535 ms | **478 ms** | **9.5x** | 74 / 73 |
| `exam_chinese_primary` | 1920x2560 | 5175 ms | **729 ms** | **7.1x** | 64 / 66 |
| `id_card_china` | 1148x672 | 1003 ms | **245 ms** | **4.1x** | 12 / 11 |
| `medical_lab_report` | 430x267 | 4202 ms | **462 ms** | **9.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 1326 ms | **299 ms** | **4.4x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 2304 ms | **417 ms** | **5.5x** | 32 / 32 |
| `scene_vertical_plaque` | 720x1150 | 634 ms | **115 ms** | **5.5x** | 2 / 2 |
| `ticket_train` | 670x510 | 1669 ms | **366 ms** | **4.6x** | 19 / 19 |
| **均值** | — | **2606 ms** | **389 ms** | **6.70x** | 288 / 288 |
| **合計** | — | **20847 ms** | **3112 ms** | **6.70x** | — |

</details>

**模型越大，優勢越明顯。** v5 上快 4.25 倍，v6 medium 上快 6.70 倍——因為 OnnxOCR 在大模型上沒有對輸入解析度與記憶體 arena 做取捨，全部按最大設定跑；NOOCR 依形狀特徵選解析度並關閉 arena 重規劃，模型越大這部分省得越多。

四檔的文字行數都與對方基本一致（v5 296/296、medium 288/288），說明預處理口徑對等，差距來自實作而非「誰認得更多字」。差距主要來自三處：det 的輸入解析度與記憶體 arena 策略（形狀集合有限離散，關掉 `dynamic_shape` 讓 arena 生效後提速約 32%）、rec 的字典裁剪，以及工作階段重用。

復現：`scripts/perf/compare_onnxocr.md`（需本機另備一份 OnnxOCR 檢出與相依環境）。

## 後端對比

| 後端 | 體積 | 相對速度 | 加權信賴度 | 適用 |
|---|---|---|---|---|
| `ppocrv6-tiny` | 6.6MB | **1.95x** | 0.956 | 邊緣裝置、批次 |
| `ppocrv6-medium` | 132.8MB | 1.11x | **0.981** | 版面複雜、追求精度 |
| `ppocrv6-small` | 30.3MB | 1.03x | 0.973 | 預設 |
| `ppocrv5` | 21.1MB | 1.00x | 0.928 | 相容舊專案 |

RTX 4070 Ti SUPER、8 張範例圖、1 輪預熱後 5 輪取中位，相對速度以 `ppocrv5` 為基準；加權信賴度按字數加權（`scripts/perf/bench_tiers.py`，每檔獨立行程測量）。

四個 v6 檔位的信賴度都高於 v5，且**medium 檔比 small 還快**——雖然參數較多，但 det 解析度策略讓耗時沒上去，反而比 small 低 7%。

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
│  ├─ postprocess.py       DB 偵測後處理
│  └─ decode.py            CTC 解碼
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
│  ├─ bench_device.py      CPU / GPU 基線
│  ├─ bench_tiers.py       各檔位相對速度與信賴度
│  ├─ compare_onnxocr.md   與 OnnxOCR 對比的復現說明
│  └─ ab_*.py              arena / cuDNN / 檔位 A/B
├─ shrink_images.py        截圖壓縮
└─ models_repo_card.md     權重倉庫模型卡
```

## 致謝

本專案在建構過程中受益於以下開源專案，感謝原作者與社群：

- **[OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)** —— ONNX 推論鏈路、模型匯出與文件結構化思路，本專案的起點。
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** —— PP-OCR 系列模型與演算法設計，本專案四檔後端的權重皆來自此專案。
- **[DeepSeek-AI](https://github.com/deepseek-ai/DeepSeek-OCR)** —— OCR 精度最佳化的思路參考。

若上述專案的成果對你的工作有所幫助，請優先支持它們。

## 授權

程式碼 Apache-2.0。權重源自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0），依原授權條款使用。
