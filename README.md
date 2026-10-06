<img src="docs/images/title.jpg" alt="NewOnnxOCR" width="100%">

# NOOCR

[简体中文](README.md) | [English](README_EN.md) | [繁體中文](README_TW.md)

ONNX 全功能 OCR 系统。一个内核，多档后端，CPU / GPU 双模。

```bash
pip install -r requirements.txt              # 一条命令，CPU 与 GPU 通用
python -m noocr models --get ppocrv6-tiny    # 拉权重（约 6MB）
python -m noocr 发票.jpg                      # 识别
python -m noocr serve                         # Web 界面 http://127.0.0.1:8000
```

有 NVIDIA 显卡时加 `-d cuda`（或 `serve --device cuda`），端到端快 **20-40 倍**。

遇到问题先看 [Q&A.md](Q&A.md)。

## 特性

| | |
|---|---|
| **一份依赖** | CPU 与 NVIDIA 机器装同一个 `requirements.txt`，无需重建环境 |
| **CPU / GPU 双模** | 同一份代码，`--device cpu` 或 `--device cuda`，默认自动探测 |
| **四档后端** | PP-OCRv6 tiny（6.6MB）/ small（30.3MB，默认）/ medium（132.8MB）/ v5（21.1MB） |
| **多格式输入** | 图片、PDF、Word、Excel、PPT、URL |
| **结果可复现** | 识别输出与 `rec_batch_size` 无关 |
| **四种交付** | 库 / CLI / REST API / Web 界面 |
| **统一契约** | 任何后端都返回 `OCRResult`，调用方无需分支 |

## 安装

要求 Python ≥ 3.10。

```bash
git clone https://github.com/ITerydh/NOOCR.git
cd noocr
python -m venv .venv && . .venv/Scripts/activate    # Windows
# python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

无 NVIDIA 显卡的机器装同一份依赖也能正常安装运行，只是 GPU 部分永不启用。若只跑 CPU 且在意安装体积，可换轻量版（识别结果完全一致）：

```bash
pip install -r requirements-cpu.txt     # 约 15MB
```

### GPU 需要额外的系统库

CUDA 与 cuDNN 是系统级运行库，pip 装不了，须单独安装：NVIDIA 驱动（≥ 525）、CUDA 12.x、**cuDNN 9**。

装完告诉项目 DLL 在哪（Windows 必填，Linux/macOS 一般不需要）：

```bash
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin            # Windows
export NOOCR_GPU_LIB_DIR=/usr/local/cudnn/lib       # Linux
```

项目也会自动在项目根、虚拟环境的 `site-packages/cudnn/` 等常见位置查找。装不上时的排查见 [Q&A.md](Q&A.md#gpu-不生效)。

## 获取权重

全部 19 个权重（223.6MB）托管在 ModelScope：**[iterhui/noocr-onnx](https://www.modelscope.cn/models/iterhui/noocr-onnx)**，按后端按需下载：

```bash
python -m noocr models                      # 查看各后端状态
python -m noocr models --get ppocrv6-tiny   # 6.1MB，最快
python -m noocr models --get ppocrv6-small  # 30.5MB，默认
python -m noocr models --get ppocrv6-medium # 132.8MB，服务器档，精度最高
python -m noocr models --get ppocrv5        # 21.1MB
```

或一次性拉全部：

```bash
pip install modelscope
modelscope download --model iterhui/noocr-onnx --local_dir ./models
```

下载后目录结构必须如下（`models.py` 按此路径解析）：

```
<项目根>/
├─ noocr/
└─ models/
   ├─ ppocrv6/
   │  ├─ det/
   │  │  ├─ PP-OCRv6_det_tiny.onnx      # tiny 档
   │  │  └─ PP-OCRv6_det_small.onnx     # small 档
   │  ├─ rec/
   │  │  ├─ PP-OCRv6_rec_tiny.onnx
   │  │  └─ PP-OCRv6_rec_small.onnx
   │  ├─ cls/
   │  │  └─ cls.onnx                    # 方向分类器（180°纠正）
   │  ├─ ppocrv6_tiny_dict.txt          # tiny 字典（6906 类）
   │  └─ ppocrv6_dict.txt               # small 字典（18710 类）
   └─ ppocrv5/
      ├─ det/det.onnx
      ├─ rec/rec.onnx
      ├─ cls/cls.onnx
      └─ ppocrv5_dict.txt               # 6623 类
```

可选后端（`--get plate` / `orientation` / `layout` / `table`）对应：

```
models/license_plate/{car_plate_detect.onnx, plate_rec.onnx}
models/orientation/rapid_orientation.onnx
models/layout/{layout_cdla.onnx, layout_publaynet.onnx}
models/table/slanet-plus.onnx
```

手动放置时，缺哪个文件 `python -m noocr models` 会用 `缺失` 标出。想放到别处，设 `NOOCR_MODELS_DIR`。

## 命令行

```bash
noocr <文件>                     # 打印文本
noocr <文件> -o out.json        # 结构化 JSON（含坐标与置信度）
noocr 论文.pdf -o 论文.md -f md  # PDF 转 Markdown
noocr bench 试卷.jpg             # 分阶段耗时剖析
noocr backends                   # 列出后端
noocr models                     # 权重状态
noocr serve --port 8000          # Web 界面 + API
```

常用选项：`-b/--backend` 选后端、`-f/--format` 选 `json|text|md`、`-d/--device` 选 `auto|cpu|cuda`、`--batch` 调批大小、`--no-cls` 关闭 180° 纠正。

## 作为库

```python
from noocr import ocr

result = ocr("扫描件.jpg")
print(result.text)
print(result.to_markdown())

for line in result.all_lines:
    print(f"{line.confidence:.2f}  {line.text}  {line.box.points}")
```

指定后端与参数：

```python
# 按次覆盖：这一次用 CPU / 换后端，不影响别的调用
r = ocr("扫描件.jpg", device="cpu")        # 强制 CPU
r = ocr("扫描件.jpg", backend="ppocrv5")   # 临时换后端
r = ocr("扫描件.jpg", dpi=300, max_pages=5)

# 固化配置：长驻服务应该自己持有一个实例，模型只加载一次
from noocr import OCRPipeline

pipe = OCRPipeline(backend="ppocrv6-tiny", device="cuda", rec_batch_size=8)
for path in paths:
    print(pipe(path).text)
```

## Web 界面与 API

```bash
noocr serve --host 0.0.0.0 --port 8000 --device cuda
```

界面 `http://127.0.0.1:8000/` —— 左侧上传与参数、右侧图文对照双列联动、下方识别记录与示例图。顶栏设备徽标可直接点击切换 CPU / GPU，无需重启。接口文档 `http://127.0.0.1:8000/docs`。

<details open>
<summary><b>界面图例</b>（点击展开）</summary>

**图文对照双列** —— 左侧上传与参数，右侧原图带识别框与识别文本双列联动，下方为识别记录与示例图。顶栏设备徽标显示当前实际生效的设备。

![Web 界面：图文对照双列](docs/images/demo-overview.jpg)

**明细联动** —— 点明细表任一行，图上对应文本框高亮为蓝色，其余保持绿色。

![Web 界面：点明细行图上高亮联动](docs/images/webui-detail.jpg)

</details>

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/backends` | 可用后端列表 |
| `GET` | `/api/device` | 设备偏好与**实际生效**的设备 |
| `POST` | `/api/device` | 切换设备（`auto`/`cpu`/`cuda`） |
| `POST` | `/api/ocr` | 上传文件识别（multipart） |
| `POST` | `/api/ocr/path?path=...` | 识别服务器本地路径 |
| `GET` | `/api/page/{doc_id}/{i}` | 取第 i 页渲染图（多页翻页用） |
| `POST` | `/api/warmup` | 预加载后端 |

```bash
curl -X POST http://127.0.0.1:8000/api/ocr \
  -F "file=@发票.jpg" -F "backend=ppocrv6-tiny"
```

## 性能

RTX 4070 Ti SUPER + `ppocrv6-small`，与性能对比表同一批 8 张图、含图片解码，每档独立进程、3 轮预热后取 10 轮均值：

| 图片 | 行数 | CPU | GPU | 加速比 |
|---|---|---|---|---|
| 对比表格 | 73 | 1380 ms | **446 ms** | 3.1x |
| 小学试卷 | 65 | 1932 ms | **711 ms** | 2.7x |
| 身份证 | 11 | 1573 ms | **248 ms** | 6.3x |
| 化验单 | 69 | 1253 ms | **355 ms** | 3.5x |
| 规格书 | 16 | 1517 ms | **262 ms** | 5.8x |
| 银行流水 | 31 | 1608 ms | **349 ms** | 4.6x |
| 竖式牌匾 | 2 | 1265 ms | **98 ms** | 13.0x |
| 火车票 | 19 | 1696 ms | **295 ms** | 5.8x |
| **合计** | — | **12223 ms** | **2763 ms** | **4.4x** |

加速比与「行数多少」关系不大，主要看版面复杂度——`对比表格` 是密集小字，CPU 上光后处理就占了 9 秒；`竖式牌匾` 只有 2 行但图大，GPU 上只要 98ms。CPU 上密集版面反而更吃亏。

GPU 侧首次请求含约 2.4s 的模型加载与 CUDA kernel 编译，之后稳定在 0.1-0.8s。复现：

```bash
python scripts/perf/bench_device.py cpu 10   > device_cpu.json
python scripts/perf/bench_device.py cuda 10  > device_cuda.json
python scripts/sync_readme_tables.py          # 把结果写回 README
```

两个 device 要分开跑——同一进程里连续跑会让 CPU 线程配置互相干扰，测出来的加速比没有意义。

### 与 OnnxOCR 实测对比

**四个档位、两侧跑同一份权重文件**（det / rec / 字典经 MD5 校验逐字节一致）、同一块 RTX 4070 Ti SUPER、同为 ONNX Runtime 1.23.2，两侧都开启方向分类、含图片解码，每图 3 轮预热后取 10 轮均值：

<details open>
<summary><b>PP-OCRv5</b>（点击展开）</summary>

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv5 | NOOCR<br>PP-OCRv5 | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 2913 ms | **349 ms** | **8.3x** | 82 / 79 |
| `exam_chinese_primary` | 1920x2560 | 3204 ms | **558 ms** | **5.7x** | 68 / 73 |
| `id_card_china` | 1148x672 | 558 ms | **201 ms** | **2.8x** | 9 / 10 |
| `medical_lab_report` | 430x267 | 2484 ms | **297 ms** | **8.4x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 774 ms | **206 ms** | **3.8x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1370 ms | **278 ms** | **4.9x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 299 ms | **72 ms** | **4.2x** | 2 / 2 |
| `ticket_train` | 670x510 | 956 ms | **213 ms** | **4.5x** | 19 / 17 |
| **均值** | — | **1570 ms** | **272 ms** | **5.78x** | 296 / 296 |
| **合计** | — | **12558 ms** | **2173 ms** | **5.78x** | — |

</details>
<details>
<summary><b>PP-OCRv6 tiny</b>（点击展开）</summary>

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv6 tiny | NOOCR<br>PP-OCRv6 tiny | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 1851 ms | **303 ms** | **6.1x** | 73 / 74 |
| `exam_chinese_primary` | 1920x2560 | 2206 ms | **458 ms** | **4.8x** | 70 / 75 |
| `id_card_china` | 1148x672 | 390 ms | **160 ms** | **2.4x** | 9 / 9 |
| `medical_lab_report` | 430x267 | 1721 ms | **244 ms** | **7.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 542 ms | **138 ms** | **3.9x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 952 ms | **235 ms** | **4.1x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 196 ms | **63 ms** | **3.1x** | 2 / 2 |
| `ticket_train` | 670x510 | 657 ms | **172 ms** | **3.8x** | 19 / 20 |
| **均值** | — | **1064 ms** | **222 ms** | **4.80x** | 289 / 295 |
| **合计** | — | **8514 ms** | **1773 ms** | **4.80x** | — |

</details>
<details>
<summary><b>PP-OCRv6 small</b>（点击展开）</summary>

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv6 small | NOOCR<br>PP-OCRv6 small | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 2661 ms | **437 ms** | **6.1x** | 71 / 73 |
| `exam_chinese_primary` | 1920x2560 | 2930 ms | **696 ms** | **4.2x** | 71 / 65 |
| `id_card_china` | 1148x672 | 567 ms | **245 ms** | **2.3x** | 10 / 11 |
| `medical_lab_report` | 430x267 | 2463 ms | **353 ms** | **7.0x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 750 ms | **254 ms** | **3.0x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1316 ms | **341 ms** | **3.9x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 292 ms | **95 ms** | **3.1x** | 2 / 2 |
| `ticket_train` | 670x510 | 928 ms | **288 ms** | **3.2x** | 19 / 19 |
| **均值** | — | **1488 ms** | **339 ms** | **4.39x** | 289 / 285 |
| **合计** | — | **11908 ms** | **2710 ms** | **4.39x** | — |

</details>
<details>
<summary><b>PP-OCRv6 medium</b>（点击展开）</summary>

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv6 medium | NOOCR<br>PP-OCRv6 medium | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 4343 ms | **468 ms** | **9.3x** | 74 / 73 |
| `exam_chinese_primary` | 1920x2560 | 4791 ms | **713 ms** | **6.7x** | 64 / 66 |
| `id_card_china` | 1148x672 | 956 ms | **240 ms** | **4.0x** | 12 / 11 |
| `medical_lab_report` | 430x267 | 4050 ms | **447 ms** | **9.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 1286 ms | **290 ms** | **4.4x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 2227 ms | **386 ms** | **5.8x** | 32 / 32 |
| `scene_vertical_plaque` | 720x1150 | 603 ms | **110 ms** | **5.5x** | 2 / 2 |
| `ticket_train` | 670x510 | 1621 ms | **338 ms** | **4.8x** | 19 / 19 |
| **均值** | — | **2485 ms** | **374 ms** | **6.64x** | 288 / 288 |
| **合计** | — | **19877 ms** | **2993 ms** | **6.64x** | — |

</details>

**模型越大，优势越明显。** v5 上快 5.78 倍，v6 medium 上快 6.64 倍——因为 OnnxOCR 在大模型上没有做输入分辨率与内存 arena 的取舍，全部按最大配置跑；NOOCR 按形状特征选分辨率并关闭 arena 重规划，模型越大这部分省得越多。

四档的文本行数都与对方基本一致（v5 296/296、medium 288/288），说明预处理口径对等，差距来自实现而非「谁认得更多字」。差距主要来自三处：det 的输入分辨率与内存 arena 策略（形状集合有限离散，关掉 `dynamic_shape` 让 arena 生效后提速约 32%）、rec 的字典裁剪，以及会话复用。

复现：`scripts/perf/compare_onnxocr.md`（需本机另备一份 OnnxOCR 检出与依赖环境）。

## 后端对比

| 后端 | 体积 | 相对速度 | 加权置信度 | 适用 |
|---|---|---|---|---|
| `ppocrv6-tiny` | 6.6MB | **1.23x** | 0.956 | 边缘设备、批量 |
| `ppocrv5` | 21.1MB | 1.00x | 0.928 | 兼容旧项目 |
| `ppocrv6-small` | 30.3MB | 0.80x | 0.973 | 默认 |
| `ppocrv6-medium` | 132.8MB | 0.73x | **0.981** | 版面复杂、追求精度 |

RTX 4070 Ti SUPER、8 张示例图、3 轮预热后10 轮取均值（含图片解码），相对速度以 `ppocrv5` 为基准；耗时取自与上文对比表**同一批产物**，口径完全一致。加权置信度按字数加权（`scripts/perf/bench_tiers.py`，每档独立进程测量）。

四个 v6 档位的置信度都高于 v5，但**只有 tiny 比 v5 快**。small 与 medium 都更慢，换来的是更高的置信度：medium比 small 再慢约 10%，置信度 0.981 对 0.973（v5 为 0.928）。

> 要快选 tiny，要均衡选 small，要精度选 medium。v5 仅在需要兼容旧项目时使用。

## 项目结构

```
noocr/
├─ types.py              统一数据模型（Pydantic）
├─ models.py             权重清单与下载
├─ cli.py                命令行
├─ engine/               推理引擎
│  ├─ session.py           session 缓存、设备探测、GPU 运行库挂载
│  ├─ imageops.py          图像几何与预处理
│  └─ base.py              后端抽象
├─ backends/             OCR 后端
│  ├─ ppocr.py             PP-OCRv5
│  ├─ ppocrv6.py           PP-OCRv6
│  ├─ postprocess.py       DB 检测后处理
│  └─ decode.py            CTC 解码
├─ inputs/loader.py      多格式输入
├─ document/             文档结构化
├─ pipeline/             编排
└─ web/                  Web 界面与 API
    ├─ app.py
    └─ templates/index.html
tests/                   测试与基准
scripts/
├─ publish_weights.py      权重发布脚本
├─ perf/                   性能基准与 A/B 脚本
│  ├─ bench_device.py      CPU / GPU 基线
│  ├─ bench_noocr.py       单引擎基准（主对比表 NOOCR 侧）
│  ├─ compare_make_table.py 由产物生成对比表
│  ├─ bench_tiers.py       各档位相对速度与置信度
│  ├─ compare_onnxocr.md   与 OnnxOCR 的对比复现说明
│  └─ ab_*.py              arena / cuDNN / 档位 A/B
├─ shrink_images.py        截图压缩
└─ models_repo_card.md     权重仓库模型卡
```

## 致谢

本项目在以下开源项目的启发与基础上构建，感谢原作者与社区：

- **[OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)** —— ONNX 推理链路、模型导出与文档结构化思路，是本项目的起点。
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** —— PP-OCR 系列模型与算法设计，本项目的四档后端权重全部来自该项目。
- **[DeepSeek-AI](https://github.com/deepseek-ai/DeepSeek-OCR)** —— OCR 精度优化的思路参考。

若上述项目的成果对你的工作有帮助，请优先支持它们。

## 许可

代码 Apache-2.0。权重源自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0），按原许可条款使用。