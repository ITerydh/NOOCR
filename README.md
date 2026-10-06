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
| **三档后端** | PP-OCRv6 tiny（6MB）/ small（32MB，默认）/ v5（16MB） |
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

全部 17 个权重（95.8MB）托管在 ModelScope：**[iterhui/noocr-onnx](https://www.modelscope.cn/models/iterhui/noocr-onnx)**，按后端按需下载：

```bash
python -m noocr models                     # 查看各后端状态
python -m noocr models --get ppocrv6-tiny  # 6.1MB，最快
python -m noocr models --get ppocrv6-small # 30.5MB，默认
python -m noocr models --get ppocrv5       # 15.6MB
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

RTX 4070 Ti SUPER + ppocrv6-small，同一批示例图取中位数：

| 图片 | 行数 | CPU | GPU | 加速比 |
|---|---|---|---|---|
| 票据 | 19 | 15537 ms | **594 ms** | 26x |
| 银行流水 | 30 | 13040 ms | **577 ms** | 23x |
| 化验单 | 69 | 10396 ms | **567 ms** | 18x |
| 身份证 | 11 | 13317 ms | **400 ms** | 33x |
| 银行网点 | 4 | 9414 ms | **249 ms** | 38x |

GPU 侧首次请求含约 2.4s 的模型加载与 CUDA kernel 编译，之后稳定在 0.25-0.6s。复现：

```bash
python scripts/perf/bench_device.py cpu     # CPU 基线
python scripts/perf/bench_device.py cuda    # GPU 基线
```

## 后端对比

| 后端 | 体积 | 相对速度 | 加权置信度 | 适用 |
|---|---|---|---|---|
| `ppocrv6-tiny` | 6.1MB | **0.58x** | 0.961 | 边缘设备、批量 |
| `ppocrv6-small` | 30.5MB | 1.73x | **0.971** | 默认 |
| `ppocrv5` | 15.6MB | 1.00x | 0.936 | 兼容旧项目 |

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
└─ models_repo_card.md     权重仓库模型卡
```

## 许可

代码 Apache-2.0。权重源自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0），按原许可条款使用。
