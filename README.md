# NOOCR

ONNX 全功能 OCR 系统。一个内核，多档后端，CPU / GPU 双模。

```bash
pip install -r requirements.txt      # 一条命令，CPU 与 GPU 通用
python -m noocr models --get ppocrv6-tiny   # 拉权重（约 6MB）
python -m noocr 发票.jpg                     # 识别
python -m noocr serve                        # Web 界面 http://127.0.0.1:8000
```

有 NVIDIA 显卡时加 `-d cuda`（或 `serve --device cuda`），端到端快 **20-40 倍**。

---

## 特性

| | |
|---|---|
| **一份依赖** | CPU 与 NVIDIA 机器装同一个 `requirements.txt`，无需重建环境 |
| **CPU / GPU 双模** | 同一份代码，`--device cpu` 或 `--device cuda`，可自动探测 |
| **三档后端** | PP-OCRv6 tiny（6MB）/ small（32MB，默认）/ v5（22MB） |
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

### 为什么只有一份依赖

`requirements.txt` 装的是 `onnxruntime-gpu` 而非 `onnxruntime`，因为前者是后者的**超集**——它同时含CPU、CUDA、TensorRT 三个 EP，装一个包就覆盖了两种机器：

| 包 | 可用 EP | 体积 |
|---|---|---|
| `onnxruntime` | CPU | ~15 MB |
| `onnxruntime-gpu` | CPU / CUDA / TensorRT | ~700 MB |

没有 NVIDIA 显卡的机器装GPU 版**照样能正常安装与运行**，只是GPU 部分永不启用。所以不需要区分「CPU 环境」和「GPU 环境」，也不需要为GPU 重建虚拟环境。

代价是体积。若只跑 CPU 且在意安装速度/磁盘，换一份轻量的：

```bash
pip install -r requirements-cpu.txt     # 约 15MB 的 ORT，识别结果完全一致
```

哪天要开 GPU，改为执行 `pip install -r requirements.txt` 即可（GPU 版会覆盖 CPU 版）。

### 系统级依赖（仅 GPU 需要）

pip 只管 Python 包，CUDA 与 cuDNN 是**系统级运行库**，须单独安装：

1. NVIDIA 驱动（≥ 525）
2. CUDA 12.x
3. **cuDNN 9**（注意是 9.x，ORT 1.20+ 依赖 `cudnn64_9.dll`）

装完告诉项目 DLL 在哪（Windows 必填，Linux/macOS 一般不需要）：

```bash
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin            # Windows
export NOOCR_GPU_LIB_DIR=/usr/local/cudnn/lib       # Linux
```

项目也会自动在项目根、虚拟环境的 `site-packages/cudnn/` 等常见位置查找。

## GPU 加速

GPU 上端到端比 CPU 快 **20-40 倍**（见下方[性能](#性能)）。依赖已在 `requirements.txt` 里，无需额外安装 Python 包；只要系统层装好 CUDA 12 + cuDNN 9 即可：

```bash
python -m noocr 发票.jpg -d cuda
python -m noocr serve --device cuda
```

`--device` 可选 `auto`（默认，有 CUDA 就用）/ `cpu` / `cuda`。

### Web 界面里随时切换

`serve` 启动后，**顶栏设备徽标可直接点击切换**，无需重启服务：

- 徽标显示的是**实际生效**的设备（GPU 生效时显示显卡名并变绿），不是启动时的请求值；
- 下拉里 `自动 / 显卡 / 处理器` 三项，本机无 GPU 环境时「显卡」会置灰并写明原因；
- 切换时自动卸载另一套模型并归还显存，代价约 0.4-0.6s；切换后若已有识别结果会自动重跑，方便直接对比速度；
- 目标设备不可用时返回 400 并**保持原设备不变**，不会把能跑的服务弄成不能跑。

也可以走 API：

```bash
curl http://127.0.0.1:8000/api/device                       # 当前设备
curl -X POST -F "device=cuda" .../api/device              # 切到 GPU
```

### cuDNN 9 是必需的

ONNX Runtime 1.20+ 的 CUDA EP 依赖 **cuDNN 9**（`cudnn64_9.dll`）。装错版本的表现非常隐蔽：ORT **不报错**，只是把算子悄悄交给 CPU，你会得到纯 CPU 的性能却以为在用显卡。

本项目对此做了三重防护：

1. 启动时用微型 ONNX 模型**实测** CUDA EP 能否初始化，而非只看它是否被编译进来；
2. session 创建后**核对实际生效的 EP**，与请求不符直接报错；
3. Web 界面顶栏常驻**设备徽标**，GPU 未生效时显示为 CPU。

cuDNN 9 安装（解压后把 `bin` 下的 DLL 放到任一目录并告知项目）：

```bash
# 从 https://developer.nvidia.com/cudnn-downloads 下载 cuDNN 9 for CUDA 12
# Windows 需显式登记 DLL 搜索路径，Linux/macOS 直接给权限即可
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin
```

项目会自动在项目根、`noocr-gpu/` 等同级虚拟环境的 `site-packages/cudnn/` 下寻找，无需手动设置。

> **若 GPU 环境与 CPU 环境分开建**（推荐，避免 ORT 的 DLL 互相覆盖），把 cuDNN 的 DLL 放到 GPU 环境里：
> `<gpu-venv>/Lib/site-packages/cudnn/`。

### 一个值得记住的坑

`cudnn_conv_algo_search` **必须设在 provider 级选项里**，写成 `"DEFAULT"` 会让 PP-OCRv6-rec 的 228 个卷积集体退回 CPU 实现，单次推理从 5.9ms 劣化到 90.2ms（15倍）。且 provider 级配置**优先级高于** SessionOptions，写错地方会被静默覆盖。本项目已在 `build_providers()` 里固定为 `"EXHAUSTIVE"`。

## 获取权重

全部 17 个权重（95.8MB）托管在 ModelScope：**[iterhui/noocr-onnx](https://www.modelscope.cn/models/iterhui/noocr-onnx)**，按后端按需下载：

```bash
python -m noocr models                     # 查看各后端状态
python -m noocr models --get ppocrv6-tiny  # 6.9MB，最快
python -m noocr models --get ppocrv6-small # 32MB，默认
python -m noocr models --get ppocrv5       # 22MB，上一代
```

也可SDK 一次性拉全部：

```bash
pip install modelscope
modelscope download --model iterhui/noocr-onnx --local_dir ./models
```

或直接下载单个文件（路径与下表一致）：

```
https://www.modelscope.cn/models/iterhui/noocr-onnx/resolve/master/ppocrv6/det/PP-OCRv6_det_small.onnx
```

### 放置位置

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

手动放置时，缺哪个文件 `python -m noocr models` 会用 `缺失` 标出。

想放到别处，设环境变量：

```bash
export NOOCR_MODELS_DIR=/data/ocr/models    # Linux/macOS
set NOOCR_MODELS_DIR=D:\ocr\models          # Windows
```

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
# 按次覆盖：这一次用CPU / 换后端，不影响别的调用
r = ocr("扫描件.jpg", device="cpu")        # 强制 CPU
r = ocr("扫描件.jpg", backend="ppocrv5")   # 临时换后端
r = ocr("扫描件.jpg", dpi=300, max_pages=5)

# 固化配置：长驻服务应该自己持有一个实例，模型只加载一次
from noocr import OCRPipeline

pipe = OCRPipeline(backend="ppocrv6-tiny", device="cuda", rec_batch_size=8)
for path in paths:
    print(pipe(path).text)
```

也可以直接拿后端：

```python
from noocr.backends import get_backend

backend = get_backend("ppocrv6-tiny", rec_batch_size=8, device="cuda")
backend.load()
page = backend.recognize_image(image)      # image 为 BGR ndarray
print(page.debug["stage_ms"])              # 分阶段耗时，便于定位慢在哪一步
```

## Web 界面与 API

```bash
noocr serve --host 0.0.0.0 --port 8000 --device cuda
```

界面 `http://127.0.0.1:8000/` —— 拖入文件即识别，文本 / Markdown / 图文对照三视图，图文与明细双列联动。
接口文档 `http://127.0.0.1:8000/docs`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/backends` | 可用后端列表 |
| `GET` | `/api/device` | 设备偏好与**实际生效**的设备 |
| `POST` | `/api/ocr` | 上传文件识别（multipart） |
| `POST` | `/api/ocr/path?path=...` | 识别服务器本地路径 |
| `GET` | `/api/page/{doc_id}/{i}` | 取第i 页渲染图（多页翻页用） |
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

GPU 侧首次请求含约 2.4s 的模型加载与 CUDA kernel 编译，之后稳定在 0.25-0.6s。

复现：

```bash
python scripts/perf/bench_device.py cpu     # CPU 基线
python scripts/perf/bench_device.py cuda    # GPU 基线
```

其他工具：

```bash
python scripts/perf/bench_buckets.py cuda   # 分桶与串行调用次数
python scripts/perf/ab_tiers.py cuda# 档位数A/B（含输出一致性校验）
python scripts/perf/ab_cudnn.py             # cuDNN 算法搜索 A/B
python scripts/perf/prof_rec.py cuda        # ORT profiler 逐算子耗时
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
├─ perf/                   性能基准与A/B 脚本
└─ models_repo_card.md     权重仓库模型卡
```

## 许可

代码 Apache-2.0。权重源自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0），按原许可条款使用。