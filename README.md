# NOOCR

ONNX 全功能 OCR 系统。一个内核，多档后端，CPU 优先。

```bash
pip install -r requirements.txt
python -m noocr models --get ppocrv6-tiny   # 拉权重（约 6MB）
python -m noocr 发票.jpg                     # 识别
python -m noocr serve                        # Web 界面 http://127.0.0.1:8000
```

---

## 特性

| | |
|---|---|
| **纯 CPU 推理** | 依赖 onnxruntime，无需显卡 |
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

可选依赖：

```bash
pip install "noocr[doc]"    # PDF / Word / Excel / PPT
pip install "noocr[web]"    # Web 界面与 REST API
pip install "noocr[gpu]"    # NVIDIA GPU 加速
```

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
noocr backends                # 列出后端
noocr models                  # 权重状态
noocr serve --port 8000        # Web 界面 + API
```

常用选项：`-b/--backend` 选后端、`-f/--format` 选 `json|text|md`、`--batch` 调批大小、`--no-cls` 关闭 180° 纠正。

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
from noocr.backends import get_backend

backend = get_backend("ppocrv6-tiny", rec_batch_size=8)
backend.load()
page = backend.recognize_image(image)      # image 为 BGR ndarray
```

## Web 界面与 API

```bash
noocr serve --host 0.0.0.0 --port 8000
```

界面 `http://127.0.0.1:8000/` —— 拖入文件即识别，支持文本 / Markdown / 图文对照 / 明细四个视图。
接口文档 `http://127.0.0.1:8000/docs`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/backends` | 可用后端列表 |
| `POST` | `/api/ocr` | 上传文件识别（multipart） |
| `POST` | `/api/ocr/path?path=...` | 识别服务器本地路径 |
| `POST` | `/api/warmup` | 预加载后端 |

```bash
curl -X POST http://127.0.0.1:8000/api/ocr \
  -F "file=@发票.jpg" -F "backend=ppocrv6-tiny"
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
│  ├─ session.py           session 缓存与设备管理
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
└─ models_repo_card.md     权重仓库模型卡
```

## 许可

代码 Apache-2.0。权重源自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0），按原许可条款使用。