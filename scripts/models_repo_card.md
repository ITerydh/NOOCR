# NOOCR ONNX 权重

[NOOCR](https://github.com/iterhui/noocr) 使用的全部 ONNX 权重，按项目内的 `models/` 目录结构组织，
因此下载后无需任何路径映射。

## 目录结构

```
ppocrv5/          PP-OCRv5 通用 OCR（det + rec + cls + 6623 类字典）
ppocrv6/          PP-OCRv6 通用 OCR（tiny / small 两档 + 方向分类器 + 字典）
license_plate/    车牌检测与识别
orientation/      文字方向分类
layout/           版面分析（CDLA / PubLayNet）
table/            表格结构识别（SLANet+）
```

## 用法

按需下载单个后端（推荐）：

```bash
pip install noocr
noocr models --get ppocrv6-tiny
```

一次性拉全部：

```bash
pip install modelscope
modelscope download --model iterhui/noocr-onnx --local_dir ./models
```

或直接下载单个文件：

```
https://www.modelscope.cn/models/iterhui/noocr-onnx/resolve/master/ppocrv6/det/PP-OCRv6_det_small.onnx
```

## 许可

权重源自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)，Apache-2.0。
