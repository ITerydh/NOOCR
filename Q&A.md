# Q&A

[简体中文](README.md) | [English](README_EN.md) | [繁體中文](README_TW.md)

NOOCR 使用中可能遇到的问题与排查方法。按现象分类，每条给出原因与处理方式。

---

## 安装与环境

### `pip install -r requirements.txt` 装不上或报错

先确认 Python ≥ 3.10：

```bash
python --version
```

`requirements.txt` 装的是 `onnxruntime-gpu`（约 700MB，含 CPU / CUDA / TensorRT 三个 EP）。它体积大、下载慢，装不上时可以先用轻量版：

```bash
pip install -r requirements-cpu.txt
```

轻量版识别结果与 GPU 版**完全一致**，只是没有 GPU 加速。之后想换 GPU 版，直接执行 `pip install -r requirements.txt` 覆盖即可。

### 为什么要装 GPU 版而不是 CPU 版

`onnxruntime-gpu` 是 `onnxruntime` 的超集，一个包覆盖两种机器，所以不需要区分「CPU 环境」和「GPU 环境」。没有 NVIDIA 显卡的机器装它照样能正常安装运行，只是 GPU 部分永不启用。

### Windows 终端输出中文报 `UnicodeEncodeError`

控制台默认编码是 cp1252。项目已做三层防护（包内 `logging_config`、各脚本的 `reconfigure`、CI 的 `PYTHONIOENCODING`）；若你是在自己的脚本里遇到，运行前设置：

```powershell
$env:PYTHONIOENCODING="utf-8"
```

Linux/macOS 下通常不需要，设了也无害。

---

## GPU 不生效

**这是最常见也最难排查的问题。** 根因通常是 cuDNN 版本不对，而它的表现非常隐蔽：ONNX Runtime **不报错**，只是把算子悄悄交给 CPU，你会得到纯 CPU 的性能却以为在用显卡。

先确认实际生效的设备：

```bash
python -m noocr 文件.jpg -d cuda
```

输出里的「实际=」才是真正生效的设备。若显示 `cpu`，按下面几步排查。

### 确认装的是 GPU 版

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

输出里必须有 `CUDAExecutionProvider`。只有 `CPUExecutionProvider` 说明装的是纯 CPU 版：

```bash
pip install -r requirements.txt
```

### 确认 cuDNN 是 9.x

ONNX Runtime 1.20+ 依赖 **`cudnn64_9.dll`**（cuDNN **9.x**）。装成 cuDNN 8 就是不生效。

检查 DLL 是否就位：

```powershell
dir D:\libs\cudnn9\bin\cudnn64_9.dll
```

然后告诉项目 DLL 在哪（Windows 必填）：

```powershell
set NOOCR_GPU_LIB_DIR=D:\libs\cudnn9\bin
```

项目也会自动在项目根、`noocr-gpu/` 等同级虚拟环境的 `site-packages/cudnn/` 下寻找。**若 GPU 环境与 CPU 环境分开建**（推荐，避免 ORT 的 DLL 互相覆盖），把 DLL 放到 GPU 环境里：

```
<gpu-venv>/Lib/site-packages/cudnn/
```

### 确认驱动与 CUDA 版本

NVIDIA 驱动需 ≥ 525，CUDA 需12.x。`nvidia-smi` 能正常输出即说明驱动没问题。

### 项目做了什么防护

本项目不只靠「包是否编译进来」判断，而是三重校验：

1. 启动时用微型 ONNX 模型**实测** CUDA EP 能否初始化；
2. session 创建后**核对实际生效的 EP**，与请求不符直接报错；
3. Web 界面顶栏常驻**设备徽标**，GPU 未生效时显示为 CPU。

Web 界面里点击顶栏设备徽标可随时切换，下拉中本机无 GPU 环境时「显卡」会置灰并写明原因。

### 加速比不如预期

GPU 上「首次识别」含约 2.4s 的模型加载与 CUDA kernel 编译。第二次请求起才稳定在 0.25-0.6s，请用第二次以后的耗时对比。

同时确认用的是 `ppocrv6-small`（默认档）。极小的图在 GPU 上启动开销占主导，加速比会偏低。

---

## 权重与模型

### 提示权重缺失

```bash
python -m noocr models
```

会逐项标出 `缺失` 的文件。按 [README 的放置位置](README.md#获取权重)补齐目录结构即可。

### 权重放哪里

项目默认读 `./models/`。想放到别处：

```bash
export NOOCR_MODELS_DIR=/data/ocr/models    # Linux/macOS
set NOOCR_MODELS_DIR=D:\ocr\models          # Windows
```

### 有哪些档位，各自多大

```bash
python -m noocr models --get ppocrv6-tiny
```

四个后端档位：

| 后端 | 整档体积 | 适用 |
|---|---|---|
| `ppocrv6-tiny` | 6.6MB | 边缘设备、批量扫描，追求极致速度 |
| `ppocrv6-small` | 30.3MB | 默认档，速度与精度均衡 |
| `ppocrv6-medium` | 132.8MB | 服务器档，版面复杂、追求最高精度 |
| `ppocrv5` | 21.1MB | 上一代，兼容旧项目 |

四个档位共用同一份方向分类器与字典，所以四个都装也只有 190MB 左右。

### 下拉框里出现两个名字相同的后端

不应该出现。若出现，说明 `ppocrv6` 与 `ppocrv6-small` 被同时当成了独立后端——两者是同一档位（前者是后者的别名）。更新到最新版即可。

---

## 识别效果

### 识别结果与批大小不一致

正常现象。相同图片的输出应与 `rec_batch_size` 完全无关，本项目保证这一点。若发现不一致，说明环境有问题，请提 issue。

### 竖排文字或旋转文字识别不准

确认没有关掉 180° 纠正（默认开启）：

```bash
python -m noocr 文件.jpg              # 开启（默认）
python -m noocr 文件.jpg --no-cls     # 关闭
```

`--no-cls` 会跳过方向分类器，图片本身是正的时可以用。

### PDF 识别不完整或很慢

受 DPI 与最多页数影响：

```bash
python -m noocr 论文.pdf --dpi 300 --max-pages 10
```

`--dpi` 默认 200。扫描件质量差时可提到 300；页数多时用 `--max-pages` 截断，避免一次处理几十页。

### 置信度偏低

置信度低于 0.8 的行在 Web 界面里可以勾选「只看低置信」单独筛出来。整体偏低时：

- 换更高精度的档位：`ppocrv6-medium` 加权置信度 0.981，是四档里最高的；`ppocrv6-small`（默认档）0.973；`ppocrv5` 0.928。数值来自 8 张示例图的字数加权实测（`scripts/perf/bench_tiers.py`）；
- 提高输入图片分辨率；
- 确认方向纠正没有被误判（倾斜角度识别错会连带拉低置信度）。

### 想要最高精度，但 medium 比 small 慢

正常。`ppocrv6-medium` 参数是 small 的 4 倍，精度最高（加权置信度 0.981，
small 为 0.973），代价是耗时更长：8 张示例图实测 medium 2993ms、small 2710ms，
慢约 10%。想要更快就用 small，版面复杂、精度优先再上 medium。

（早期版本曾给出「medium 反而比 small 快」的结论，那是测量方法有问题：
预热轮数不足且解码没计入耗时。修正后数据见上。）

---

## Web 界面

### 图片显示不完整

点图片右上角的「适应」按钮恢复自动缩放。手动放大后图片会超出可视区并出现滚动条，这是预期行为。

### 识别记录里的图片点不开

识别记录（下方区域）只保存**文本快照**，页面图仅在当前这一轮结果中可用。这是设计如此：后端文档缓存按容量淘汰，历史记录的图不一定还在。想完整回看，请重新识别。

### 点明细行图上没反应

明细与图片是联动的。若图上没有高亮框，通常是「显示文本框」被勾掉了，在左侧参数区重新勾选即可。

### 上传 PDF / Word 后没有缩略图

只有图片会在上传位置显示缩略图。PDF 与 Office 文件会在右侧显示首屏预览。

---

## 性能调优

### 换 GPU 后速度没变

先按 [GPU 不生效](#gpu-不生效) 排查。若 GPU 确实生效，可继续调优：

```bash
python -m noocr bench 文件.jpg -d cuda      # 分阶段耗时剖析
python scripts/perf/ab_arena.py             # 内存 arena 策略 A/B
python scripts/perf/ab_cudnn.py             # cuDNN 算法搜索 A/B
python scripts/perf/ab_tiers.py cuda        # 档位数 A/B（含输出一致性校验）
```

### CPU 下怎么快一点

- 用 `ppocrv6-tiny`（四档里最快，8 张示例图比 `ppocrv5` 快约 1.95 倍，精度略降）；
- 调 `--batch`（CPU 建议 6，GPU 可调大）；
- 降低 PDF 的 `--dpi`。

### 推理很慢，怀疑内存 arena 配置

ONNX Runtime 的 `dynamic_shape=True` 会连带关掉内存 arena。对**宽度持续变化**的模型这是必须的（rec 的输入宽度随文本长度任意增长，强行开 arena 会退化到秒级）；但对**形状集合有限且离散**的模型则是白扔。

PP-OCRv6 的 det 短边恒为 736、长边被上限截断且对齐到 32，属于后者。5 张不同长宽比示例图交错 A/B、各 12 轮：

| det 配置 | 中位耗时/张 |
|---|---|
| `dynamic_shape=True`（arena 关） | 233.3 ms |
| `dynamic_shape=False`（arena 开） | **157.6 ms**（-32.4%） |

两组波动区间（A 227.7~239.2、B 149.8~168.5）完全不重叠，差异显著。端到端从 1541 ms 降到 1430 ms。项目已默认这么配，复现用 `python scripts/perf/ab_arena.py`。

该脚本在两组波动区间重叠时会明确报告「差异不显著」而不是硬下结论——本机负载高时结论就该是这句。

---

## 开发与部署

### `git push` 报 `Permission to <仓库> denied to <另一个用户名>`

部分 Git 发行版在**系统级** `gitconfig` 里配了 `credential.helper = helper-selector`，它优先于你的 `~/.gitconfig`，会返回 Windows 凭据管理器里存着的**另一个账号**——于是明明配对了 token 仍被拒。

在仓库目录内加一条仓库级配置即可覆盖，不必改动全局设置：

```bash
git config --local credential.helper ""
git config --local credential.helper store
```

先清空再写入是关键：只写第二条是追加而非替换，空条目负责顶掉上层。

验证凭据是否被正确读取（`password` 不会回显）：

```bash
printf 'protocol=https\nhost=github.com\n\n' \
  | git -c credential.helper= -c credential.helper=store credential fill
```

输出里的 `username` 不是仓库所有者，就说明凭据还没生效。

### 推送时网络不通

设代理后重试：

```powershell
$Env:http_proxy="http://127.0.0.1:1080"; $Env:https_proxy="http://127.0.0.1:1080"
```

先测连通性，通了再 push：

```bash
curl -s -o /dev/null -w "%{http_code}" -m 15 https://github.com
```

返回 000 说明直连不通，200 才是通的。

### 想部署成对外服务

```bash
noocr serve --host 0.0.0.0 --port 8000 --device cuda
```

注意：`/api/ocr/path` 允许读取服务器本地路径，对外暴露前请自行加访问控制。
