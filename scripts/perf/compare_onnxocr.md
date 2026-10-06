# 与 OnnxOCR 的性能对比

复现 README「与 OnnxOCR 实测对比」一节的数据。

## 为什么拆成三个脚本

两个引擎的依赖**不能装在同一个环境里**——OnnxOCR 需要
`numpy<2.0` 与自己那套依赖树，和本项目的 `onnxruntime-gpu`
版本要求冲突（新版 ORT 要 CUDA 13，旧版要 CUDA 12）。
强行合并的结果是两边都跑不起来。

所以流程是「各自环境独立跑，产出 JSON，再合成表」。

## 前置

两个环境都要能真正启用 CUDA。**这里有个容易踩的坑**：
Windows 上光改 `PATH` 不够，必须 `os.add_dll_directory`
登记，ORT 才能加载 provider 桥接 dll。少了这步 CUDA 会
**静默初始化失败并回退 CPU**，测出来的数字全是CPU 的，
但表面上一切正常。

`compare_onnxocr_side.py` 里的 `_register_dll_dirs()`
就是干这个的——OnnxOCR 自身没有这层处理。

另需确认两侧 ORT 版本一致（本次用 1.23.2），否则 CUDA
依赖不同，数字没有可比性。

## 步骤

```bash
# 0. 检出 OnnxOCR，并另建依赖环境
git clone --depth 1 https://github.com/jingsongliujing/OnnxOCR.git
python -m venv onnxocr-env
./onnxocr-env/Scripts/pip install -i https://pypi.org/simple \
    onnxruntime-gpu==1.23.2 opencv-python-headless shapely \
    pyclipper loguru colorlog requests pydantic pillow tqdm

# 1. 测 OnnxOCR 侧（工作目录放在 bench/，脚本按相对路径找检出目录）
python compare_onnxocr_side.py          # -> bench_onnxocr.json

# 2. 测 NOOCR 侧（本项目的 GPU 环境）
python bench_noocr.py ppocrv6-small cuda # -> bench_noocr_ppocrv6_small.json
python bench_noocr.py ppocrv6-tiny  cuda # -> bench_noocr_ppocrv6_tiny.json

# 3. 合成表格
python compare_make_table.py             # -> compare_table.md
```

## 统计口径

两侧完全一致，缺一条对比就不成立：

| 口径 | 取值 | 原因 |
|---|---|---|
| 图集 | `images/` 下同样8 张 | 变量只剩引擎本身 |
| 预热 | 2 轮，不计入 | 含 CUDA kernel 编译与首次内存分配 |
| 测量 | 5 轮取中位数 | 单次抖动可到 ±20%，均值会被带偏 |
| 计时范围 | 图片解码 + 推理 | 只算纯推理反而不真实 |
| 方向分类 | 两侧都开 | OnnxOCR 默认关，不开等于让它少做一步 |
| 设备 | 同一块 GPU | — |

## 输出

- `bench_onnxocr.json` / `bench_noocr_*.json` —— 逐图明细与总统计
- `compare_table.md` —— 表格
- `compare_summary.json` —— 对账用，便于以后复核