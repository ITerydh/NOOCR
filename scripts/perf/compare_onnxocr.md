# 与 OnnxOCR 的性能对比

复现 README「与 OnnxOCR 实测对比」一节的数据。覆盖四个档位：
`ppocrv5` / `ppocrv6-tiny` / `ppocrv6-small` / `ppocrv6-medium`。

## 为什么拆成三个脚本

两个引擎的依赖**不能装在同一个环境里**——OnnxOCR 需要
`numpy<2.0` 与自己那套依赖树，和本项目的 `onnxruntime-gpu`
版本要求冲突（新版 ORT 要 CUDA 13，旧版要 CUDA 12）。

## 前置：CUDA 必须真生效

两个环境都要能真正启用 CUDA。**这里有个容易踩的坑**：
Windows 上光改 `PATH` 不够，必须 `os.add_dll_directory`
登记，ORT 才能加载 provider 桥接 dll。少了这步 CUDA 会
**静默初始化失败并回退 CPU**——不报错，照样输出耗时，
只是那个数字毫无意义。

判据是看 `get_providers()` 而不是看日志：

```python
print(session.get_providers())
# 必须是 ['CUDAExecutionProvider', ...]
# 只有 ['CPUExecutionProvider'] 就是没生效
```

## 准备共用权重目录

**这是整张表的地基。** 四个档位两侧必须读同一批权重文件。

v5 好办：两个仓库各自带一份 ppocrv5，det / rec / cls 三个 onnx
经md5 校验完全一致，字典仅换行符不同。

v6 麻烦：本项目与 OnnxOCR 各自导出的 v6 det 图**不同源**——我们的那份
带 `p2o.pd_op.*` 图优化痕迹，文件大小差 0.5%，无法证明逐字节相同。
字典内容一致（18710 / 6906 词条），rec 的图结构也一致（tiny 的标识符
序列相似度 1.0000，small 0.9448，差异只是节点编号偏移），但只要det
不能证明，整张表就站不住。

解法是准备一个共用目录，两侧都指向它：

```bash
# OnnxOCR 的 v6 权重在 ppocrv6 分支上，默认分支只有 v5
git clone --depth 1 --branch ppocrv6 \
    https://github.com/jingsongliujing/OnnxOCR.git

# 复制成本项目的 models/ 布局
S=OnnxOCR/onnxocr/models
D=weights_shared
mkdir -p $D/ppocrv5/{det,rec,cls} $D/ppocrv6/{det,rec,cls}
cp $S/ppocrv5/det/det.onnx        $D/ppocrv5/det/
cp $S/ppocrv5/rec/rec.onnx        $D/ppocrv5/rec/
cp $S/ppocrv5/cls/cls.onnx        $D/ppocrv5/cls/
cp $S/ppocrv5/ppocrv5_dict.txt    $D/ppocrv5/
cp $S/ppocrv5/cls/cls.onnx        $D/ppocrv6/cls/     # v6 复用同一份 cls
for t in tiny small medium; do
  cp $S/ppocrv6/$t/det/det.onnx   $D/ppocrv6/det/PP-OCRv6_det_$t.onnx
  cp $S/ppocrv6/$t/rec/rec.onnx   $D/ppocrv6/rec/PP-OCRv6_rec_$t.onnx
done
cp $S/ppocrv6/ppocrv6_dict.txt      $D/ppocrv6/
cp $S/ppocrv6/ppocrv6_tiny_dict.txt $D/ppocrv6/
```

`compare_make_table.py` 会逐个校验这些文件与源文件 md5 一致，
不一致直接抛错。

## 步骤

```bash
# 0. 另建 OnnxOCR 依赖环境
python -m venv onnxocr-env
./onnxocr-env/Scripts/pip install -i https://pypi.org/simple \
    onnxruntime-gpu==1.23.2 opencv-python-headless shapely \
    pyclipper loguru colorlog requests pydantic pillow tqdm \
    nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12

# 1. 测 OnnxOCR 侧（四档）
for t in ppocrv5 ppocrv6-tiny ppocrv6-small ppocrv6-medium; do
  ./onnxocr-env/Scripts/python compare_onnxocr_side.py $t
done

# 2. 测 NOOCR 侧（四档，指向同一份权重）
export NOOCR_MODELS_DIR=$PWD/weights_shared
export BENCH_SUFFIX=_onnxocrw
for t in ppocrv5 ppocrv6-tiny ppocrv6-small ppocrv6-medium; do
  python bench_noocr.py $t cuda
done

# 3. 合成表格
python compare_make_table.py
```

## 统计口径

| 口径 | 取值 | 原因 |
|---|---|---|
| 图集 | `images/` 下同样 8 张 | 变量只剩引擎本身 |
| 权重 | `weights_shared/` 共用目录 | 两侧读同一批文件 |
| 预热 | 1 轮，不计入 | 含CUDA kernel 编译与首次内存分配 |
| 测量 | 10 轮取**均值** | 见下|
| 计时范围 | 图片解码 + 推理 | 只算纯推理反而不真实 |
| 方向分类 | 两侧都开 | OnnxOCR 默认关，不开等于让它少做一步 |
| 设备 | 同一块 GPU | — |

### 为什么取均值而不是中位数

本机实测单次抖动可到 **±40%**——v5 的 `doc_comparison_table`
首样本 4951ms、末样本 10135ms。中位数会把这种方差藏起来，
看上去很稳，实则掩盖了真实情况。轮数拉到 10 轮取均值，
均值的抖动收敛到 5% 以内，反映的是真实期望耗时。

### 为什么只预热 1 轮

预热的目的是让CUDA kernel 完成编译、内存池完成首次分配。
1 轮足够达成这个目的；多轮预热只会把机器状态的波动一并抹平，
让后续测量失真。

## 一个反直觉的观察

**模型越大，NOOCR 的优势越明显**（v5 4.25x→ medium 6.70x）。
原因是 OnnxOCR 对所有模型都按最大配置跑，不做输入分辨率与
arena 的取舍；而 NOOCR 会按形状特征选分辨率、并对 det 关掉
`dynamic_shape` 让 arena 生效（det 提速约 32%）。模型越大，
这笔"省下来的"越多。