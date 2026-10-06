"""让 pytest 忽略 tests/ 下的脚本。

``tests/`` 里是**可直接执行的脚本**，不是 pytest 用例：导入即跑完整逻辑，
结尾 ``sys.exit()``。pytest 在收集阶段执行模块导入，``SystemExit`` 会直接
打断它并报 ``INTERNALERROR> SystemExit``——看起来像测试失败，实际是收集
失败，一个用例都没跑。

所以无论谁误敲了 ``pytest tests``，都明确排除而不是让它崩。
真要跑测试请直接执行脚本：

    python tests/test_units.py
    python tests/test_multilang.py
    python tests/test_rotate.py

CI 也是这样调的（见 ``.github/workflows/ci.yml``），项目不依赖 pytest。
"""

collect_ignore_glob = ["test_*.py"]
