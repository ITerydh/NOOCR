"""多语言与多场景 OCR 能力验证。

检查两件事：
1. 实际识别出的文本是否合理（不是乱码/空白）
2. 不同 rec_batch_size 下结果是否**完全一致**（可复现性契约）
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noocr.backends.ppocr import PPOCRBackend
from noocr.engine.imageops import imread

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "noocr/web/static"

#: 覆盖各类真实场景。每个文件名都对应图片的实际内容（逐张读图确认）。
CASES = [
    ("语文试卷", "exam_chinese_primary.jpg"),
    ("数学试卷", "exam_math_primary.jpg"),
    ("化验单", "medical_lab_report.jpg"),
    ("化验单倾斜", "medical_lab_report_tilted.jpg"),
    ("财报表格", "finance_shareholder_table.jpg"),
    ("银行卡凭条", "receipt_bank_statement.jpg"),
    ("火车票", "ticket_train.jpg"),
    ("机票行程单", "ticket_flight_itinerary.jpg"),
    ("身份证", "id_card_china.jpg"),
    ("洗标", "receipt_clothing_care_label.jpg"),
    ("产品说明", "product_spec_sheet.jpg"),
    ("银行门头", "scene_bank_branch.jpg"),
    ("路牌遮挡", "scene_road_sign_occluded.jpg"),
    ("竖排牌匾", "scene_vertical_plaque.jpg"),
    ("双栏论文", "doc_twocolumn_paper_highlight.jpg"),
    ("英文论文", "doc_english_single_column.jpg"),
    ("对比表格", "doc_comparison_table.jpg"),
    ("日文海报", "lang_japanese_poster.jpg"),
    ("日文词云", "lang_japanese_wordcloud.jpg"),
    ("韩文图文", "lang_korean_quote_art.jpg"),
    ("法文路牌", "lang_french_signs.jpg"),
    ("德文标牌", "lang_german_village_sign.jpg"),
    ("德文漫画", "lang_german_proverb_cartoon.jpg"),
    ("车牌", "plate_vehicle_blue.jpg"),
    ("微信支付", "payment_wechat.jpg"),
    ("支付宝", "payment_alipay.jpg"),
    ("微信群码", "qrcode_wechat_group.jpg"),
    ("QQ群码", "qrcode_qq_group.jpg"),
    ("个人名片码", "qrcode_personal_card.jpg"),
]

backend = PPOCRBackend(device="cpu", rec_batch_size=6, use_cls=True)
backend.load()

print(f"{'用例':14s} {'尺寸':>11s} {'行数':>5s} {'平均置信':>8s} {'耗时ms':>8s}  样例")
print("-" * 96)

results = {}
for label, fname in CASES:
    p = STATIC / fname
    if not p.is_file():
        print(f"{label:14s} 跳过（缺文件 {fname}）")
        continue
    img = imread(p)
    if img is None:
        print(f"{label:14s} 跳过（无法解码）")
        continue
    t = time.perf_counter()
    page = backend.recognize_image(img)
    dt = (time.perf_counter() - t) * 1000
    lines = [ln for ln in page.lines if ln.text.strip()]
    avg = sum(ln.confidence for ln in lines) / len(lines) if lines else 0.0
    sample = " | ".join(ln.text for ln in lines[:3])[:40]
    results[label] = (img.shape[1], img.shape[0], [(ln.text, ln.confidence) for ln in page.lines])
    print(f"{label:14s} {img.shape[1]:5d}x{img.shape[0]:<5d} {len(page.lines):5d} "
          f"{avg:8.3f} {dt:8.0f}  {sample}")

print("\n=== 可复现性契约：不同 rec_batch_size 必须给出完全相同的结果 ===")
print(f"{'用例':14s} {'bs=1 vs 6':>12s} {'bs=6 vs 32':>12s} {'行数':>6s}")
print("-" * 50)
all_ok = True
for label, fname in CASES:
    p = STATIC / fname
    if not p.is_file() or label not in results:
        continue
    img = imread(p)
    outs = {}
    for bs in (1, 6, 32):
        be = PPOCRBackend(device="cpu", rec_batch_size=bs, use_cls=True)
        be.load()
        outs[bs] = [(ln.text, round(ln.confidence, 4),
                     tuple(round(c, 1) for pt in ln.box.points for c in pt))
                    for ln in be.recognize_image(img).lines]
    ok1 = outs[1] == outs[6]
    ok2 = outs[6] == outs[32]
    if not (ok1 and ok2):
        all_ok = False
        for i, (a, b_) in enumerate(zip(outs[6], outs[1])):
            if a != b_:
                print(f"    {label} 首个差异 @{i}: bs6={a[:2]} bs1={b_[:2]}")
                break
    print(f"{label:14s} {'一致' if ok1 else '不一致 ✗':>12s} "
          f"{'一致' if ok2 else '不一致 ✗':>12s} {len(outs[6]):6d}")

print("\n" + "=" * 60)
print(f"可复现性: {'全部通过 ✓' if all_ok else '存在不一致 ✗'}")
sys.exit(0 if all_ok else 1)