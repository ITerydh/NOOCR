"""按图片内容重命名测试图。每条映射都经过实际读图确认。"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "noocr/web/static"

# (旧名, 新名, 内容说明)—— 说明来自逐张读图确认，非猜测
MAPPING = [
    ("00009282.jpg", "receipt_clothing_care_label.jpg", "衣物洗涤说明标签"),
    ("00015504.jpg", "finance_shareholder_table.jpg", "财报前十大股东表"),
    ("00018069.jpg", "medical_lab_report.jpg", "医学化验单"),
    ("00056221.jpg", "ticket_train.jpg", "纸质火车票"),
    ("00057937.jpg", "medical_lab_report_tilted.jpg", "倾斜拍摄的化验单"),
    ("00059985.jpg", "exam_math_primary.jpg", "小学数学单元测试题"),
    ("00111002.jpg", "receipt_bank_statement.jpg", "银行卡交易凭条"),
    ("00207393.jpg", "scene_bank_branch.jpg", "银行门头招牌"),
    ("12.jpg", "scene_road_sign_occluded.jpg", "户外指示牌（花草遮挡）"),
    ("8f113149-ff64-4c9f-8dc0-e34100365aa4.jpg", "id_card_china.jpg", "居民身份证"),
    ("32e0869f54edcf90cc8e93b981f7235.jpg", "qrcode_personal_card.jpg", "个人二维码名片"),
    ("aircard.jpg", "ticket_flight_itinerary.jpg", "机票行程单"),
    ("test1.jpg", "scene_vertical_plaque.jpg", "竖排文字牌匾"),
    ("test2.jpg", "product_spec_sheet.jpg", "产品参数说明图"),
    ("french_0.jpg", "lang_french_signs.jpg", "法文指示牌（艺术字体）"),
    ("ger_1.jpg", "lang_german_village_sign.jpg", "德文村庄标牌"),
    ("ger_2.jpg", "lang_german_proverb_cartoon.jpg", "德文谚语漫画"),
    ("japan_1.jpg", "lang_japanese_poster.jpg", "日文海报"),
    ("japan_2.jpg", "lang_japanese_wordcloud.jpg", "日文词云（字号跨度大）"),
    ("korean_1.jpg", "lang_korean_quote_art.jpg", "韩文励志图文"),
    ("layout_cdla.jpg", "doc_twocolumn_paper_highlight.jpg", "双栏论文（含高亮标注）"),
    ("layout_publaynet.jpg", "doc_english_single_column.jpg", "英文单栏论文"),
    ("table.jpg", "doc_comparison_table.jpg", "英文论文对比表"),
    ("license_plate_single_blue.jpg", "plate_vehicle_blue.jpg", "蓝色车牌"),
    ("myQR.jpg", "qrcode_generic.jpg", "通用二维码"),
    ("weixin_pay.jpg", "payment_wechat.jpg", "微信支付收款码"),
    ("zhifubao_pay.jpg", "payment_alipay.jpg", "支付宝收款码"),
    ("QQ群.jpg", "qrcode_qq_group.jpg", "QQ 群二维码"),
    ("微信群.jpg", "qrcode_wechat_group.jpg", "微信群二维码"),
]

renamed = missing = skipped = 0
for old, new, desc in MAPPING:
    src, dst = STATIC / old, STATIC / new
    if not src.is_file():
        print(f"缺失 {old}")
        missing += 1
        continue
    if src == dst:
        skipped += 1
        continue
    if dst.exists():
        print(f"目标已存在，跳过: {new}")
        skipped += 1
        continue
    src.rename(dst)
    print(f"✓ {old:46s} -> {new:38s} # {desc}")
    renamed += 1

print(f"\n重命名 {renamed} 张，跳过 {skipped}，缺失 {missing}")
print(f"目录现有{len(list(STATIC.glob('*.jpg')))} 张")