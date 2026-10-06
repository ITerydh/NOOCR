### PP-OCRv5

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv5 | NOOCR<br>PP-OCRv5 | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 2913 ms | **349 ms** | **8.3x** | 82 / 79 |
| `exam_chinese_primary` | 1920x2560 | 3204 ms | **558 ms** | **5.7x** | 68 / 73 |
| `id_card_china` | 1148x672 | 558 ms | **201 ms** | **2.8x** | 9 / 10 |
| `medical_lab_report` | 430x267 | 2484 ms | **297 ms** | **8.4x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 774 ms | **206 ms** | **3.8x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1370 ms | **278 ms** | **4.9x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 299 ms | **72 ms** | **4.2x** | 2 / 2 |
| `ticket_train` | 670x510 | 956 ms | **213 ms** | **4.5x** | 19 / 17 |
| **均值** | — | **1570 ms** | **272 ms** | **5.78x** | 296 / 296 |
| **合计** | — | **12558 ms** | **2173 ms** | **5.78x** | — |

### PP-OCRv6 tiny

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv6 tiny | NOOCR<br>PP-OCRv6 tiny | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 1851 ms | **303 ms** | **6.1x** | 73 / 74 |
| `exam_chinese_primary` | 1920x2560 | 2206 ms | **458 ms** | **4.8x** | 70 / 75 |
| `id_card_china` | 1148x672 | 390 ms | **160 ms** | **2.4x** | 9 / 9 |
| `medical_lab_report` | 430x267 | 1721 ms | **244 ms** | **7.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 542 ms | **138 ms** | **3.9x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 952 ms | **235 ms** | **4.1x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 196 ms | **63 ms** | **3.1x** | 2 / 2 |
| `ticket_train` | 670x510 | 657 ms | **172 ms** | **3.8x** | 19 / 20 |
| **均值** | — | **1064 ms** | **222 ms** | **4.80x** | 289 / 295 |
| **合计** | — | **8514 ms** | **1773 ms** | **4.80x** | — |

### PP-OCRv6 small

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv6 small | NOOCR<br>PP-OCRv6 small | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 2661 ms | **437 ms** | **6.1x** | 71 / 73 |
| `exam_chinese_primary` | 1920x2560 | 2930 ms | **696 ms** | **4.2x** | 71 / 65 |
| `id_card_china` | 1148x672 | 567 ms | **245 ms** | **2.3x** | 10 / 11 |
| `medical_lab_report` | 430x267 | 2463 ms | **353 ms** | **7.0x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 750 ms | **254 ms** | **3.0x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1316 ms | **341 ms** | **3.9x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 292 ms | **95 ms** | **3.1x** | 2 / 2 |
| `ticket_train` | 670x510 | 928 ms | **288 ms** | **3.2x** | 19 / 19 |
| **均值** | — | **1488 ms** | **339 ms** | **4.39x** | 289 / 285 |
| **合计** | — | **11908 ms** | **2710 ms** | **4.39x** | — |

### PP-OCRv6 medium

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv6 medium | NOOCR<br>PP-OCRv6 medium | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 4343 ms | **468 ms** | **9.3x** | 74 / 73 |
| `exam_chinese_primary` | 1920x2560 | 4791 ms | **713 ms** | **6.7x** | 64 / 66 |
| `id_card_china` | 1148x672 | 956 ms | **240 ms** | **4.0x** | 12 / 11 |
| `medical_lab_report` | 430x267 | 4050 ms | **447 ms** | **9.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 1286 ms | **290 ms** | **4.4x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 2227 ms | **386 ms** | **5.8x** | 32 / 32 |
| `scene_vertical_plaque` | 720x1150 | 603 ms | **110 ms** | **5.5x** | 2 / 2 |
| `ticket_train` | 670x510 | 1621 ms | **338 ms** | **4.8x** | 19 / 19 |
| **均值** | — | **2485 ms** | **374 ms** | **6.64x** | 288 / 288 |
| **合计** | — | **19877 ms** | **2993 ms** | **6.64x** | — |
