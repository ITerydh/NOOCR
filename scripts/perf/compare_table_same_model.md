### PP-OCRv5

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv5 | NOOCR<br>PP-OCRv5 | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 3691 ms | **453 ms** | **8.2x** | 82 / 79 |
| `exam_chinese_primary` | 1920x2560 | 3259 ms | **1062 ms** | **3.1x** | 68 / 73 |
| `id_card_china` | 1148x672 | 551 ms | **205 ms** | **2.7x** | 9 / 10 |
| `medical_lab_report` | 430x267 | 2512 ms | **309 ms** | **8.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 778 ms | **213 ms** | **3.6x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1376 ms | **287 ms** | **4.8x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 303 ms | **111 ms** | **2.7x** | 2 / 2 |
| `ticket_train` | 670x510 | 961 ms | **522 ms** | **1.8x** | 19 / 17 |
| **均值** | — | **1679 ms** | **395 ms** | **4.25x** | 296 / 296 |
| **合计** | — | **13432 ms** | **3162 ms** | **4.25x** | — |

### PP-OCRv6 tiny

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv6 tiny | NOOCR<br>PP-OCRv6 tiny | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 1838 ms | **311 ms** | **5.9x** | 73 / 74 |
| `exam_chinese_primary` | 1920x2560 | 2210 ms | **477 ms** | **4.6x** | 70 / 75 |
| `id_card_china` | 1148x672 | 374 ms | **171 ms** | **2.2x** | 9 / 9 |
| `medical_lab_report` | 430x267 | 1729 ms | **264 ms** | **6.5x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 527 ms | **146 ms** | **3.6x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 932 ms | **257 ms** | **3.6x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 191 ms | **64 ms** | **3.0x** | 2 / 2 |
| `ticket_train` | 670x510 | 631 ms | **188 ms** | **3.4x** | 19 / 20 |
| **均值** | — | **1054 ms** | **235 ms** | **4.49x** | 289 / 295 |
| **合计** | — | **8433 ms** | **1878 ms** | **4.49x** | — |

### PP-OCRv6 small

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv6 small | NOOCR<br>PP-OCRv6 small | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 2786 ms | **457 ms** | **6.1x** | 71 / 73 |
| `exam_chinese_primary` | 1920x2560 | 2964 ms | **921 ms** | **3.2x** | 71 / 65 |
| `id_card_china` | 1148x672 | 555 ms | **257 ms** | **2.2x** | 10 / 11 |
| `medical_lab_report` | 430x267 | 2501 ms | **363 ms** | **6.9x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 768 ms | **266 ms** | **2.9x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 1345 ms | **352 ms** | **3.8x** | 31 / 30 |
| `scene_vertical_plaque` | 720x1150 | 308 ms | **100 ms** | **3.1x** | 2 / 2 |
| `ticket_train` | 670x510 | 949 ms | **300 ms** | **3.2x** | 19 / 19 |
| **均值** | — | **1522 ms** | **377 ms** | **4.04x** | 289 / 285 |
| **合计** | — | **12177 ms** | **3016 ms** | **4.04x** | — |

### PP-OCRv6 medium

| 示例图 | 分辨率 | OnnxOCR<br>PP-OCRv6 medium | NOOCR<br>PP-OCRv6 medium | 倍数 | 文本行 |
|---|---|---|---|---|---|
| `doc_comparison_table` | 371x293 | 4535 ms | **478 ms** | **9.5x** | 74 / 73 |
| `exam_chinese_primary` | 1920x2560 | 5175 ms | **729 ms** | **7.1x** | 64 / 66 |
| `id_card_china` | 1148x672 | 1003 ms | **245 ms** | **4.1x** | 12 / 11 |
| `medical_lab_report` | 430x267 | 4202 ms | **462 ms** | **9.1x** | 69 / 69 |
| `product_spec_sheet` | 500x500 | 1326 ms | **299 ms** | **4.4x** | 16 / 16 |
| `receipt_bank_statement` | 500x667 | 2304 ms | **417 ms** | **5.5x** | 32 / 32 |
| `scene_vertical_plaque` | 720x1150 | 634 ms | **115 ms** | **5.5x** | 2 / 2 |
| `ticket_train` | 670x510 | 1669 ms | **366 ms** | **4.6x** | 19 / 19 |
| **均值** | — | **2606 ms** | **389 ms** | **6.70x** | 288 / 288 |
| **合计** | — | **20847 ms** | **3112 ms** | **6.70x** | — |
