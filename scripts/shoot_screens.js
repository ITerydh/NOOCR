// 生成 README 图例：驱动真实 WebUI 走完整流程后截图。
// 用法: node scripts/shoot_screens.js [out_dir] [base_url]
// 依赖 playwright-core；产出的 PNG 需再经 scripts/shrink_images.py 压缩
const { chromium } = require("playwright-core");
const path = require("path");
const fs = require("fs");

const EXE = "C:/Users/iterhui/AppData/Local/ms-playwright/chromium-1129/chrome-win/chrome.exe";
const OUT = process.argv[2] || "C:/Users/iterhui/Desktop/ocr/docs/images";
const BASE = process.argv[3] || "http://127.0.0.1:8820";
const STATIC = "C:/Users/iterhui/Desktop/ocr/noocr/web/static";

// 用上传而非点示例图：走 setFile() 才会渲染上传缩略图，
// 这是「上传 → 缩略图 → 识别 → 对照 → 明细联动」完整链路，
// 点示例图只覆盖后半段，与 demo.png 的展示内容高度重合。
const CASES = [
  {
    file: "webui-result.png",
    sample: "doc_comparison_table.jpg",
    label: "图文对照双列 + 上传缩略图",
    theme: "dark", w: 1600, h: 980,
  },
  {
    file: "webui-detail.png",
    sample: "doc_twocolumn_paper_highlight.jpg",
    label: "点明细行→ 图上高亮联动",
    theme: "dark", w: 1600, h: 980,
    // 截完前点一行明细，验证图上高亮框确实跟着走
    clickRow: 2,
  },
  {
    file: "webui-light.png",
    sample: "receipt_bank_statement.jpg",
    label: "浅色主题",
    theme: "light", w: 1600, h: 980,
  },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitDone(page) {
  await page.waitForFunction(
    () => {
      const st = document.getElementById("stats");
      const s = document.getElementById("status");
      return st && st.querySelectorAll("div").length > 3 &&
             s && !/识别中|处理中|busy/.test(s.textContent);
    },
    { timeout: 90000 }
  );
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ executablePath: EXE });
  const errors = [];

  for (const c of CASES) {
    const ctx = await browser.newContext({
      viewport: { width: c.w, height: c.h },
      deviceScaleFactor: 1,
    });
    const page = await ctx.newPage();
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(`${c.file}: ${m.text()}`);
    });
    page.on("pageerror", (e) => errors.push(`${c.file}: ${e.message}`));

    await page.goto(BASE, { waitUntil: "networkidle" });
    await page.evaluate(() => localStorage.clear());
    await page.reload({ waitUntil: "networkidle" });

    if (c.theme === "light") {
      await page.evaluate(() =>
        document.documentElement.setAttribute("data-theme", "light"));
    }

    // 走真实上传链路
    await page.setInputFiles("#file", path.join(STATIC, c.sample));
    await waitDone(page);
    await sleep(1500); // 等缩略图与文本框渲染完

    if (c.clickRow) {
      await page.click(`#tBody2 tr:nth-child(${c.clickRow})`);
      await sleep(700);
    }

    const out = path.join(OUT, c.file);
    await page.screenshot({ path: out });
    console.log("saved", c.file, "->", c.label);
    await ctx.close();
  }

  await browser.close();
  if (errors.length) {
    console.log("\n控制台报错:");
    errors.forEach((e) => console.log("  " + e));
    process.exitCode = 1;
  } else {
    console.log("\n控制台零报错");
  }
})().catch((e) => {
  console.error("FAILED:", e.message);
  process.exit(1);
});