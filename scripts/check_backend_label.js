// 跑一次真实识别，验证统计条与历史记录里的后端短标签。
// 关注点：shortBackend 不再依赖硬编码映射，新增档位也能显示对。
// 用法: node scripts/check_backend_label.js [base_url]
const { chromium } = require("playwright-core");

const EXE = "C:/Users/iterhui/AppData/Local/ms-playwright/chromium-1129/chrome-win/chrome.exe";
const BASE = process.argv[2] || "http://127.0.0.1:8934";
const STATIC = "C:/Users/iterhui/Desktop/ocr/noocr/web/static";

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage();
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForFunction(
    () => document.querySelectorAll("#backend option").length > 0,
    null, { timeout: 15000 });

  // 先直接验标签推导：四档都必须给出短标签，且没有半截名字
  const derived = await page.evaluate(() => {
    const out = {};
    for (const o of document.querySelectorAll("#backend option")) {
      out[o.value] = shortBackend(o.value);
    }
    return out;
  });
  console.log("短标签推导：");
  for (const [k, v] of Object.entries(derived)) console.log(`  ${k.padEnd(18)} -> ${v}`);

  // 再用 medium 真跑一张图，看统计条里显示什么
  await page.selectOption("#backend", "ppocrv6-medium");
  await page.setInputFiles("#file", `${STATIC}/id_card_china.jpg`);
  await page.click("#run");
  await page.waitForFunction(
    () => /行/.test(document.querySelector("#stats")?.textContent || ""),
    null, { timeout: 180000 });
  await page.waitForTimeout(800);

  const statText = await page.$eval("#stats", (e) => e.textContent.replace(/\s+/g, " ").trim());
  const histText = await page.evaluate(() => {
    const h = document.querySelector("#histBody");
    return h ? h.textContent.replace(/\s+/g, " ").trim() : "";
  });
  console.log(`\n统计条: ${statText.slice(0, 150)}`);
  console.log(`\n记录区: ${histText.slice(0, 150)}`);

  const bad = Object.entries(derived).filter(([, v]) => /^v\d/.test(v) === false);
  await page.screenshot({ path: "C:/Users/iterhui/Desktop/ocr/.tmp_medium_webui.png", fullPage: false });
  await browser.close();

  if (bad.length) {
    console.log(`\n短标签异常: ${JSON.stringify(bad)}`);
    process.exit(1);
  }
  if (!statText.includes("v6 medium")) {
    console.log("\n统计条未显示 'v6 medium'");
    process.exit(1);
  }
  console.log("\n统计条正确显示 v6 medium");
})();