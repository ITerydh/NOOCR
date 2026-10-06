// 检查 WebUI 的「识别后端」下拉框实际渲染了几个选项。
// 用途：后端注册表加了新档位后，光看 /api/backends 返回值不够——
// 还要确认页面上的 <select> 真的把它渲染出来了。
// 用法: node scripts/check_backend_options.js [base_url]
const { chromium } = require("playwright-core");

const EXE = "C:/Users/iterhui/AppData/Local/ms-playwright/chromium-1129/chrome-win/chrome.exe";
const BASE = process.argv[2] || "http://127.0.0.1:8933";

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await page.goto(BASE, { waitUntil: "networkidle" });
  // 等 <select> 自身可见，而不是等 <option>——折叠状态下 option
  // 的可见性判定不成立，会一直等到超时。
  await page.waitForFunction(
    () => document.querySelectorAll("#backend option").length > 0,
    null, { timeout: 15000 });

  const opts = await page.$$eval("#backend option", (os) =>
    os.map((o) => ({ value: o.value, text: o.textContent.trim() })));
  const selected = await page.$eval("#backend", (s) => s.value);
  const width = await page.$eval("#backend", (s) => s.getBoundingClientRect().width);

  console.log(`下拉框可见宽度 ${width.toFixed(0)}px`);
  console.log(`选项数 ${opts.length}，当前选中 ${selected}\n`);
  for (const o of opts) {
    console.log(`  ${o.value === selected ? ">" : " "} ${o.value.padEnd(18)} ${o.text}`);
  }
  if (errors.length) {
    console.log("\n页面报错:");
    errors.forEach((e) => console.log("  " + e));
  }
  await browser.close();

  const names = opts.map((o) => o.value);
  const want = ["ppocrv5", "ppocrv6-tiny", "ppocrv6-small", "ppocrv6-medium"];
  const missing = want.filter((w) => !names.includes(w));
  if (missing.length) {
    console.log(`\n缺档位: ${missing.join(", ")}`);
    process.exit(1);
  }
  console.log("\n四档齐全");
})();