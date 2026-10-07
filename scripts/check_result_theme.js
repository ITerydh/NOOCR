// 识别后才出现的元素（导出栏、识别文本、记录条目）的明暗主题检查。
// 为什么单独一个脚本：check_theme.js 跑在「刚打开、还没识别」的状态，
// 而 .out / .out-bar button / .hist-item 只有识别之后才进 DOM——
// 上一轮漏掉的主题问题恰好全在这一类（写死深色但浅色没覆盖）。
// 用法: node scripts/check_result_theme.js [base_url]
const { chromium } = require("playwright-core");

const EXE = "C:/Users/iterhui/AppData/Local/ms-playwright/chromium-1129/chrome-win/chrome.exe";
const BASE = process.argv[2] || "http://127.0.0.1:8940";
const STATIC = "C:/Users/iterhui/Desktop/ocr/noocr/web/static";

const READ = () => {
  const g = (sel, prop = "color") => {
    const e = document.querySelector(sel);
    return e ? getComputedStyle(e)[prop] : "(不存在)";
  };
  const bg = (sel) => {
    const e = document.querySelector(sel);
    if (!e) return "(不存在)";
    const s = getComputedStyle(e);
    return s.backgroundColor !== "rgba(0, 0, 0, 0)"
      ? s.backgroundColor : `image:${s.backgroundImage.slice(0, 46)}`;
  };
  return {
    out: g(".out"),
    outBarBtnBg: bg(".out-bar button"),
    outBarBtnColor: g(".out-bar button"),
    histItemNm: g(".hist-item .nm"),
    histItemMeta: g(".hist-item .meta"),
    histItemHoverBg: bg(".hist-item"),
    detailThBg: bg(".detail thead th"),
    // 明细表里的置信度色块要跟着主题走，否则深色下看不清高低置信
    confBar: (() => {
      const e = document.querySelector(".detail .cf, .detail .conf, .detail td .bar");
      return e ? getComputedStyle(e).background : "(无置信条)";
    })(),
  };
};

/* 相对亮度（WCAG 定义）。用来判断「文字压在自己的底色上」是否可读：
   只看色值不看对比度的话，深底深字这种组合会一路放行。 */
function lum(rgb) {
  const m = String(rgb).match(/(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)/);
  if (!m) return null;
  const [r, g, b] = m.slice(1).map((x) => {
    const v = Number(x) / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function ratio(a, b) {
  const la = lum(a); const lb = lum(b);
  if (la === null || lb === null) return null;
  const [hi, lo] = la > lb ? [la, lb] : [lb, la];
  return (hi + 0.05) / (lo + 0.05);
}

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  const snaps = [];
  for (const theme of ["light", "dark"]) {
    await page.goto(BASE, { waitUntil: "networkidle" });
    await page.evaluate((t) => {
      localStorage.setItem("noocr.theme", t);
      document.documentElement.setAttribute("data-theme", t);
    }, theme);
    await page.waitForFunction(
      () => document.querySelectorAll("#backend option").length > 0,
      null, { timeout: 15000 });

    // 切到「文本」结果页：导出栏按钮只在这个页签里
    await page.setInputFiles("#file", `${STATIC}/id_card_china.jpg`);
    await page.$eval("#run", (el) => el.click());
    await page.waitForFunction(
      () => /行/.test(document.querySelector("#stats")?.textContent || ""),
      null, { timeout: 180000 });
    await page.waitForTimeout(600);
    await page.screenshot({ path: `C:/Users/iterhui/Desktop/ocr/.tmp_res_${theme}.png` });
    snaps.push({ theme, ...(await page.evaluate(READ)) });
  }

  const keys = Object.keys(snaps[0]).filter((k) => k !== "theme");
  console.log("字段".padEnd(16) + "浅色".padEnd(26) + "深色");
  for (const k of keys) {
    console.log(k.padEnd(16)
      + String(snaps[0][k]).padEnd(26) + String(snaps[1][k]));
  }

  await browser.close();

  const problems = [];
  const [L, D] = snaps;
  // 这些元素的主题态必须不同
  for (const k of ["outBarBtnBg", "histItemNm", "histItemMeta", "out"]) {
    if (L[k] === D[k]) problems.push(`${k} 不随主题变化（都是 ${L[k]}）`);
  }
  for (const [name, s] of [["浅色", L], ["深色", D]]) {
    for (const k of keys) {
      if (s[k] === "(不存在)") problems.push(`${name}下${k}未找到，检查项失效`);
    }
  }
  // 对比度：导出栏按钮文字压自己的底色
  for (const [name, s] of [["浅色", L], ["深色", D]]) {
    const r = ratio(s.outBarBtnColor, s.outBarBtnBg);
    if (r !== null && r < 4.5) {
      problems.push(`${name}下导出栏按钮对比度仅 ${r.toFixed(1)}:1（需 >=4.5）`);
    }
  }
  if (errors.length) problems.push(`页面报错 ${errors.length} 条`);

  if (problems.length) {
    console.log("\n发现问题:");
    problems.forEach((p) => console.log("  - " + p));
    process.exit(1);
  }
  console.log("\n识别后元素配色正常：两主题不同、对比度达标、元素均存在");
})();
