// 检查顶栏设备下拉（.devpop）在明暗两个主题下的实际配色。
// 关注点：这个弹层曾被硬编码成深色 rgba(18,25,37,.97)，
// 于是浅色主题下弹出来是一块深色板——变量全对，只有这一个元素没跟上。
// 用法: node scripts/check_device_popup.js [base_url]
const { loadPlaywright, findChromium, P } = require("./_browser");
const { chromium } = loadPlaywright();

const EXE = findChromium();
const BASE = process.argv[2] || "http://127.0.0.1:8940";

/* 弹层里每一类文字都要单独取。只看容器背景会漏掉「容器对了但
   里面某行文字仍是深色」——那行字在深底上照样看不见。 */
const READ = () => {
  const pop = document.querySelector("#devPop");
  if (!pop) return { err: "找不到 #devPop" };
  if (pop.classList.contains("hide")) return { err: "弹层处于隐藏态" };
  const cs = (sel) => {
    const e = pop.querySelector(sel);
    return e ? getComputedStyle(e) : null;
  };
  const px = (el, prop) => (el ? getComputedStyle(el)[prop] : "(无)");
  const popCs = getComputedStyle(pop);
  const btn = pop.querySelector("button");
  const on = pop.querySelector("button.on") || btn;
  const tag = on ? on.querySelector(".tag") : null;
  const hint = on ? on.querySelector(".hint") : null;
  const hd = pop.querySelector(".hd");
  return {
    theme: document.documentElement.getAttribute("data-theme") || "(跟随系统)",
    popBg: popCs.backgroundColor,
    popBorder: popCs.borderTopColor,
    popColor: popCs.color,
    btnColor: px(btn, "color"),
    onBg: on ? getComputedStyle(on).backgroundColor : "(无选中项)",
    tagColor: px(tag, "color"),
    hintColor: px(hint, "color"),
    hdColor: px(hd, "color"),
    buttons: pop.querySelectorAll("button").length,
  };
};

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));

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
    await page.waitForTimeout(300);
    // 打开设备弹层。用 $eval 而非 click：重构后按钮在常驻底栏样式层里，
    // Playwright 的可点击性判定会被祖先 pointer-events 拦截。
    await page.$eval("#devPill", (el) => el.click());
    await page.waitForFunction(
      () => !document.querySelector("#devPop")?.classList.contains("hide"),
      null, { timeout: 8000 });
    await page.waitForTimeout(300);
    const s = await page.evaluate(READ);
    snaps.push({ theme, ...s });
    if (theme === "light") {
      await page.screenshot({ path: P(".tmp_devpop_light.png") });
    } else {
      await page.screenshot({ path: P(".tmp_devpop_dark.png") });
    }
    // 收起，避免影响下一轮
    await page.$eval("#devPill", (el) => el.click());
  }

  console.log("字段".padEnd(12) + "浅色".padEnd(24) + "深色");
  const keys = ["popBg", "popBorder", "popColor", "btnColor", "onBg",
    "tagColor", "hintColor", "hdColor"];
  for (const k of keys) {
    console.log(k.padEnd(12)
      + String(snaps[0][k]).padEnd(24) + String(snaps[1][k]));
  }
  console.log(`\n选项数 ${snaps[0].buttons} / ${snaps[1].buttons}`);

  await browser.close();

  const problems = [];
  const [L, D] = snaps;
  if (L.err) problems.push(`浅色下${L.err}`);
  if (D.err) problems.push(`深色下${D.err}`);
  // 两个主题的弹层背景必须不同——相同就说明有一处写死了
  if (L.popBg === D.popBg) {
    problems.push(`弹层背景不随主题变化（都是 ${L.popBg}）`);
  }
  // 浅色主题下弹层必须是浅底。深色弹层配深色文字就等于隐形文字
  if (L.popBg && !/rgba?\(2[0-5][0-9],/.test(L.popBg) && L.popBg !== "rgba(0, 0, 0, 0)") {
    problems.push(`浅色主题下弹层仍是深色底 ${L.popBg}`);
  }
  // 反之深色主题下不能是白底
  if (D.popBg && /rgba?\(2[45][0-9], 2[45][0-9], 2[45][0-9]/.test(D.popBg)) {
    problems.push(`深色主题下弹层是浅色底 ${D.popBg}`);
  }
  for (const [name, s] of [["浅色", L], ["深色", D]]) {
    for (const k of ["btnColor", "tagColor", "hdColor", "hintColor"]) {
      if (s[k] === "rgba(0, 0, 0, 0)") problems.push(`${name}下 ${k} 全透明`);
    }
  }
  if (errors.length) problems.push(`页面报错 ${errors.length} 条`);

  if (problems.length) {
    console.log("\n发现问题:");
    problems.forEach((p) => console.log("  - " + p));
    process.exit(1);
  }
  console.log("\n设备下拉配色正常：两主题背景不同、浅色下为浅底、文字均非透明");
})();
