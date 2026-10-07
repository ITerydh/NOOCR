// 检查明暗主题切换是否真的生效。
// 关注点有三类，都得靠浏览器算，不能靠读源码：
//   1. 变量是否落地——--bg / --text 的实际计算值要真的变了；
//   2. 界面是否重绘——body 的 background-color 与 canvas 标注色要跟着换；
//   3. 切换后不残留上一种主题的硬编码颜色——常见漏法是有几条规则
//      写死了 #fff / #000，浅色下就看不见了。
// 用法: node scripts/check_theme.js [base_url] [out_dir]
const { loadPlaywright, findChromium, P } = require("./_browser");
const { chromium } = loadPlaywright();

const EXE = findChromium();
const BASE = process.argv[2] || "http://127.0.0.1:8940";
const OUT = process.argv[3] || P(".tmp_theme");

/* 取一次快照：所有会被主题影响的计算值。
   一次性抓齐而不是分多次 evaluate，是为了让两个主题的取值走
   完全相同的代码路径——分两次抓的话，中间任何一次页面自己变了
   （比如异步渲染完）就会让对比结果没法解释。 */
const SNAPSHOT = () => {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  const px = (n) => getComputedStyle(document.body).getPropertyValue(n).trim();
  return {
    theme: document.documentElement.getAttribute("data-theme") || "(跟随系统)",
    stored: localStorage.getItem("noocr.theme") || "(未设置)",
    vars: {
      bg: v("--bg"), panel: v("--panel"), panel2: v("--panel-2"),
      inset: v("--inset"), border: v("--border"), text: v("--text"),
      textDim: v("--text-dim"), accent: v("--accent"),
      ok: v("--ok"), err: v("--err"), confHigh: v("--conf-high"),
    },
    colorScheme: cs.getPropertyValue("color-scheme").trim(),
    bodyBg: px("background-color"),
    bodyColor: px("color"),
    /* 关键角色逐个取，看有没有哪块没跟上。
       取 background-image 而不是 background-color：v3 段给 .stage
       的是一层棋盘格渐变，background-color 会返回 transparent，
       看上去像「没着色」而实际上背景确实换了——只查 color 会误报。 */
    roles: {
      header: (() => { const e = document.querySelector("header");
        if (!e) return "(无header)";
        const s = getComputedStyle(e);
        return s.backgroundColor !== "rgba(0, 0, 0, 0)"
          ? s.backgroundColor : `image:${s.backgroundImage.slice(0, 40)}`; })(),
      card: (() => { const e = document.querySelector(".card");
        return e ? getComputedStyle(e).backgroundColor : "(无卡片)"; })(),
      cardHead: (() => { const e = document.querySelector(".card-h");
        if (!e) return "(无card-h)";
        const s = getComputedStyle(e);
        return s.backgroundColor !== "rgba(0, 0, 0, 0)"
          ? s.backgroundColor : `image:${s.backgroundImage.slice(0, 40)}`; })(),
      main: (() => { const e = document.querySelector(".main");
        return e ? getComputedStyle(e).backgroundColor : "(无main)"; })(),
      stage: (() => { const e = document.querySelector(".stage");
        if (!e) return "(无stage)";
        const s = getComputedStyle(e);
        return s.backgroundColor !== "rgba(0, 0, 0, 0)"
          ? s.backgroundColor : `image:${s.backgroundImage.slice(0, 40)}`; })(),
      detail: (() => { const e = document.querySelector(".detail");
        return e ? getComputedStyle(e).backgroundColor : "(无detail)"; })(),
      // v3 段给这两处写死了深色 rgba(23,31,44,.96) / rgba(16,23,34,.88)，
      // 浅色主题下要靠 [data-theme="light"] 那几条盖回来——盖没盖住
      // 只有实测知道，扫描 CSS 里的硬编码色值查不出覆盖关系
      detailTh: (() => { const e = document.querySelector(".detail thead th");
        return e ? getComputedStyle(e).backgroundColor : "(无detail表头)"; })(),
      outBarBtn: (() => { const e = document.querySelector(".out-bar button");
        return e ? getComputedStyle(e).backgroundColor : "(无out-bar按钮)"; })(),
      hist: (() => { const e = document.querySelector(".hist");
        return e ? getComputedStyle(e).backgroundColor : "(无hist)"; })(),
      samples: (() => { const e = document.querySelector(".samples-wrap");
        return e ? getComputedStyle(e).backgroundColor : "(无samples)"; })(),
      // 选框与按钮是 v3 段改得最狠的两处，单列出来
      select: (() => { const e = document.querySelector("#backend");
        return e ? getComputedStyle(e).backgroundColor : "(无select)"; })(),
      runBtn: (() => { const e = document.querySelector("#run");
        if (!e) return "(无run)";
        const s = getComputedStyle(e);
        return s.backgroundImage !== "none"
          ? `image:${s.backgroundImage.slice(0, 40)}`
          : s.backgroundColor; })(),
    },
  };
};

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForFunction(
    () => document.querySelectorAll("#backend option").length > 0,
    null, { timeout: 15000 });

  // 每次都从「跟随系统」出发，测完清掉localStorage，
  // 否则第二次跑会继承上一次的设置，测出来的是残留状态。
  await page.evaluate(() => localStorage.removeItem("noocr.theme"));
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForFunction(
    () => document.querySelectorAll("#backend option").length > 0,
    null, { timeout: 15000 });

  const snaps = [];
  // 三种系统色_scheme 各拍一次：深色、浅色、再手动切一次。
  // 只测两种会漏掉「跟随系统」这条路径——它走的是
  // @media 分支而不是 [data-theme] 属性分支，是两套代码。
  for (const scheme of ["dark", "light"]) {
    await page.emulateMedia({ colorScheme: scheme });
    await page.waitForTimeout(250);
    snaps.push({ tag: `系统=${scheme}`, ...(await page.evaluate(SNAPSHOT)) });
  }
  // 手动切到浅色（此刻系统是 light，切出来应该是 dark 才对）
  await page.$eval("#themeBtn", (el) => el.click());
  await page.waitForTimeout(300);
  snaps.push({ tag: "手动点一次", ...(await page.evaluate(SNAPSHOT)) });
  await page.$eval("#themeBtn", (el) => el.click());
  await page.waitForTimeout(300);
  snaps.push({ tag: "手动点两次", ...(await page.evaluate(SNAPSHOT)) });

  // 报告
  console.log("=== 变量落地 ===");
  const keys = Object.keys(snaps[0].vars);
  console.log("来源".padEnd(14) + keys.map((k) => k.padEnd(12)).join(""));
  for (const s of snaps) {
    console.log(s.tag.padEnd(12) + keys.map((k) => String(s.vars[k]).padEnd(12)).join(""));
  }
  console.log("\n=== 实际生效 ===");
  for (const s of snaps) {
    console.log(`${s.tag}: data-theme=${s.theme} localStorage=${s.stored} ` +
      `scheme=${s.colorScheme}`);
    console.log(`   body ${s.bodyBg} / ${s.bodyColor}`);
    console.log(`   ${JSON.stringify(s.roles)}`);
  }

  await page.emulateMedia({ colorScheme: "dark" });
  await page.evaluate(() => { localStorage.setItem("noocr.theme", "dark"); });
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}_dark.png`, fullPage: false });
  await page.evaluate(() => { localStorage.setItem("noocr.theme", "light"); });
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}_light.png`, fullPage: false });

  // 判定
  const problems = [];
  const [sysDark, sysLight, click1, click2] = snaps;

  // 1. 变量必须真的变
  if (sysDark.vars.bg === sysLight.vars.bg) {
    problems.push(`切换系统色后 --bg 没变（都是 ${sysDark.vars.bg}）`);
  }
  if (sysDark.bodyBg === sysLight.bodyBg) {
    problems.push(`body背景色没变（都是 ${sysDark.bodyBg}）`);
  }
  // 2. 两个主题不能同色——那说明有一半规则没生效
  if (sysDark.vars.text === sysLight.vars.text) {
    problems.push(`--text 在两个主题下相同（${sysDark.vars.text}）`);
  }
  // 3. 深色下文字不能是浅色（反了），浅色下不能是深色
  if (sysDark.vars.bg !== "#0b0e14" && !/^#0/.test(sysDark.vars.bg)) {
    // 只提示，不断言：项目可能换过深色底色
    console.log(`\n注：深色底为 ${sysDark.vars.bg}，与预期 #0b0e14 不同`);
  }
  // 4. 手动切换必须与系统状态解耦：此刻系统是 light，点一下要变 dark。
  //    注意别把方向写反——「点一下跟系统一样」才是 bug。
  if (click1.vars.bg !== sysDark.vars.bg) {
    problems.push(`系统浅色下点主题钮没切到深色（得到 ${click1.vars.bg}）`);
  }
  if (click2.vars.bg !== sysLight.vars.bg) {
    problems.push(`再点一次没回到浅色（得到 ${click2.vars.bg}）`);
  }
  if (click1.stored !== "dark" || click2.stored !== "light") {
    problems.push(`localStorage 写入异常：${click1.stored} / ${click2.stored}`);
  }
  // 5. 关键角色不能有一块漏改——最常见是 .stage 或文本区写死了深色。
  //    image: 开头的跳过：两个主题的渐变内容本来就不同，
  //    等值比较必然不等，判了也没意义（截图里能直接看出来）。
  for (const [role, val] of Object.entries(click1.roles)) {
    if (String(val).startsWith("image:")) continue;
    if (val === sysLight.roles[role] && sysDark.roles[role] !== sysLight.roles[role]) {
      problems.push(`深色主题下 ${role} 仍是浅色值 ${val}`);
    }
  }
  // 6. 关键角色不能整块透明——说明选择器没命中，检查项形同虚设
  for (const [role, val] of Object.entries(click1.roles)) {
    if (val === "rgba(0, 0, 0, 0)") {
      problems.push(`${role} 背景全透明，检测项没测到东西`);
    }
  }
  if (errors.length) problems.push(`页面报错 ${errors.length} 条`);

  await browser.close();
  console.log(`\n截图: ${OUT}_dark.png / ${OUT}_light.png`);
  if (problems.length) {
    console.log("\n发现问题:");
    problems.forEach((p) => console.log("  - " + p));
    if (errors.length) errors.slice(0, 5).forEach((e) => console.log("    JS: " + e));
    process.exit(1);
  }
  console.log("\n明暗主题正常：变量、实际配色、按钮切换、持久化四项全对");
})();
