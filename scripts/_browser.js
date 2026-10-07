// 浏览器类检测脚本的公共引导。
//
// 这些脚本（check_theme / check_device_popup / check_result_theme /
// check_backend_options / check_backend_label / shoot_screens）都做同一件事：
// 用真实浏览器打开 WebUI、读getComputedStyle 的实际计算值。
// 三件事原先在每个脚本里各写一遍，且都写死了本机绝对路径，换台机器
// 或换个用户名就全部报MODULE_NOT_FOUND 或找不到浏览器。收敛到这里：
//
//   1. ROOT     —— 项目根，按 __dirname 上溯两层算，不依赖 cwd。
//                  之前写死 "C:/Users/iterhui/Desktop/ocr"，从别的目录
//                  调用 node scripts/check_theme.js 就会存错位置。
//   2. chromium —— playwright-core 不在项目依赖里（它是开发期工具，
//                  装进 requirements 是污染），require 会失败。这里按序
//                  尝试若干来源，并在全失败时给出可操作的提示——直接
//                  抛MODULE_NOT_FOUND 的话，看到的人不知道该干什么。
//   3. exe      —— 已下载的 Chromium 路径。版本号会变（chromium-1129
//                  → chromium-1234），写死版本号等于给脚本设了个过期
//                  炸弹，所以扫目录取最新的一个。

const fs = require("fs");
const path = require("path");
const os = require("os");

/** 项目根目录（本文件在 scripts/ 下，故上溯两层）。 */
const ROOT = path.resolve(__dirname, "..");

/** 项目内路径拼接。用它替代一切硬编码绝对路径。 */
const P = (...parts) => path.join(ROOT, ...parts);

/* playwright-core 的候选来源。按可靠性排序：
   1. 项目本地 node_modules（装了依赖的正常路径）
   2. NODE_PATH —— 调用方显式指定的
   3. 同机其他项目的 node_modules —— 本机的实际情况：包在
      ~/WorkBuddy/uitest/node_modules 下，不设NODE_PATH 就找不到 */
const CANDIDATES = [
  P("node_modules"),
  process.env.NODE_PATH,
  path.join(os.homedir(), "WorkBuddy", "uitest", "node_modules"),
].filter(Boolean);

function loadPlaywright() {
  const tried = [];
  for (const dir of CANDIDATES) {
    try {
      // 直接require 绝对路径下的包，绕开模块解析器的查找顺序
      return require(path.join(dir, "playwright-core"));
    } catch (e) {
      tried.push(dir);
    }
  }
  console.error(
    "找不到 playwright-core。已尝试：\n" +
      tried.map((p) => `  - ${p}`).join("\n") +
      "\n\n装一下即可（仅开发期需要，不进requirements）：\n" +
      "  npm i -D playwright-core && npx playwright install chromium\n" +
      "或指定已有安装的位置：\n" +
      "  set NODE_PATH=C:\\path\\to\\node_modules",
  );
  process.exit(2);
}

/* 扫 ms-playwright 缓存目录取 Chromium。取版本号最大的一个：
   playwright 升级后旧目录不会自动删，按目录名排序取最大即可，
   写死具体版本号则会在升级后静默失效。 */
function findChromium() {
  const roots = [
    process.env.PLAYWRIGHT_BROWSERS_PATH,
    path.join(os.homedir(), "AppData", "Local", "ms-playwright"),
    path.join(os.homedir(), "Library", "Caches", "ms-playwright"),
    path.join(os.homedir(), ".cache", "ms-playwright"),
  ].filter(Boolean);

  for (const root of roots) {
    let entries;
    try {
      entries = fs.readdirSync(root, { withFileTypes: true });
    } catch {
      continue; // 该根目录不存在，换下一个
    }
    const dirs = entries
      .filter((e) => e.isDirectory() && e.name.startsWith("chromium"))
      .map((e) => e.name)
      .sort()
      .reverse(); // chromium-<build>，字符串倒序≈ 版本倒序
    for (const d of dirs) {
      for (const rel of [
        ["chrome-win", "chrome.exe"],       // Windows
        ["chrome-linux", "chrome"],         // Linux
        ["chrome-mac", "Chromium.app", "Contents", "MacOS", "Chromium"], // macOS
      ]) {
        const exe = path.join(root, d, ...rel);
        if (fs.existsSync(exe)) return exe;
      }
    }
  }
  console.error(
    "找不到已下载的 Chromium。装一下：\n" +
      "  npx playwright install chromium",
  );
  process.exit(2);
}

/** 项目静态资源目录（部分脚本要直接读 noocr/web/static 下的样例图）。 */
const STATIC = P("noocr", "web", "static");

module.exports = { ROOT, P, STATIC, loadPlaywright, findChromium };
