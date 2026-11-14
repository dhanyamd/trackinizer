// Full-height shot of the blocks answer page, for the pre-display gate.
import { chromium } from "@playwright/test";

const out = process.argv[2] ?? "/tmp/blocks.png";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: Number(process.argv[3] ?? 1100) }, deviceScaleFactor: 2 });
const errs = [];
p.on("requestfailed", (r) => errs.push("REQFAIL " + r.url()));
p.on("response", (r) => {
  if (r.status() >= 400) errs.push(`HTTP ${r.status()} ${r.url()}`);
});
p.on("pageerror", (e) => errs.push("PAGEERROR " + e.message));
p.on("console", (m) => {
  if (m.type() === "error") errs.push("CONSOLE " + m.text().slice(0, 300));
});
await p.goto("http://127.0.0.1:8799/app/#/blocks", { waitUntil: "networkidle" });
await p.waitForTimeout(6000);
const txt = await p.evaluate(() => document.body.innerText);
console.log("textlen", txt.length);
console.log("title on page:", txt.includes("1/97"), "| chart svg:", await p.locator("svg.marks").count());
console.log("errors:", errs.slice(0, 6).join("\n") || "none");
await p.screenshot({ path: out, fullPage: true });
console.log("shot", out);
await b.close();
