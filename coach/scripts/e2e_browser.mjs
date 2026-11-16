// Real-browser end-to-end check: Chromium with a fake microphone fed from a WAV file.
//   node scripts/e2e_browser.mjs http://127.0.0.1:8799 /path/to/looped-speech.wav out_dir
// Verifies: UI loads, mic capture -> WebSocket -> VAD/turn detection -> reply audio ->
// playback reports -> latency panel, and the review screen. Prints a JSON summary.
import fs from "node:fs";
import { createRequire } from "node:module";

// require() honours NODE_PATH (ESM imports do not), so a global Playwright install works.
const { chromium } = createRequire(import.meta.url)("playwright");

const [url, wav, outDir = "."] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${wav}`,
    "--autoplay-policy=no-user-gesture-required",
  ],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const consoleErrors = [];
page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(String(e)));

const result = { ok: false, steps: [] };
const step = (name, extra = {}) => result.steps.push({ name, t: Date.now(), ...extra });
try {
  await page.goto(url);
  await page.getByRole("heading", { name: "British Conversation Coach" }).waitFor();
  step("ui_loaded");
  await page.getByTestId("start").click();
  await page.locator(".bubble.assistant").first().waitFor({ timeout: 15000 });
  step("coach_opened", { text: await page.locator(".bubble.assistant").first().innerText() });
  await page.locator(".bubble.user").first().waitFor({ timeout: 30000 });
  step("user_turn_from_microphone", { text: await page.locator(".bubble.user").first().innerText() });
  // Let a few turns happen (the WAV loops, so the "user" keeps talking).
  await page.waitForTimeout(25000);
  const metrics = await (await page.request.get(`${url}/api/metrics`)).json();
  result.metrics_summary = metrics.summary;
  result.turns = metrics.turns.length;
  result.user_bubbles = await page.locator(".bubble.user").count();
  result.assistant_bubbles = await page.locator(".bubble.assistant").count();
  result.interrupted_bubbles = await page.locator(".bubble.interrupted").count();
  result.latency_panel = await page.getByTestId("latency").innerText();
  await page.screenshot({ path: `${outDir}/conversation.png` });
  step("screenshot_conversation");
  await page.getByRole("button", { name: "End & review" }).click();
  await page.getByTestId("review").getByText("Five useful British expressions").waitFor({ timeout: 20000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${outDir}/review.png`, fullPage: true });
  step("review_shown");
  result.ok = result.user_bubbles >= 1 && result.assistant_bubbles >= 2;
} catch (e) {
  result.error = String(e);
  await page.screenshot({ path: `${outDir}/failure.png` }).catch(() => {});
}
result.console_errors = consoleErrors;
await browser.close();
console.log(JSON.stringify(result, null, 2));
process.exit(result.ok ? 0 : 1);
