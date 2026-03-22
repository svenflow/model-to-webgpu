import { defineConfig } from "@playwright/test";

/**
 * Playwright config for WebGPU activation matching tests.
 *
 * Launches Chromium with WebGPU enabled. On macOS, uses Metal via ANGLE.
 * On Linux CI, use Vulkan instead (swap the args).
 */
export default defineConfig({
  testDir: "./tests",
  timeout: 120_000, // WebGPU model loading can be slow
  retries: 0, // Activation matching must be deterministic
  use: {
    browserName: "chromium",
    launchOptions: {
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
        // macOS alternative (often more stable):
        // "--use-angle=metal",
      ],
    },
  },
  webServer: {
    command: "npx serve . -l 3000 --no-clipboard",
    port: 3000,
    reuseExistingServer: true,
  },
});
