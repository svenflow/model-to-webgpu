import { defineConfig } from "tsup";

/**
 * Zero-dependency ESM build config.
 *
 * Produces:
 *   dist/index.js    — minified ESM bundle
 *   dist/index.d.ts  — TypeScript declarations
 *
 * No external dependencies — everything is bundled inline.
 * Shaders are string literals in shaders.ts, not external .wgsl files.
 */
export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  dts: true,
  minify: true,
  clean: true,
  outDir: "dist",
  target: "esnext",
  splitting: false,
  sourcemap: false,
  treeshake: true,
});
