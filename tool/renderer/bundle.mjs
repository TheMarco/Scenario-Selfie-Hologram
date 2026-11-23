import * as esbuild from 'esbuild';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

await esbuild.build({
  entryPoints: [resolve(__dirname, 'src/main.js')],
  bundle: true,
  format: 'esm',
  minify: false,
  sourcemap: true,
  outfile: resolve(__dirname, 'dist/app.js'),
  target: ['chrome120'],
});

console.error('[holome] Bundled renderer -> renderer/dist/app.js');
