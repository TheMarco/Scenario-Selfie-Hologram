#!/usr/bin/env node
import { Command } from 'commander';
import { chromium } from 'playwright-core';
import { spawn } from 'child_process';
import http from 'http';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const PYTHON_BIN = process.env.HOLOME_PYTHON
  || (fs.existsSync(path.join(ROOT_DIR, '.venv/bin/python'))
    ? path.join(ROOT_DIR, '.venv/bin/python')
    : 'python3');
const LANDMARKER_PATH = path.join(ROOT_DIR, 'assets/models/face_landmarker.task');
const SHEET_TEMPLATE_PATH = path.join(ROOT_DIR, 'assets/templates/sheet_template.png');

const RESOLUTION_PRESETS = {
  '720p': [1280, 720],
  '1080p': [1920, 1080],
  '4k': [3840, 2160],
  'vertical': [1080, 1920],
  'reels': [1080, 1920],
  'shorts': [1080, 1920],
  'square': [1080, 1080],
};

function parseResolution(spec) {
  const key = String(spec || '1080p').toLowerCase().trim();
  if (RESOLUTION_PRESETS[key]) return RESOLUTION_PRESETS[key];
  const m = key.match(/^(\d+)x(\d+)$/);
  if (!m) throw new Error(`Invalid resolution "${spec}". Use 720p, 1080p, 4k, vertical, square, or WxH.`);
  return [parseInt(m[1], 10), parseInt(m[2], 10)];
}

function findChromeExecutable() {
  const candidates = [
    process.env.HOLOME_CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return undefined; // Let Playwright try its default
}

function runCommand(bin, args, { label = bin } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(bin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (d) => { stdout += d.toString(); });
    proc.stderr.on('data', (d) => { stderr += d.toString(); });
    proc.on('close', (code) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`${label} exited with code ${code}:\n${stderr || stdout}`));
    });
  });
}

function startStaticServer(routes) {
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',
    '.json': 'application/json',
    '.bin': 'application/octet-stream',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.map': 'application/json',
  };

  const server = http.createServer((req, res) => {
    const urlPath = (req.url || '/').split('?')[0];
    for (const [prefix, baseDir] of Object.entries(routes)) {
      if (urlPath === prefix || urlPath.startsWith(prefix + '/')) {
        const rel = urlPath === prefix ? '' : urlPath.slice(prefix.length + 1);
        const filePath = rel ? path.join(baseDir, rel) : baseDir;
        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
          const ext = path.extname(filePath);
          res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
          fs.createReadStream(filePath).pipe(res);
          return;
        }
      }
    }
    res.writeHead(404);
    res.end('Not found');
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, port: addr.port, baseUrl: `http://127.0.0.1:${addr.port}` });
    });
  });
}

function resolveOutputPaths(outSpec, format) {
  const absOut = path.resolve(outSpec);
  const ext = path.extname(absOut).toLowerCase();
  const stem = ['.mp4', '.mov', '.webm', '.png'].includes(ext)
    ? absOut.slice(0, -ext.length)
    : absOut;

  fs.mkdirSync(path.dirname(stem), { recursive: true });

  if (format === 'both') {
    return { mov: `${stem}.mov`, mp4: `${stem}.mp4` };
  }
  if (format === 'mov') return { mov: ext === '.mov' ? absOut : `${stem}.mov` };
  if (format === 'mp4') return { mp4: ext === '.mp4' ? absOut : `${stem}.mp4` };
  if (format === 'webm') return { webm: ext === '.webm' ? absOut : `${stem}.webm` };
  if (format === 'png') return { png: ext === '.png' ? absOut : `${stem}.png` };
  throw new Error(`Unknown format "${format}". Use both, mov, mp4, webm, or png.`);
}

async function handleRender(opts) {
  const startTime = Date.now();
  const log = (...a) => console.error('[holome]', ...a);

  const [width, height] = parseResolution(opts.resolution);
  const fps = parseInt(opts.fps, 10);
  const isPreview = opts.previewFrame !== undefined || opts.format === 'png';
  const format = isPreview ? 'png' : String(opts.format || 'both').toLowerCase();

  if (!opts.audio && !opts.drivingVideo && !opts.mocapVideo && !isPreview) {
    throw new Error('--audio <path> (or --mocap-video / --driving-video <path>) is required for video rendering.');
  }

  const bustDir = path.resolve(opts.bust || path.join(ROOT_DIR, 'assets/bust'));
  if (!fs.existsSync(path.join(bustDir, 'bust.bin')) || !fs.existsSync(path.join(bustDir, 'bust_meta.json'))) {
    throw new Error(`Bust assets not found in ${bustDir}. Run "holome build-bust --sheet <path> --out <dir>" first, then pass --bust <dir>.`);
  }

  // Ensure renderer bundle exists
  const bundlePath = path.join(ROOT_DIR, 'renderer/dist/app.js');
  if (!fs.existsSync(bundlePath)) {
    log('Building renderer bundle...');
    await runCommand('node', [path.join(ROOT_DIR, 'renderer/bundle.mjs')], { label: 'esbuild' });
  }

  // Resolve optional transcript text
  let transcriptText = opts.text || null;
  if (!transcriptText && opts.transcript) {
    transcriptText = fs.readFileSync(path.resolve(opts.transcript), 'utf-8');
  }

  // Step 1: Analyze audio OR extract live-action driving video into timeline + face projection frames
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'holome-'));
  const timelinePath = path.join(tmpDir, 'timeline.json');
  let timelineMeta = { duration: 0, numFrames: 1, fps };
  let drivingFramesDir = null;
  const effectiveAudioAbs = opts.audio
    ? path.resolve(opts.audio)
    : opts.mocapVideo
      ? path.resolve(opts.mocapVideo)
      : opts.drivingVideo
        ? path.resolve(opts.drivingVideo)
        : null;

  try {
    if (opts.mocapVideo) {
      const vidAbs = path.resolve(opts.mocapVideo);
      if (!fs.existsSync(vidAbs)) throw new Error(`Mocap video not found: ${vidAbs}`);
      if (!fs.existsSync(effectiveAudioAbs)) throw new Error(`Audio file not found: ${effectiveAudioAbs}`);
      log(`Tracking 3D facial mocap from ${path.basename(vidAbs)} @ ${fps}fps...`);
      const pyArgs = [
        path.join(ROOT_DIR, 'scripts/extract_mocap.py'),
        '--video', vidAbs,
        '--audio', effectiveAudioAbs,
        '--landmarker', path.join(ROOT_DIR, 'assets/models/face_landmarker.task'),
        '--out', timelinePath,
        '--fps', String(fps),
        '--glitch', String(opts.glitch),
      ];
      const { stdout } = await runCommand(PYTHON_BIN, pyArgs, { label: 'extract_mocap' });
      timelineMeta = JSON.parse(stdout.trim());
      log(`Mocap timeline ready: ${timelineMeta.duration.toFixed(2)}s (${timelineMeta.trackedFrames}/${timelineMeta.numFrames} tracked frames)`);
    } else if (opts.drivingVideo) {
      const vidAbs = path.resolve(opts.drivingVideo);
      if (!fs.existsSync(vidAbs)) throw new Error(`Driving video not found: ${vidAbs}`);
      if (!fs.existsSync(effectiveAudioAbs)) throw new Error(`Audio file not found: ${effectiveAudioAbs}`);
      log(`Extracting live talking-head projection from ${path.basename(vidAbs)} @ ${fps}fps...`);
      const pyArgs = [
        path.join(ROOT_DIR, 'scripts/extract_driving_video.py'),
        '--video', vidAbs,
        '--audio', effectiveAudioAbs,
        '--bust-meta', path.join(bustDir, 'bust_meta.json'),
        '--landmarker', path.join(ROOT_DIR, 'assets/models/face_landmarker.task'),
        '--out-dir', tmpDir,
        '--fps', String(fps),
        '--glitch', String(opts.glitch),
      ];
      const { stdout } = await runCommand(PYTHON_BIN, pyArgs, { label: 'extract_driving_video' });
      timelineMeta = JSON.parse(stdout.trim());
      drivingFramesDir = timelineMeta.framesDir;
      log(`Driving timeline ready: ${timelineMeta.duration.toFixed(2)}s (${timelineMeta.numFrames} frames)`);
    } else if (opts.audio) {
      const audioAbs = path.resolve(opts.audio);
      if (!fs.existsSync(audioAbs)) throw new Error(`Audio file not found: ${audioAbs}`);
      log(`Analyzing speech audio: ${path.basename(audioAbs)} @ ${fps}fps...`);
      const pyArgs = [
        path.join(ROOT_DIR, 'scripts/analyze_audio.py'),
        '--audio', audioAbs,
        '--out', timelinePath,
        '--fps', String(fps),
        '--glitch', String(opts.glitch),
      ];
      if (transcriptText) {
        pyArgs.push('--text', transcriptText);
      }
      const { stdout } = await runCommand(PYTHON_BIN, pyArgs, { label: 'analyze_audio' });
      timelineMeta = JSON.parse(stdout.trim());
      log(`Timeline ready: ${timelineMeta.duration.toFixed(2)}s (${timelineMeta.numFrames} frames)`);
    } else {
      // Static hero preview frame (slightly open expressive pose if requested)
      const tSec = parseFloat(opts.previewFrame || '0') || 0;
      const idleTimeline = {
        version: 1,
        fps,
        duration: tSec,
        numFrames: 1,
        frames: [{
          t: tSec,
          energy: 0.42,
          jawOpen: opts.previewJaw ? parseFloat(opts.previewJaw) : 0.0,
          mouthFunnel: opts.previewFunnel ? parseFloat(opts.previewFunnel) : 0.0,
          mouthWide: opts.previewWide ? parseFloat(opts.previewWide) : 0.0,
          mouthClose: 0.0,
          mouthF: 0.0,
          upperLipRaise: opts.previewJaw ? parseFloat(opts.previewJaw) * 0.5 : 0.0,
          blink: opts.previewBlink ? parseFloat(opts.previewBlink) : 0.0,
          browRaise: 0.15,
          headYaw: 0.04,
          headPitch: -0.02,
          headRoll: 0.01,
          glitch: parseFloat(opts.glitch) * 0.25,
          spectrum: [0.35, 0.65, 0.82, 0.60, 0.48, 0.35, 0.24, 0.18],
        }],
      };
      fs.writeFileSync(timelinePath, JSON.stringify(idleTimeline));
    }

    // Decide renderer background mode
    let bgMode = String(opts.background || 'auto').toLowerCase();
    if (bgMode === 'auto') {
      if (format === 'both' || format === 'mov' || format === 'webm') bgMode = 'transparent';
      else if (format === 'png') bgMode = 'black';
      else bgMode = 'black';
    }

    // Step 2: Start local static server for renderer + bust + timeline (+ optional driving video face frames)
    const routes = {
      '/renderer': path.join(ROOT_DIR, 'renderer'),
      '/bust': bustDir,
      '/timeline.json': timelinePath,
    };
    if (drivingFramesDir) {
      routes['/driving'] = drivingFramesDir;
    }
    const { server, baseUrl } = await startStaticServer(routes);

    const outputs = resolveOutputPaths(opts.out, format);

    // Step 3: Launch headless Chrome with hardware WebGL Metal acceleration
    const chromePath = findChromeExecutable();
    log(`Launching GPU renderer (${width}x${height}, format=${format}, bg=${bgMode})...`);
    const browser = await chromium.launch({
      executablePath: chromePath,
      headless: true,
      args: [
        '--headless=new',
        '--use-gl=angle',
        '--use-angle=metal',
        '--enable-webgl',
        '--ignore-gpu-blocklist',
        '--no-sandbox',
        '--hide-scrollbars',
        '--disable-dev-shm-usage',
      ],
    });

    try {
      const context = await browser.newContext({
        viewport: { width, height },
        deviceScaleFactor: 1,
      });
      const page = await context.newPage();
      page.on('pageerror', (err) => log('Renderer pageerror:', err.message));
      page.on('console', (msg) => {
        if (msg.type() === 'error') log('Renderer console error:', msg.text());
      });

      await page.goto(`${baseUrl}/renderer/index.html`, { waitUntil: 'networkidle' });

      const initConfig = {
        width,
        height,
        background: bgMode,
        color: opts.color,
        camera: opts.camera,
        cameraMotion: opts.cameraMotion,
        cameraYaw: opts.cameraYaw !== undefined ? parseFloat(opts.cameraYaw) : undefined,
        cameraPitch: opts.cameraPitch !== undefined ? parseFloat(opts.cameraPitch) : undefined,
        cameraZoom: opts.cameraZoom !== undefined ? parseFloat(opts.cameraZoom) : undefined,
        lookAtCamera: opts.lookAtCamera,
        hud: opts.hud,
        emitter: opts.emitter,
        bloom: parseFloat(opts.bloom),
        density: parseFloat(opts.density),
        intensity: parseFloat(opts.intensity),
        pointScale: parseFloat(opts.pointScale),
        bustMetaUrl: `${baseUrl}/bust/bust_meta.json`,
        bustBinUrl: `${baseUrl}/bust/bust.bin`,
        timelineUrl: `${baseUrl}/timeline.json`,
        drivingBaseUrl: drivingFramesDir ? `${baseUrl}/driving` : null,
      };

      const engineInfo = await page.evaluate((cfg) => window.holome.init(cfg), initConfig);
      const canvasLoc = page.locator('#holo-canvas');

      if (isPreview) {
        const targetSec = parseFloat(opts.previewFrame || '0') || 0;
        const frameIdx = (opts.audio || opts.drivingVideo)
          ? Math.min(engineInfo.numFrames - 1, Math.max(0, Math.round(targetSec * fps)))
          : 0;
        await page.evaluate((idx) => window.holome.renderFrame(idx), frameIdx);
        const pngBuf = await canvasLoc.screenshot({ type: 'png', omitBackground: bgMode === 'transparent' });
        fs.writeFileSync(outputs.png, pngBuf);
        log(`Saved preview frame -> ${outputs.png}`);
      } else {
        // Step 4: Spawn FFmpeg pipe encoder (fast JPEG q96 pipe for opaque MP4, lossless PNG pipe for alpha!)
        const audioAbs = effectiveAudioAbs;
        const useJpegPipe = bgMode !== 'transparent' && format === 'mp4';
        const ffArgs = [
          '-y', '-v', 'error',
          '-f', 'image2pipe',
          '-vcodec', useJpegPipe ? 'mjpeg' : 'png',
          '-framerate', String(fps),
          '-i', '-',
          '-i', audioAbs,
        ];

        if (format === 'both') {
          ffArgs.push(
            '-filter_complex',
            `[0:v]split=2[v_alpha][v_for_blk];color=c=black:s=${width}x${height}:r=${fps}[bg];[bg][v_for_blk]overlay=0:0:shortest=1:format=auto[v_blk]`,
            '-map', '[v_alpha]', '-map', '1:a:0',
            '-c:v', 'prores_ks', '-profile:v', '4444', '-pix_fmt', 'yuva444p10le',
            '-c:a', 'pcm_s16le', '-shortest', outputs.mov,
            '-map', '[v_blk]', '-map', '1:a:0',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest', outputs.mp4
          );
        } else if (format === 'mov') {
          ffArgs.push(
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'prores_ks', '-profile:v', '4444', '-pix_fmt', 'yuva444p10le',
            '-c:a', 'pcm_s16le', '-shortest', outputs.mov
          );
        } else if (format === 'webm') {
          ffArgs.push(
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-crf', '22', '-b:v', '0', '-auto-alt-ref', '0',
            '-c:a', 'libopus', '-b:a', '128k', '-shortest', outputs.webm
          );
        } else if (format === 'mp4') {
          if (bgMode === 'transparent') {
            ffArgs.push(
              '-filter_complex',
              `color=c=black:s=${width}x${height}:r=${fps}[bg];[bg][0:v]overlay=0:0:shortest=1:format=auto[v_blk]`,
              '-map', '[v_blk]', '-map', '1:a:0',
              '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
              '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest', outputs.mp4
            );
          } else {
            ffArgs.push(
              '-map', '0:v:0', '-map', '1:a:0',
              '-vf', 'scale=in_range=full:out_range=tv,format=yuv420p',
              '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
              '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest', outputs.mp4
            );
          }
        }

        const ffmpeg = spawn('ffmpeg', ffArgs, { stdio: ['pipe', 'ignore', 'pipe'] });
        let ffStderr = '';
        ffmpeg.stderr.on('data', (d) => { ffStderr += d.toString(); });
        const ffDone = new Promise((resolve, reject) => {
          ffmpeg.on('close', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`ffmpeg exited with code ${code}:\n${ffStderr}`));
          });
        });

        const totalFrames = engineInfo.numFrames;
        const omitBg = bgMode === 'transparent';
        const reportEvery = Math.max(1, Math.floor(totalFrames / 10));

        for (let i = 0; i < totalFrames; i++) {
          await page.evaluate((idx) => window.holome.renderFrame(idx), i);
          const frameBuf = useJpegPipe
            ? await canvasLoc.screenshot({ type: 'jpeg', quality: 96 })
            : await canvasLoc.screenshot({ type: 'png', omitBackground: omitBg });
          const ok = ffmpeg.stdin.write(frameBuf);
          if (!ok) {
            await new Promise((r) => ffmpeg.stdin.once('drain', r));
          }
          if ((i + 1) % reportEvery === 0 || i === totalFrames - 1) {
            const pct = (((i + 1) / totalFrames) * 100).toFixed(0);
            log(`Rendered ${i + 1}/${totalFrames} frames (${pct}%)`);
          }
        }

        ffmpeg.stdin.end();
        await ffDone;
      }
    } finally {
      await browser.close();
      server.close();
    }

    const elapsedSec = (Date.now() - startTime) / 1000;
    const result = {
      status: 'ok',
      outputs,
      resolution: `${width}x${height}`,
      fps,
      durationSec: timelineMeta.duration,
      numFrames: isPreview ? 1 : timelineMeta.numFrames,
      background: bgMode,
      camera: opts.camera,
      hud: opts.hud,
      emitter: opts.emitter,
      color: opts.color,
      renderTimeSec: Number(elapsedSec.toFixed(2)),
    };

    if (opts.json) {
      process.stdout.write(JSON.stringify(result, null, 2) + '\n');
    } else {
      log(`Done in ${result.renderTimeSec}s:`);
      for (const [k, p] of Object.entries(outputs)) {
        const sizeMB = (fs.statSync(p).size / 1e6).toFixed(2);
        log(`  [${k}] ${p} (${sizeMB} MB)`);
      }
    }
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

async function handleBuildBust(opts) {
  const views = [opts.front, opts.left, opts.right];
  const haveViews = views.every(Boolean);
  if (views.some(Boolean) && !haveViews) throw new Error('--front, --left and --right must be given together.');
  if (!opts.sheet && !haveViews) throw new Error('Give --sheet <path>, or --front, --left and --right.');
  if (opts.sheet && haveViews) throw new Error('Give either --sheet or the three view images, not both.');

  const outAbs = path.resolve(opts.out);
  const args = [
    path.join(ROOT_DIR, 'scripts/build_bust.py'),
    '--out', outAbs,
    '--landmarker', LANDMARKER_PATH,
    '--rows', String(opts.rows),
    '--cols', String(opts.cols),
    '--layout', String(opts.layout),
  ];
  if (opts.sheet) args.push('--sheet', path.resolve(opts.sheet));
  else args.push('--front', path.resolve(opts.front), '--left', path.resolve(opts.left), '--right', path.resolve(opts.right));
  if (opts.swapProfiles) args.push('--swap-profiles');
  if (opts.mirrorProfile) args.push('--mirror-profile', String(opts.mirrorProfile));
  const { stdout } = await runCommand(PYTHON_BIN, args, { label: 'build_bust' });
  if (opts.json) {
    process.stdout.write(stdout);
    return;
  }
  const result = JSON.parse(stdout.trim());
  console.error(`[holome] Built 3D bust (${result.numVertices.toLocaleString()} vertices, layout=${result.layout}) -> ${outAbs}`);
  console.error(`[holome] QA preview: ${result.qaPreview}`);
  for (const w of result.warnings) console.error(`[holome] WARNING: ${w}`);
}

async function runPythonJson(script, args, opts) {
  const { stdout } = await runCommand(PYTHON_BIN, [path.join(ROOT_DIR, 'scripts', script), ...args], { label: script });
  const result = JSON.parse(stdout.trim());
  if (opts.json) process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  return result;
}

async function handlePickFrame(opts) {
  const result = await runPythonJson('pick_frame.py', [
    '--video', path.resolve(opts.video),
    '--out', path.resolve(opts.out),
    '--landmarker', LANDMARKER_PATH,
  ], opts);
  if (!opts.json) {
    console.error(`[holome] Best frame at ${result.timeSec}s (yaw ${result.yawDeg} deg, jawOpen ${result.jawOpen}) -> ${result.frame}`);
    for (const w of result.warnings) console.error(`[holome] WARNING: ${w}`);
  }
}

async function handleSplitSheet(opts) {
  const args = [
    '--sheet', path.resolve(opts.sheet),
    '--out', path.resolve(opts.outDir),
    '--landmarker', LANDMARKER_PATH,
    '--layout', String(opts.layout),
    '--panels-only',
  ];
  if (opts.swapProfiles) args.push('--swap-profiles');
  if (opts.mirrorProfile) args.push('--mirror-profile', String(opts.mirrorProfile));
  const result = await runPythonJson('build_bust.py', args, opts);
  if (!opts.json) {
    console.error(`[holome] Split sheet (layout=${result.layout}); QA preview: ${result.qaPreview}`);
    for (const [k, p] of Object.entries(result.panels)) console.error(`  ${k}: ${p}`);
    for (const w of result.warnings) console.error(`[holome] WARNING: ${w}`);
  }
}

async function handleSheetTemplate(opts) {
  const result = await runPythonJson('make_sheet_template.py', ['--out', path.resolve(opts.out)], opts);
  if (!opts.json) console.error(`[holome] Wrote sheet template -> ${result.template}`);
}

async function handleSplitAudio(opts) {
  const result = await runPythonJson('split_audio.py', [
    '--audio', path.resolve(opts.audio),
    '--out-dir', path.resolve(opts.outDir),
    '--max-sec', String(opts.maxSec),
    '--min-sec', String(opts.minSec),
  ], opts);
  if (!opts.json) {
    console.error(`[holome] ${result.durationSec.toFixed(2)}s split into ${result.segments.length} segment(s) -> ${result.plan}`);
    for (const s of result.segments) console.error(`  ${s.index}: ${s.start.toFixed(2)}-${s.end.toFixed(2)}s ${s.path}`);
  }
}

async function handleDrivingFrame(opts) {
  const result = await runPythonJson('driving_frame.py', [
    '--bust', path.resolve(opts.bust),
    '--out', path.resolve(opts.out),
    '--aspect', String(opts.aspect),
    '--height', String(opts.height),
    '--landmarker', LANDMARKER_PATH,
    ...(opts.panel ? ['--panel', path.resolve(opts.panel)] : []),
  ], opts);
  if (!opts.json) console.error(`[holome] Wrote ${result.width}x${result.height} opening frame -> ${result.frame}`);
}

async function probeVideoSize(file) {
  const { stdout } = await runCommand('ffprobe', [
    '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', file,
  ], { label: 'ffprobe' });
  const s = JSON.parse(stdout).streams?.[0];
  if (!s) throw new Error(`No video stream in ${file}`);
  return [s.width, s.height];
}

async function handleJoinDriving(opts) {
  const plan = JSON.parse(fs.readFileSync(path.resolve(opts.plan), 'utf-8'));
  const videos = opts.videos.map((v) => path.resolve(v));
  if (videos.length !== plan.segments.length) {
    throw new Error(`Plan has ${plan.segments.length} segment(s) but ${videos.length} video(s) were given.`);
  }
  for (const v of videos) if (!fs.existsSync(v)) throw new Error(`Video not found: ${v}`);
  const fps = parseInt(opts.fps, 10);
  const [w, h] = await probeVideoSize(videos[0]);
  const outAbs = path.resolve(opts.out);
  fs.mkdirSync(path.dirname(outAbs), { recursive: true });

  // Each clip is resampled, padded by repeating its last frame, then trimmed to its exact audio
  // segment length, so the joined video stays frame-locked to the original audio.
  const args = ['-y', '-v', 'error'];
  for (const v of videos) args.push('-i', v);
  args.push('-i', plan.source);
  const chains = plan.segments.map((seg, i) => (
    `[${i}:v]fps=${fps},scale=${w}:${h}:force_original_aspect_ratio=decrease,`
    + `pad=${w}:${h}:(ow-iw)/2:(oh-ih)/2,setsar=1,tpad=stop_mode=clone:stop_duration=2,`
    + `trim=duration=${seg.duration},setpts=PTS-STARTPTS[v${i}]`
  ));
  const concatIn = plan.segments.map((_, i) => `[v${i}]`).join('');
  args.push(
    '-filter_complex', `${chains.join(';')};${concatIn}concat=n=${videos.length}:v=1:a=0[outv]`,
    '-map', '[outv]', '-map', `${videos.length}:a:0`,
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-b:a', '192k', '-shortest', outAbs,
  );
  await runCommand('ffmpeg', args, { label: 'ffmpeg join-driving' });
  const result = { status: 'ok', drivingVideo: outAbs, segments: videos.length, fps, size: `${w}x${h}`, durationSec: plan.durationSec };
  if (opts.json) process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  else console.error(`[holome] Joined ${videos.length} clip(s) -> ${outAbs}`);
}

async function handleDoctor(opts) {
  const checks = {};
  const ok = (name, pass, detail) => { checks[name] = { ok: Boolean(pass), detail }; };

  for (const pkg of ['commander', 'playwright-core', 'three', 'esbuild']) {
    const p = path.join(ROOT_DIR, 'node_modules', pkg, 'package.json');
    ok(`node:${pkg}`, fs.existsSync(p), fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf-8')).version : 'missing (run setup.sh)');
  }
  try {
    const { stdout } = await runCommand(PYTHON_BIN, ['-c',
      'import sys, cv2, mediapipe, torch, transformers, faster_whisper, cmudict, scipy, trimesh, rtree, PIL; '
      + 'print(sys.version.split()[0], "mps" if torch.backends.mps.is_available() else "cpu")'], { label: 'python' });
    ok('python', true, `${PYTHON_BIN} (${stdout.trim()})`);
  } catch (e) {
    ok('python', false, `${PYTHON_BIN}: ${e.message.split('\n').slice(-2).join(' ').trim()}`);
  }
  ok('model:face_landmarker', fs.existsSync(LANDMARKER_PATH), LANDMARKER_PATH);
  ok('template:sheet', fs.existsSync(SHEET_TEMPLATE_PATH), SHEET_TEMPLATE_PATH);
  for (const bin of ['ffmpeg', 'ffprobe']) {
    try {
      const { stdout } = await runCommand(bin, ['-version'], { label: bin });
      ok(bin, true, stdout.split('\n')[0]);
    } catch (e) {
      ok(bin, false, 'not found on PATH');
    }
  }
  let chrome = findChromeExecutable();
  if (!chrome) {
    try {
      const p = chromium.executablePath();
      if (p && fs.existsSync(p)) chrome = p;
    } catch (e) { /* no bundled browser */ }
  }
  ok('chrome', Boolean(chrome), chrome || 'no Chrome/Chromium found (set HOLOME_CHROME_PATH or run: npx playwright-core install chromium)');

  const allOk = Object.values(checks).every((c) => c.ok);
  const result = { status: allOk ? 'ok' : 'missing', root: ROOT_DIR, checks };
  if (opts.json) process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  else {
    for (const [k, c] of Object.entries(checks)) console.error(`${c.ok ? 'ok     ' : 'MISSING'} ${k}: ${c.detail}`);
    console.error(allOk ? '[holome] All dependencies found.' : '[holome] Some dependencies are missing. Run setup.sh.');
  }
  if (!allOk) process.exitCode = 1;
}

async function handleImportGlb(opts) {
  const glbAbs = path.resolve(opts.glb);
  const outAbs = path.resolve(opts.out);
  const args = [
    path.join(ROOT_DIR, 'scripts/import_glb.py'),
    '--glb', glbAbs,
    '--out', outAbs,
    '--landmarker', LANDMARKER_PATH,
    '--rows', String(opts.rows),
    '--cols', String(opts.cols),
  ];
  if (opts.yaw !== undefined) args.push('--yaw', String(opts.yaw));
  const { stdout } = await runCommand(PYTHON_BIN, args, { label: 'import_glb' });
  if (opts.json) process.stdout.write(stdout);
  else console.error('[holome] Imported external 3D bust:', stdout.trim());
}

const program = new Command();
program
  .name('holome')
  .description('High-end 3D audio-driven hologram generator CLI for video & agentic pipelines')
  .version('1.0.0');

program
  .command('render')
  .description('Render a talking 3D hologram video (or still preview PNG) from spoken-word audio')
  .option('-a, --audio <path>', 'Spoken-word audio file (.wav, .mp3, .m4a, .flac, .ogg)')
  .option('-m, --mocap-video <path>', 'Talking-head video (.MOV, .mp4) tracked locally via 52 ARKit blendshapes + 3D head pose onto the 3D hologram rig')
  .option('-d, --driving-video <path>', 'Optional live-action talking-head video (e.g. Scenario LTX-2.5 or webcam) projected live onto the 3D bust')
  .requiredOption('-o, --out <path>', 'Output file path or base stem (e.g. outputs/clip or outputs/clip.mp4)')
  .option('-f, --format <type>', 'Output format: both (ProRes 4444 .mov + black .mp4), mov, mp4, webm, png', 'both')
  .option('-r, --resolution <res>', 'Resolution preset (720p, 1080p, 4k, vertical, square) or WxH', '1080p')
  .option('--fps <num>', 'Frames per second (24, 30, 60)', '30')
  .option('--camera <preset>', 'Camera angle preset: hero, front, left34, right34, orbit', 'hero')
  .option('--camera-motion <mode>', 'Camera motion: subtle, static, orbit', 'subtle')
  .option('--camera-yaw <deg>', 'Override camera yaw in degrees')
  .option('--camera-pitch <deg>', 'Override camera pitch in degrees')
  .option('--camera-zoom <num>', 'Camera zoom multiplier (e.g. 1.15)')
  .option('--look-at-camera', 'Aim head and eye gaze directly at the camera lens', true)
  .option('--no-look-at-camera', 'Keep head facing strictly forward without tracking camera')
  .option('--hud', 'Include floating 3D sci-fi HUD telemetry', true)
  .option('--no-hud', 'Disable floating 3D HUD telemetry')
  .option('--emitter', 'Include holographic projector base and upward light cone', true)
  .option('--no-emitter', 'Disable emitter base (clean floating bust for overlays)')
  .option('--background <mode>', 'Background: auto, transparent, black, lab', 'auto')
  .option('--color <spec>', 'Hologram color preset (cyan, ice, amber, emerald, crimson, violet) or #RRGGBB', 'cyan')
  .option('--bloom <num>', 'Bloom glow intensity (0.0 - 2.0)', '1.0')
  .option('--glitch <num>', 'Audio-reactive glitch intensity (0.0 - 1.0)', '0.35')
  .option('--density <num>', 'Point cloud density (0.3 - 1.0)', '1.0')
  .option('--intensity <num>', 'Overall hologram brightness multiplier', '1.0')
  .option('--point-scale <num>', 'Point dot size multiplier', '1.08')
  .option('--text <string>', 'Optional spoken transcript for tighter phoneme lip-sync alignment')
  .option('--transcript <path>', 'Optional text file containing spoken transcript')
  .option('--bust <dir>', 'Path to bust asset directory (default: assets/bust)')
  .option('--preview-frame <sec>', 'Render a single PNG preview frame at timestamp <sec> instead of video')
  .option('--preview-jaw <num>', 'Jaw open amount (0..1) for audio-less preview frame testing')
  .option('--preview-funnel <num>', 'Mouth O/U funnel amount (0..1) for preview frame testing')
  .option('--preview-wide <num>', 'Mouth E/I wide amount (0..1) for preview frame testing')
  .option('--preview-blink <num>', 'Eye blink amount (0..1) for audio-less preview frame testing')
  .option('--json', 'Emit machine-readable JSON metadata to stdout (ideal for agentic pipelines)', false)
  .action((opts) => handleRender(opts).catch((err) => {
    console.error('[holome] ERROR:', err.message);
    process.exit(1);
  }));

program
  .command('build-bust')
  .description('Reconstruct & auto-rig a 3D hologram bust from a 3-panel Front/Left/Right character sheet')
  .option('--sheet <path>', 'Path to 3-view character sheet image')
  .option('--front <path>', 'Separate front view image (with --left and --right, instead of --sheet)')
  .option('--left <path>', 'Separate left profile image (nose pointing to the image left edge)')
  .option('--right <path>', 'Separate right profile image (nose pointing to the image right edge)')
  .requiredOption('--out <dir>', 'Output bust asset directory (one per person)')
  .option('--layout <mode>', 'How to split --sheet: auto, bordered, subjects, thirds', 'auto')
  .option('--swap-profiles', 'Swap the left and right panels (profiles were generated in the wrong slots)')
  .option('--mirror-profile <side>', 'Use only this profile (left or right) and mirror it for the other')
  .option('--rows <num>', 'Vertical mesh resolution', '320')
  .option('--cols <num>', 'Circumferential mesh resolution', '340')
  .option('--json', 'Emit JSON output', false)
  .action((opts) => handleBuildBust(opts).catch((err) => {
    console.error('[holome] ERROR:', err.message);
    process.exit(1);
  }));

const failOnError = (fn) => (opts) => fn(opts).catch((err) => {
  console.error('[holome] ERROR:', err.message);
  process.exit(1);
});

program
  .command('doctor')
  .description('Check that Node packages, the Python environment, models, ffmpeg and Chrome are installed')
  .option('--json', 'Emit JSON output', false)
  .action(failOnError(handleDoctor));

program
  .command('pick-frame')
  .description('Pick the sharpest frontal, eyes-open, mouth-closed still from a selfie video')
  .requiredOption('--video <path>', 'Selfie video (.MOV, .mp4)')
  .requiredOption('--out <path>', 'Output JPEG path')
  .option('--json', 'Emit JSON output', false)
  .action(failOnError(handlePickFrame));

program
  .command('split-sheet')
  .description('Split a character sheet into full-resolution FRONT/LEFT/RIGHT panels (for Tripo) plus a QA preview')
  .requiredOption('--sheet <path>', 'Character sheet image')
  .requiredOption('--out-dir <dir>', 'Directory for panel_*_hires.png and qa_panels.png')
  .option('--layout <mode>', 'auto, bordered, subjects, thirds', 'auto')
  .option('--swap-profiles', 'Swap the left and right panels')
  .option('--mirror-profile <side>', 'Use only this profile (left or right) and mirror it for the other')
  .option('--json', 'Emit JSON output', false)
  .action(failOnError(handleSplitSheet));

program
  .command('sheet-template')
  .description('Write the empty FRONT/LEFT/RIGHT character sheet template used as a layout reference')
  .requiredOption('--out <path>', 'Output PNG path')
  .option('--json', 'Emit JSON output', false)
  .action(failOnError(handleSheetTemplate));

program
  .command('split-audio')
  .description('Split audio at pauses into segments short enough for audio-to-video models (LTX: 20s max)')
  .requiredOption('-a, --audio <path>', 'Spoken-word audio file')
  .requiredOption('--out-dir <dir>', 'Directory for segment WAVs and segments.json')
  .option('--max-sec <num>', 'Maximum segment length in seconds', '19.5')
  .option('--min-sec <num>', 'Minimum segment length in seconds', '3')
  .option('--json', 'Emit JSON output', false)
  .action(failOnError(handleSplitAudio));

program
  .command('driving-frame')
  .description('Crop a bust\'s front panel into the opening frame for an audio-to-video talking-head model')
  .requiredOption('--bust <dir>', 'Bust asset directory')
  .requiredOption('--out <path>', 'Output PNG path')
  .option('--panel <path>', 'Front image to crop instead of the bust\'s saved front panel (e.g. split-sheet output)')
  .option('--aspect <ratio>', '9:16 or 16:9', '9:16')
  .option('--height <px>', 'Output height in pixels', '1280')
  .option('--json', 'Emit JSON output', false)
  .action(failOnError(handleDrivingFrame));

program
  .command('join-driving')
  .description('Join per-segment talking-head clips into one driving video locked to the original audio')
  .requiredOption('--plan <path>', 'segments.json written by split-audio')
  .requiredOption('--videos <paths...>', 'One clip per segment, in segment order')
  .requiredOption('-o, --out <path>', 'Output .mp4 path')
  .option('--fps <num>', 'Output frame rate', '30')
  .option('--json', 'Emit JSON output', false)
  .action(failOnError(handleJoinDriving));

program
  .command('import-glb')
  .description('Import & auto-rig an external 3D bust (.glb/.gltf/.obj, e.g. from Scenario.com) into HoloMe format')
  .requiredOption('--glb <path>', 'Path to .glb/.gltf/.obj bust file')
  .requiredOption('--out <dir>', 'Output bust asset directory (one per person)')
  .option('--rows <num>', 'Vertical contour ring resolution', '320')
  .option('--cols <num>', 'Circumferential column resolution', '340')
  .option('--yaw <deg>', 'Manual yaw override in degrees (auto-detected if omitted)')
  .option('--json', 'Emit JSON output', false)
  .action((opts) => handleImportGlb(opts).catch((err) => {
    console.error('[holome] ERROR:', err.message);
    process.exit(1);
  }));

program.parse();
