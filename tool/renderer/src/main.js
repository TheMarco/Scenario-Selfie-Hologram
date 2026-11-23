import * as THREE from 'three';
import { HologramBust } from './HologramBust.js';
import { HoloHUD } from './HoloHUD.js';
import { EmitterBase } from './EmitterBase.js';
import {
  FULLSCREEN_VERT,
  BLOOM_EXTRACT_FRAG,
  BLUR_FRAG,
  COMPOSITE_FRAG,
} from './shaders.js';

const PRESET_COLORS = {
  cyan: '#00d8ff',
  ice: '#4ce8ff',
  amber: '#ff9d2e',
  emerald: '#00f0a8',
  crimson: '#ff3366',
  violet: '#a259ff',
};

function makePalette(colorSpec = 'cyan') {
  const hex = PRESET_COLORS[colorSpec.toLowerCase()] || colorSpec;
  const primary = new THREE.Color(hex);
  const hsl = {};
  primary.getHSL(hsl);

  const hot = new THREE.Color().setHSL(hsl.h, Math.max(0.15, hsl.s * 0.38), Math.min(0.96, hsl.l + 0.42));
  const deep = new THREE.Color().setHSL((hsl.h + 0.04) % 1.0, Math.min(1.0, hsl.s * 1.05), Math.max(0.12, hsl.l * 0.45));
  return { primary, hot, deep };
}

class HoloEngine {
  constructor() {
    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.bust = null;
    this.hud = null;
    this.emitter = null;
    this.timeline = null;
    this.config = null;
  }

  async init(config) {
    this.config = config;
    const width = config.width || 1920;
    const height = config.height || 1080;
    const bgModeStr = config.background || 'black'; // 'black' | 'transparent' | 'lab'
    const bgModeInt = bgModeStr === 'transparent' ? 1 : bgModeStr === 'lab' ? 2 : 0;

    const canvas = document.getElementById('holo-canvas');
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.autoClear = false;

    this.scene = new THREE.Scene();

    // Camera setup (portrait/landscape adaptive FOV)
    const aspect = width / height;
    const fov = aspect < 1.0 ? 44 : 32;
    this.camera = new THREE.PerspectiveCamera(fov, aspect, 0.1, 30);

    const palette = makePalette(config.color || 'cyan');

    // Load 3D bust assets
    const [metaRes, binRes] = await Promise.all([
      fetch(config.bustMetaUrl),
      fetch(config.bustBinUrl),
    ]);
    const meta = await metaRes.json();
    const binBuf = await binRes.arrayBuffer();
    const floatData = new Float32Array(binBuf);

    // Load timeline JSON if provided
    if (config.timelineUrl) {
      const tlRes = await fetch(config.timelineUrl);
      this.timeline = await tlRes.json();
    } else {
      this.timeline = { fps: 30, numFrames: 1, duration: 0, frames: [this._defaultFrame(0)] };
    }

    // Build Hologram Bust (use min(width, height) so micro-dot size stays crisp & identical in 16:9 and 9:16!)
    const refRes = Math.min(width, height);
    this.hasDriving = this.timeline.mode === 'driving_video';
    this.bust = new HologramBust(meta, floatData, palette, {
      intensity: config.intensity ?? 1.0,
      density: config.density ?? 1.0,
      pointScale: config.pointScale ?? 1.08,
      height: refRes,
      pixelRatio: 1,
      hasDriving: this.hasDriving,
    });
    this.scene.add(this.bust.group);

    // Build Optional Floating 3D HUD (auto-adapts cleanly to portrait 9:16 vs landscape 16:9!)
    if (config.hud !== false) {
      this.hud = new HoloHUD(palette, { isPortrait: aspect < 0.85 });
      this.scene.add(this.hud.group);
    }

    // Build Optional Emitter Base + Projection Rays
    if (config.emitter !== false) {
      this.emitter = new EmitterBase(palette, { intensity: config.intensity ?? 1.0 });
      this.scene.add(this.emitter.group);
    }

    // Setup Post-Processing (Alpha-Safe 2-Scale HDR Bloom + Composite)
    this._setupPostProcess(width, height, bgModeInt, palette, config);

    return {
      ready: true,
      width,
      height,
      numFrames: this.timeline.numFrames,
      fps: this.timeline.fps,
      duration: this.timeline.duration,
    };
  }

  _defaultFrame(t) {
    return {
      t,
      energy: 0.2,
      jawOpen: 0,
      mouthFunnel: 0,
      mouthWide: 0,
      mouthClose: 0,
      mouthF: 0,
      upperLipRaise: 0,
      blink: 0,
      browRaise: 0,
      headYaw: 0,
      headPitch: 0,
      headRoll: 0,
      glitch: 0,
      spectrum: [0.2, 0.35, 0.45, 0.3, 0.25, 0.2, 0.15, 0.1],
    };
  }

  _setupPostProcess(width, height, bgModeInt, palette, config) {
    const rtOpts = {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: true,
      stencilBuffer: false,
    };
    const blurOpts = { ...rtOpts, depthBuffer: false };

    this.sceneRT = new THREE.WebGLRenderTarget(width, height, rtOpts);

    // Tight bloom at 1/2 resolution, wide atmospheric halo at 1/4 resolution
    const w2 = Math.max(2, Math.floor(width / 2));
    const h2 = Math.max(2, Math.floor(height / 2));
    const w4 = Math.max(2, Math.floor(width / 4));
    const h4 = Math.max(2, Math.floor(height / 4));

    this.bloomTightA = new THREE.WebGLRenderTarget(w2, h2, blurOpts);
    this.bloomTightB = new THREE.WebGLRenderTarget(w2, h2, blurOpts);
    this.bloomWideA = new THREE.WebGLRenderTarget(w4, h4, blurOpts);
    this.bloomWideB = new THREE.WebGLRenderTarget(w4, h4, blurOpts);

    this.postScene = new THREE.Scene();
    this.postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const quadGeo = new THREE.PlaneGeometry(2, 2);

    this.extractMat = new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: BLOOM_EXTRACT_FRAG,
      uniforms: {
        tDiffuse: { value: null },
        uThreshold: { value: 0.24 },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.blurMat = new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: BLUR_FRAG,
      uniforms: {
        tDiffuse: { value: null },
        uDirection: { value: new THREE.Vector2(1, 0) },
        uResolution: { value: new THREE.Vector2(w2, h2) },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.compositeMat = new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: COMPOSITE_FRAG,
      uniforms: {
        tScene: { value: this.sceneRT.texture },
        tBloomTight: { value: this.bloomTightA.texture },
        tBloomWide: { value: this.bloomWideA.texture },
        uBloomStrength: { value: config.bloom ?? 1.0 },
        uGlitch: { value: 0 },
        uTime: { value: 0 },
        uBgMode: { value: bgModeInt },
        uColorPrimary: { value: palette.primary.clone() },
        uResolution: { value: new THREE.Vector2(width, height) },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.postQuad = new THREE.Mesh(quadGeo, this.extractMat);
    this.postScene.add(this.postQuad);
    this._dims = { w2, h2, w4, h4 };
  }

  _updateCamera(t) {
    const cfg = this.config;
    const preset = cfg.camera || 'hero';
    const motion = cfg.cameraMotion || 'subtle';

    const isPortrait = (cfg.width || 1920) < (cfg.height || 1080);
    let baseYaw = 0;
    let basePitch = 0.04;
    let baseDist = cfg.emitter !== false ? (isPortrait ? 4.75 : 4.35) : (isPortrait ? 4.30 : 3.95);
    let targetY = cfg.emitter !== false ? (isPortrait ? -0.06 : -0.14) : -0.02;

    if (preset === 'hero') {
      baseYaw = -0.14; // Subtle engaging presenter angle (~8deg) with direct camera eye contact
      basePitch = 0.03;
    } else if (preset === 'front') {
      baseYaw = 0.0;
      basePitch = 0.02;
    } else if (preset === 'left34') {
      baseYaw = 0.36;
      basePitch = 0.04;
    } else if (preset === 'right34') {
      baseYaw = -0.36;
      basePitch = 0.04;
    } else if (preset === 'orbit') {
      baseYaw = -0.55 + t * 0.24;
      basePitch = 0.06;
    }

    if (typeof cfg.cameraYaw === 'number') baseYaw = (cfg.cameraYaw * Math.PI) / 180;
    if (typeof cfg.cameraPitch === 'number') basePitch = (cfg.cameraPitch * Math.PI) / 180;
    if (typeof cfg.cameraZoom === 'number' && cfg.cameraZoom > 0) baseDist /= cfg.cameraZoom;

    let yaw = baseYaw;
    let pitch = basePitch;
    let dist = baseDist;

    if (motion === 'subtle' && preset !== 'orbit') {
      yaw += Math.sin(t * 0.35) * 0.055;
      pitch += Math.cos(t * 0.28) * 0.022;
      dist += Math.sin(t * 0.22) * 0.06;
    } else if (motion === 'orbit') {
      yaw = baseYaw + t * 0.28;
    }

    const cx = Math.sin(yaw) * Math.cos(pitch) * dist;
    const cy = targetY + Math.sin(pitch) * dist;
    const cz = Math.cos(yaw) * Math.cos(pitch) * dist;

    this.camera.position.set(cx, cy, cz);
    this.camera.lookAt(0, targetY, 0.02);
  }

  _runBlur(srcTex, rtA, rtB, w, h, passes, spread) {
    this.blurMat.uniforms.uResolution.value.set(w, h);
    this.postQuad.material = this.blurMat;

    let readTex = srcTex;
    for (let i = 0; i < passes; i++) {
      const s = spread * (1 + i * 0.65);
      // Horizontal
      this.blurMat.uniforms.tDiffuse.value = readTex;
      this.blurMat.uniforms.uDirection.value.set(s, 0);
      this.renderer.setRenderTarget(rtB);
      this.renderer.render(this.postScene, this.postCam);

      // Vertical
      this.blurMat.uniforms.tDiffuse.value = rtB.texture;
      this.blurMat.uniforms.uDirection.value.set(0, s);
      this.renderer.setRenderTarget(rtA);
      this.renderer.render(this.postScene, this.postCam);

      readTex = rtA.texture;
    }
  }

  async renderFrame(frameIdx) {
    const frames = this.timeline.frames;
    const idx = Math.max(0, Math.min(frames.length - 1, frameIdx));
    const frame = frames[idx];

    if (this.hasDriving && frame.drivingFrame && this.config.drivingBaseUrl) {
      const resp = await fetch(`${this.config.drivingBaseUrl}/${frame.drivingFrame}`);
      const blob = await resp.blob();
      const bmp = await createImageBitmap(blob);
      this.bust.setDrivingFrameImage(bmp, this.timeline.drivingBounds);
    }

    this._updateCamera(frame.t);
    this.bust.update(frame, this.camera, this.config.lookAtCamera !== false);
    if (this.hud) this.hud.update(frame);
    if (this.emitter) this.emitter.update(frame);

    // 1. Render 3D scene to HalfFloat RGBA target
    this.renderer.setRenderTarget(this.sceneRT);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear(true, true, false);
    this.renderer.render(this.scene, this.camera);

    // 2. Extract bright regions for tight bloom
    this.extractMat.uniforms.tDiffuse.value = this.sceneRT.texture;
    this.postQuad.material = this.extractMat;
    this.renderer.setRenderTarget(this.bloomTightA);
    this.renderer.render(this.postScene, this.postCam);

    // 3. Separable blur tight bloom (half res)
    this._runBlur(this.bloomTightA.texture, this.bloomTightA, this.bloomTightB, this._dims.w2, this._dims.h2, 2, 1.8);

    // 4. Downsample & wide atmospheric cyan halo blur (quarter res)
    this._runBlur(this.bloomTightA.texture, this.bloomWideA, this.bloomWideB, this._dims.w4, this._dims.h4, 3, 2.8);

    // 5. Final Alpha-Safe Composite Pass to screen canvas
    this.compositeMat.uniforms.uGlitch.value = frame.glitch;
    this.compositeMat.uniforms.uTime.value = frame.t;
    this.postQuad.material = this.compositeMat;
    this.renderer.setRenderTarget(null);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear(true, false, false);
    this.renderer.render(this.postScene, this.postCam);
    this.renderer.getContext().finish();
  }
}

window.holome = new HoloEngine();
