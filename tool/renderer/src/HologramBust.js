import * as THREE from 'three';
import {
  POINT_CLOUD_VERT,
  POINT_CLOUD_FRAG,
  INNER_SHELL_VERT,
  INNER_SHELL_FRAG,
  DEPTH_OCCLUDER_FRAG,
  CONSTELLATION_VERT,
  CONSTELLATION_FRAG,
} from './shaders.js';

export class HologramBust {
  constructor(meta, floatData, palette, options = {}) {
    this.meta = meta;
    this.group = new THREE.Group();
    this.group.position.y = 0.08;
    this.options = options;

    const stride = meta.strideFloats; // 14
    const numVerts = meta.numVertices;
    const rows = meta.gridRows;
    const cols = meta.gridCols;

    const positions = new Float32Array(numVerts * 3);
    const normals = new Float32Array(numVerts * 3);
    const lum = new Float32Array(numVerts);
    const edge = new Float32Array(numVerts);
    const headW = new Float32Array(numVerts);
    const jawW = new Float32Array(numVerts);
    const lipW = new Float32Array(numVerts);
    const blinkW = new Float32Array(numVerts);
    const browW = new Float32Array(numVerts);
    const vNorm = new Float32Array(numVerts);
    const rand = new Float32Array(numVerts);

    // Deterministic LCG PRNG for reproducible particle dissolve & shimmer
    let seed = 133742;
    const nextRand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };

    for (let i = 0; i < numVerts; i++) {
      const o = i * stride;
      positions[i * 3 + 0] = floatData[o + 0];
      positions[i * 3 + 1] = floatData[o + 1];
      positions[i * 3 + 2] = floatData[o + 2];

      normals[i * 3 + 0] = floatData[o + 3];
      normals[i * 3 + 1] = floatData[o + 4];
      normals[i * 3 + 2] = floatData[o + 5];

      lum[i] = floatData[o + 6];
      edge[i] = floatData[o + 7];
      headW[i] = floatData[o + 8];
      jawW[i] = floatData[o + 9];
      lipW[i] = floatData[o + 10];
      blinkW[i] = floatData[o + 11];
      browW[i] = floatData[o + 12];
      vNorm[i] = floatData[o + 13];
      rand[i] = nextRand();
    }

    // 1x1 dummy texture fallback so WebGL2 sampler is always valid even in procedural mode
    const dummyTex = new THREE.DataTexture(new Uint8Array([128, 0, 128, 255]), 1, 1, THREE.RGBAFormat);
    dummyTex.needsUpdate = true;
    this.drivingTex = new THREE.Texture();
    this.drivingTex.minFilter = THREE.LinearFilter;
    this.drivingTex.magFilter = THREE.LinearFilter;
    this.drivingTex.generateMipmaps = false;

    const pivots = meta.pivots;
    this.sharedUniforms = {
      uJawPivot: { value: new THREE.Vector3(...pivots.jaw) },
      uNeckPivot: { value: new THREE.Vector3(...pivots.neck) },
      uMouthCenter: { value: new THREE.Vector3(...pivots.mouthCenter) },
      uMouthHalfW: { value: pivots.mouthHalfWidth },
      uTime: { value: 0 },
      uJawOpen: { value: 0 },
      uMouthFunnel: { value: 0 },
      uMouthWide: { value: 0 },
      uMouthClose: { value: 0 },
      uMouthF: { value: 0 },
      uUpperLipRaise: { value: 0 },
      uBlink: { value: 0 },
      uBrowRaise: { value: 0 },
      uHeadRot: { value: new THREE.Vector3(0, 0, 0) },
      uGlitch: { value: 0 },
      uAudioEnergy: { value: 0 },
      uColorPrimary: { value: palette.primary.clone() },
      uColorHot: { value: palette.hot.clone() },
      uColorDeep: { value: palette.deep.clone() },
      uIntensity: { value: options.intensity ?? 1.0 },
      uDensity: { value: options.density ?? 1.0 },
      uPixelRatio: { value: options.pixelRatio ?? 1.0 },
      uResolutionY: { value: options.height ?? 1080 },
      uPointScale: { value: options.pointScale ?? 1.05 },
      uHasDriving: { value: options.hasDriving ? 1.0 : 0.0 },
      uDrivingTex: { value: dummyTex },
      uDrivingBounds: { value: new THREE.Vector4(-0.58, 0.58, -0.62, 0.78) },
    };

    // --- Layer 1: Inner Translucent Fresnel Core Mesh (writes depth for crystal-clear face readability) ---
    // Cut ONLY the exact 1-row lip-parting seam (procedural mode) so mouth closes 100% watertight when jawOpen=0
    // and opens cleanly without stretched triangles when jawOpen > 0!
    const isDriving = !!options.hasDriving;
    const mcX = pivots.mouthCenter[0];
    const mcZ = pivots.mouthCenter[2];
    const slitHalfW = pivots.mouthHalfWidth * 1.06;
    const crossesMouthSlit = (a, b, c) => {
      if (isDriving) return false;
      const cx = (positions[a * 3] + positions[b * 3] + positions[c * 3]) / 3;
      const cz = (positions[a * 3 + 2] + positions[b * 3 + 2] + positions[c * 3 + 2]) / 3;
      // Only sever triangles inside the actual front mouth aperture (keep outer cheeks 100% watertight!)
      if (Math.abs(cx - mcX) > slitHalfW || cz < mcZ - 0.24) return false;
      const la = lipW[a], lb = lipW[b], lc = lipW[c];
      if (Math.max(la, lb, lc) > 0.02 && Math.min(la, lb, lc) < -0.02) return true;
      const ja = jawW[a], jb = jawW[b], jc = jawW[c];
      return (Math.max(ja, jb, jc) - Math.min(ja, jb, jc)) > 0.30;
    };

    if (rows > 1 && cols > 1 && rows * cols === numVerts) {
      const indices = [];
      for (let r = 0; r < rows - 1; r++) {
        for (let c = 0; c < cols; c++) {
          const cNext = (c + 1) % cols;
          const i00 = r * cols + c;
          const i01 = r * cols + cNext;
          const i10 = (r + 1) * cols + c;
          const i11 = (r + 1) * cols + cNext;
          if (!crossesMouthSlit(i00, i10, i01)) indices.push(i00, i10, i01);
          if (!crossesMouthSlit(i01, i10, i11)) indices.push(i01, i10, i11);
        }
      }
      const shellGeo = new THREE.BufferGeometry();
      shellGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      shellGeo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
      shellGeo.setAttribute('aLum', new THREE.BufferAttribute(lum, 1));
      shellGeo.setAttribute('aEdge', new THREE.BufferAttribute(edge, 1));
      shellGeo.setAttribute('aHeadW', new THREE.BufferAttribute(headW, 1));
      shellGeo.setAttribute('aJawW', new THREE.BufferAttribute(jawW, 1));
      shellGeo.setAttribute('aLipW', new THREE.BufferAttribute(lipW, 1));
      shellGeo.setAttribute('aBlinkW', new THREE.BufferAttribute(blinkW, 1));
      shellGeo.setAttribute('aBrowW', new THREE.BufferAttribute(browW, 1));
      shellGeo.setAttribute('aVNorm', new THREE.BufferAttribute(vNorm, 1));
      shellGeo.setIndex(indices);

      // Hardware Z-buffer depth pre-pass (opaque queue, renderOrder -10):
      // polygonOffset avoids Z-fighting with shellMesh while strictly occluding back of bust & emitter base!
      const depthMat = new THREE.ShaderMaterial({
        vertexShader: INNER_SHELL_VERT,
        fragmentShader: DEPTH_OCCLUDER_FRAG,
        uniforms: this.sharedUniforms,
        transparent: false,
        colorWrite: false,
        depthWrite: true,
        depthTest: true,
        polygonOffset: true,
        polygonOffsetFactor: 2.0,
        polygonOffsetUnits: 2.0,
        side: THREE.FrontSide,
      });
      this.depthOccluder = new THREE.Mesh(shellGeo, depthMat);
      this.depthOccluder.renderOrder = -10;
      this.group.add(this.depthOccluder);

      const shellMat = new THREE.ShaderMaterial({
        vertexShader: INNER_SHELL_VERT,
        fragmentShader: INNER_SHELL_FRAG,
        uniforms: this.sharedUniforms,
        transparent: true,
        depthWrite: true,
        depthTest: true,
        side: THREE.FrontSide,
        blending: THREE.NormalBlending,
      });

      this.shellMesh = new THREE.Mesh(shellGeo, shellMat);
      this.shellMesh.renderOrder = 1;
      this.group.add(this.shellMesh);

      // --- Layer 3: Constellation Wireframe Network (prominent along hoodie shoulders & outer silhouette) ---
      this._buildConstellation(
        rows, cols, positions, normals, headW, jawW, lipW, blinkW, browW, vNorm, nextRand
      );
    }

    // --- Layer 2: Dense Structured Holographic Point Cloud (Hero Layer!) ---
    const ptGeo = new THREE.BufferGeometry();
    ptGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    ptGeo.setAttribute('aNormal', new THREE.BufferAttribute(normals, 3));
    ptGeo.setAttribute('aLum', new THREE.BufferAttribute(lum, 1));
    ptGeo.setAttribute('aEdge', new THREE.BufferAttribute(edge, 1));
    ptGeo.setAttribute('aHeadW', new THREE.BufferAttribute(headW, 1));
    ptGeo.setAttribute('aJawW', new THREE.BufferAttribute(jawW, 1));
    ptGeo.setAttribute('aLipW', new THREE.BufferAttribute(lipW, 1));
    ptGeo.setAttribute('aBlinkW', new THREE.BufferAttribute(blinkW, 1));
    ptGeo.setAttribute('aBrowW', new THREE.BufferAttribute(browW, 1));
    ptGeo.setAttribute('aVNorm', new THREE.BufferAttribute(vNorm, 1));
    ptGeo.setAttribute('aRand', new THREE.BufferAttribute(rand, 1));

    const ptMat = new THREE.ShaderMaterial({
      vertexShader: POINT_CLOUD_VERT,
      fragmentShader: POINT_CLOUD_FRAG,
      uniforms: this.sharedUniforms,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(ptGeo, ptMat);
    this.points.renderOrder = 3;
    this.group.add(this.points);

    // --- Layer 4: High-Contrast Open Mouth Interior + Camera-Tracking Luminous Holo Eyes (procedural mode only!) ---
    if (!options.hasDriving) {
      this._buildVocalCore(pivots, palette);
      this._buildHoloEyes(pivots, palette);
    }

    // --- Layer 5: Floating Holographic Ambient Dust Motes & Vertical Light Streaks ---
    this._buildAmbientMotes(palette, nextRand);
    this._buildVerticalStreaks(palette);
  }

  _buildConstellation(rows, cols, positions, normals, headW, jawW, lipW, blinkW, browW, vNorm, nextRand) {
    // Subsampled triangulated web across the bust (denser on hoodie shoulders/collar like the reference)
    const stepR = 10;
    const stepC = 11;
    const segPositions = [];
    const segNormals = [];
    const segHeadW = [];
    const segJawW = [];
    const segLipW = [];
    const segBlinkW = [];
    const segBrowW = [];
    const segVNorm = [];
    const segStrength = [];

    const pushVert = (idx, strength) => {
      segPositions.push(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
      segNormals.push(normals[idx * 3], normals[idx * 3 + 1], normals[idx * 3 + 2]);
      segHeadW.push(headW[idx]);
      segJawW.push(jawW[idx]);
      segLipW.push(lipW[idx]);
      segBlinkW.push(blinkW[idx]);
      segBrowW.push(browW[idx]);
      segVNorm.push(vNorm[idx]);
      segStrength.push(strength);
    };

    for (let r = 8; r < rows - stepR; r += stepR) {
      for (let c = 0; c < cols; c += stepC) {
        const c2 = (c + stepC) % cols;
        const r2 = Math.min(rows - 1, r + stepR);
        const i0 = r * cols + c;
        const i1 = r * cols + c2;
        const i2 = r2 * cols + c;
        const i3 = r2 * cols + c2;

        const str = 0.36 + 0.44 * nextRand();
        const crosses = (a, b) =>
          (lipW[a] > 0.005 && lipW[b] < -0.005) ||
          (lipW[b] > 0.005 && lipW[a] < -0.005) ||
          Math.abs(jawW[a] - jawW[b]) > 0.14;
        if (nextRand() < 0.78 && !crosses(i0, i1)) {
          pushVert(i0, str); pushVert(i1, str);
        }
        if (nextRand() < 0.78 && !crosses(i0, i2)) {
          pushVert(i0, str); pushVert(i2, str);
        }
        if (nextRand() < 0.62 && !crosses(i0, i3)) {
          pushVert(i0, str * 0.88); pushVert(i3, str * 0.88);
        }
      }
    }

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(segPositions, 3));
    lineGeo.setAttribute('aNormal', new THREE.Float32BufferAttribute(segNormals, 3));
    lineGeo.setAttribute('aHeadW', new THREE.Float32BufferAttribute(segHeadW, 1));
    lineGeo.setAttribute('aJawW', new THREE.Float32BufferAttribute(segJawW, 1));
    lineGeo.setAttribute('aLipW', new THREE.Float32BufferAttribute(segLipW, 1));
    lineGeo.setAttribute('aBlinkW', new THREE.Float32BufferAttribute(segBlinkW, 1));
    lineGeo.setAttribute('aBrowW', new THREE.Float32BufferAttribute(segBrowW, 1));
    lineGeo.setAttribute('aVNorm', new THREE.Float32BufferAttribute(segVNorm, 1));
    lineGeo.setAttribute('aLineStrength', new THREE.Float32BufferAttribute(segStrength, 1));

    const lineMat = new THREE.ShaderMaterial({
      vertexShader: CONSTELLATION_VERT,
      fragmentShader: CONSTELLATION_FRAG,
      uniforms: this.sharedUniforms,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
    });

    this.constellation = new THREE.LineSegments(lineGeo, lineMat);
    this.constellation.renderOrder = 2;
    this.group.add(this.constellation);
  }

  _buildVocalCore(pivots, palette) {
    // High-contrast dark holographic mouth opening + subtle upper-teeth highlight when speaking!
    // Renders BEFORE point cloud with depthWrite:true so back-of-head dots are 100% occluded inside the open mouth.
    const mw = pivots.mouthHalfWidth * 1.95;
    const mh = pivots.mouthHalfWidth * 1.35;
    // Curve the mouth cavity slightly backward along the face cylinder so its corners never clip through cheeks!
    const geo = new THREE.PlaneGeometry(mw, mh, 16, 8);
    const posAttr = geo.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const lx = posAttr.getX(i);
      const ly = posAttr.getY(i);
      posAttr.setZ(i, -(lx * lx) * 1.85 - (ly * ly) * 0.65);
    }
    posAttr.needsUpdate = true;
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uColorPrimary: { value: palette.primary.clone() },
        uColorHot: { value: palette.hot.clone() },
        uColorDeep: { value: palette.deep.clone() },
        uOpen: { value: 0 },
        uFunnel: { value: 0 },
        uWide: { value: 0 },
        uIntensity: this.sharedUniforms.uIntensity,
      },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform vec3 uColorPrimary;
        uniform vec3 uColorHot;
        uniform vec3 uColorDeep;
        uniform float uOpen;
        uniform float uFunnel;
        uniform float uWide;
        uniform float uIntensity;
        varying vec2 vUv;

        void main() {
          if (uOpen < 0.04) discard;
          // Shape the mouth aperture dynamically: wider on E/I, rounder O-ring on O/U, taller with jawOpen!
          vec2 p = (vUv - 0.5) * 2.0;
          float widthScale = 1.0 + uWide * 0.22 - uFunnel * 0.34;
          float heightScale = max(0.08, uOpen * (0.85 + uFunnel * 0.25));
          vec2 q = vec2(p.x / max(widthScale, 0.35), p.y / heightScale);
          // Natural human mouth arch (slightly flatter top under mustache, deeper bottom curve)
          float archOffset = 0.14 * q.x * q.x;
          float d = length(vec2(q.x, q.y - archOffset));

          // Natural feathered dark mouth cavity behind the parting 3D lip dots (zero drawn-on ring!)
          float cavity = smoothstep(1.0, 0.35, d) * smoothstep(0.04, 0.25, uOpen);
          if (cavity < 0.02) discard;

          // Deep dark interior shadow so the opening reads unmistakably through the mustache & goatee
          vec3 col = uColorDeep * 0.020;

          // Fine holographic dot-matrix texture matching the 108k-point bust!
          vec2 dotCell = fract(vUv * vec2(32.0, 18.0)) - 0.5;
          float dotTex = 0.45 + 0.55 * exp(-dot(dotCell, dotCell) * 10.0);

          // Upper teeth arch peeking naturally beneath upper lip (strongest on AA/EE/F/V, tucked on OO)
          float upperTeeth = smoothstep(0.24, 0.50, q.y) * smoothstep(0.95, 0.64, q.y)
                           * smoothstep(0.78, 0.28, abs(q.x))
                           * smoothstep(0.14, 0.48, uOpen) * (1.0 - uFunnel * 0.55);
          // Subtle recessed tongue warmth & lower teeth only when mouth opens wide
          float tongue = smoothstep(-0.15, -0.58, q.y) * smoothstep(-0.98, -0.62, q.y)
                       * smoothstep(0.62, 0.15, abs(q.x))
                       * smoothstep(0.36, 0.80, uOpen);

          col += mix(uColorPrimary, uColorHot, 0.35) * upperTeeth * dotTex * 0.52 * uIntensity;
          col += uColorPrimary * tongue * dotTex * 0.24 * uIntensity;

          gl_FragColor = vec4(col, cavity * 0.96);
        }
      `,
      transparent: true,
      depthWrite: true,
      depthTest: true,
      blending: THREE.NormalBlending,
    });
    this.vocalCore = new THREE.Mesh(geo, mat);
    this.vocalCoreBasePos = new THREE.Vector3(
      pivots.mouthCenter[0],
      pivots.mouthCenter[1] - 0.008,
      pivots.mouthCenter[2] - 0.085
    );
    this.vocalCore.position.copy(this.vocalCoreBasePos);
    this.vocalCore.renderOrder = 2;
    this.group.add(this.vocalCore);

    // Watertight 3D inner-oral-cavity depth occluder (opaque queue, renderOrder -9):
    // Completely seals the hollow interior behind the open mouth & corners so rear bust/collar/base
    // geometry can NEVER bleed through the mouth opening from any camera angle!
    const occGeo = new THREE.SphereGeometry(1.0, 24, 18);
    const occRadX = Math.max(0.24, pivots.mouthHalfWidth * 2.15);
    const occRadY = Math.max(0.21, pivots.mouthHalfWidth * 1.85);
    const occRadZ = 0.24;
    occGeo.scale(occRadX, occRadY, occRadZ);
    const occMat = new THREE.MeshBasicMaterial({
      colorWrite: false,
      depthWrite: true,
      depthTest: true,
      transparent: false,
      side: THREE.FrontSide,
    });
    this.mouthDepthBlocker = new THREE.Mesh(occGeo, occMat);
    this.mouthDepthBlockerBasePos = new THREE.Vector3(
      pivots.mouthCenter[0],
      pivots.mouthCenter[1] - 0.045,
      pivots.mouthCenter[2] - 0.39
    );
    this.mouthDepthBlocker.position.copy(this.mouthDepthBlockerBasePos);
    this.mouthDepthBlocker.renderOrder = -9;
    this.group.add(this.mouthDepthBlocker);
  }

  _buildHoloEyes(pivots, palette) {
    // Seamless holographic dot-matrix eyes with live camera eye-contact tracking + crisp blink curtain!
    this.holoEyes = [];
    const eyeSpecs = [
      {
        eye: pivots.leftEye || [-0.205, 0.345, 0.085, 0.026],
        iris: pivots.leftIris || [-0.205, 0.345, 0.56],
      },
      {
        eye: pivots.rightEye || [0.200, 0.343, 0.085, 0.026],
        iris: pivots.rightIris || [0.200, 0.343, 0.56],
      },
    ];

    this.eyeUniforms = {
      uColorPrimary: { value: palette.primary.clone() },
      uColorHot: { value: palette.hot.clone() },
      uColorDeep: { value: palette.deep.clone() },
      uBlink: { value: 0 },
      uGaze: { value: new THREE.Vector2(0, 0) },
      uIntensity: this.sharedUniforms.uIntensity,
      uTime: this.sharedUniforms.uTime,
    };

    const eyeMat = new THREE.ShaderMaterial({
      uniforms: this.eyeUniforms,
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform vec3 uColorPrimary;
        uniform vec3 uColorHot;
        uniform vec3 uColorDeep;
        uniform float uBlink;
        uniform vec2 uGaze;
        uniform float uIntensity;
        uniform float uTime;
        varying vec2 vUv;

        void main() {
          // Soft feathered almond eye aperture: x in [-1..1], y in [-1..1]
          vec2 p = (vUv - 0.5) * 2.0;
          float upperArch = 0.86 * (1.0 - 0.78 * p.x * p.x);
          float lowerArch = -0.74 * (1.0 - 0.82 * p.x * p.x);
          if (p.y > upperArch + 0.16 || p.y < lowerArch - 0.16 || abs(p.x) > 1.0) discard;

          float xFade = smoothstep(1.0, 0.62, abs(p.x));
          float insideEye = smoothstep(upperArch + 0.12, upperArch - 0.22, p.y)
                          * smoothstep(lowerArch - 0.12, lowerArch + 0.22, p.y) * xFade;
          if (insideEye < 0.01) discard;

          // Upper eyelid blink curtain sweeps smoothly down over eyeball as uBlink: 0 -> 1
          float lidY = mix(upperArch + 0.08, lowerArch - 0.06, uBlink);
          float lidOpenMask = smoothstep(lidY + 0.12, lidY - 0.14, p.y);
          float lidRimGlow = smoothstep(0.28, 0.0, abs(p.y - lidY)) * insideEye * (0.20 + 0.45 * uBlink);

          // Gaze-shifted iris + dark pupil + crisp specular catchlight looking right at camera!
          vec2 irisCenter = clamp(uGaze, vec2(-0.30, -0.18), vec2(0.30, 0.18));
          vec2 dp = vec2((p.x - irisCenter.x) * 1.72, (p.y - irisCenter.y) * 1.08);
          float r = length(dp);

          // Fine holographic dot-matrix texture matching the bust point cloud!
          vec2 grid = fract(vUv * vec2(26.0, 11.0)) - 0.5;
          float dotMask = 0.52 + 0.48 * exp(-dot(grid, grid) * 9.0);

          float sclera = insideEye * lidOpenMask * 0.24;
          float irisRing = smoothstep(0.56, 0.40, r) * smoothstep(0.18, 0.34, r) * insideEye * lidOpenMask;
          float pupil = smoothstep(0.27, 0.14, r) * insideEye * lidOpenMask;
          float glint = exp(-dot(dp - vec2(-0.14, 0.15), dp - vec2(-0.14, 0.15)) * 48.0) * insideEye * lidOpenMask;

          vec3 openCol = uColorPrimary * sclera * 0.72
                       + mix(uColorPrimary, uColorHot, 0.48) * irisRing * 0.95
                       + uColorHot * glint * 1.45;
          openCol = mix(openCol, uColorDeep * 0.06, pupil * 0.88);
          openCol *= dotMask;

          // Closed eyelid darkens cleanly over eye socket with subtle glowing eyelid crease!
          float closedCover = insideEye * (1.0 - lidOpenMask);
          vec3 closedCol = uColorDeep * 0.08 + uColorPrimary * lidRimGlow * dotMask * 0.65;

          vec3 finalCol = (openCol + closedCol) * uIntensity;
          float alpha = clamp(insideEye * (0.64 * lidOpenMask + 0.88 * closedCover) + lidRimGlow * 0.35, 0.0, 0.88);
          if (alpha < 0.02) discard;

          gl_FragColor = vec4(finalCol, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.NormalBlending,
    });

    for (const spec of eyeSpecs) {
      const ew = Math.max(0.125, spec.eye[2] * 1.82);
      const eh = Math.max(0.052, spec.eye[3] * 2.15);
      const geo = new THREE.PlaneGeometry(ew, eh);
      const mesh = new THREE.Mesh(geo, eyeMat);
      const basePos = new THREE.Vector3(spec.iris[0], spec.iris[1], spec.iris[2] + 0.008);
      mesh.position.copy(basePos);
      mesh.renderOrder = 4;
      this.group.add(mesh);
      this.holoEyes.push({ mesh, basePos });
    }
  }

  _buildAmbientMotes(palette, nextRand) {
    const count = 420;
    const pos = new Float32Array(count * 3);
    const phase = new Float32Array(count);
    const size = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const angle = nextRand() * Math.PI * 2;
      const radius = 0.45 + nextRand() * 0.95;
      pos[i * 3 + 0] = Math.cos(angle) * radius;
      pos[i * 3 + 1] = -1.15 + nextRand() * 2.35;
      pos[i * 3 + 2] = Math.sin(angle) * radius * 0.75;
      phase[i] = nextRand() * 10.0;
      size[i] = 1.4 + nextRand() * 3.0;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aPhase', new THREE.BufferAttribute(phase, 1));
    geo.setAttribute('aSize', new THREE.BufferAttribute(size, 1));

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: this.sharedUniforms.uTime,
        uColor: { value: palette.hot.clone() },
        uColorPrimary: { value: palette.primary.clone() },
        uAudioEnergy: this.sharedUniforms.uAudioEnergy,
        uResolutionY: this.sharedUniforms.uResolutionY,
        uIntensity: this.sharedUniforms.uIntensity,
      },
      vertexShader: /* glsl */ `
        attribute float aPhase;
        attribute float aSize;
        uniform float uTime;
        uniform float uResolutionY;
        varying float vAlpha;
        void main() {
          vec3 p = position;
          float life = fract(aPhase * 0.1 + uTime * 0.09);
          p.y = mix(-1.12, 1.28, life);
          p.x += sin(uTime * 0.7 + aPhase) * 0.04;
          p.z += cos(uTime * 0.55 + aPhase * 1.3) * 0.04;

          vec4 mvPos = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mvPos;

          float fade = sin(life * 3.14159265);
          float twinkle = 0.55 + 0.45 * sin(uTime * 3.5 + aPhase * 6.0);
          vAlpha = fade * twinkle * 0.48;

          gl_PointSize = clamp(aSize * (uResolutionY / 1080.0) * (2.4 / max(-mvPos.z, 0.5)), 1.0, 9.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor;
        uniform vec3 uColorPrimary;
        uniform float uIntensity;
        varying float vAlpha;
        void main() {
          vec2 uv = gl_PointCoord - 0.5;
          float r2 = dot(uv, uv);
          if (r2 > 0.25) discard;
          float g = exp(-r2 * 14.0);
          vec3 col = mix(uColorPrimary, uColor, g * 0.7) * uIntensity;
          gl_FragColor = vec4(col * g * vAlpha, g * vAlpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.motes = new THREE.Points(geo, mat);
    this.motes.renderOrder = 4;
    this.group.add(this.motes);
  }

  _buildVerticalStreaks(palette) {
    // Delicate vertical cyan projection rays rising gently off top of head & shoulders (just like reference!)
    const geo = new THREE.PlaneGeometry(1.75, 2.55);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: this.sharedUniforms.uTime,
        uColorPrimary: { value: palette.primary.clone() },
        uAudioEnergy: this.sharedUniforms.uAudioEnergy,
        uIntensity: this.sharedUniforms.uIntensity,
      },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform vec3 uColorPrimary;
        uniform float uAudioEnergy;
        uniform float uIntensity;
        varying vec2 vUv;
        void main() {
          float xDist = abs(vUv.x - 0.5) * 2.0;
          float xMask = exp(-xDist * xDist * 3.2);
          float yMask = smoothstep(0.0, 0.22, vUv.y) * pow(max(1.0 - vUv.y, 0.0001), 0.95);
          float streaks = 0.35
            + 0.30 * sin(vUv.x * 95.0 + uTime * 0.4)
            + 0.22 * sin(vUv.x * 210.0 - uTime * 0.7)
            + 0.18 * sin(vUv.x * 43.0 + uTime * 0.25);
          streaks = pow(max(clamp(streaks, 0.0, 1.0), 0.0001), 2.2);
          float alpha = xMask * yMask * streaks * (0.16 + 0.045 * uAudioEnergy) * uIntensity;
          gl_FragColor = vec4(uColorPrimary * alpha, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });

    this.streaks = new THREE.Mesh(geo, mat);
    this.streaks.position.set(0, 0.22, -0.18);
    this.streaks.renderOrder = 0;
    this.group.add(this.streaks);
  }

  setDrivingFrameImage(bitmap, bounds) {
    if (bounds && bounds.length === 4) {
      this.sharedUniforms.uDrivingBounds.value.set(bounds[0], bounds[1], bounds[2], bounds[3]);
    }
    if (this._prevBitmap && typeof this._prevBitmap.close === 'function') {
      this._prevBitmap.close();
    }
    this._prevBitmap = bitmap;
    this.drivingTex.image = bitmap;
    this.drivingTex.needsUpdate = true;
    this.sharedUniforms.uDrivingTex.value = this.drivingTex;
    this.sharedUniforms.uHasDriving.value = 1.0;
  }

  update(frameState, camera = null, lookAtCamera = true) {
    const u = this.sharedUniforms;
    u.uTime.value = frameState.t;
    u.uJawOpen.value = frameState.jawOpen;
    u.uMouthFunnel.value = frameState.mouthFunnel;
    u.uMouthWide.value = frameState.mouthWide;
    u.uMouthClose.value = frameState.mouthClose;
    u.uMouthF.value = frameState.mouthF;
    u.uUpperLipRaise.value = frameState.upperLipRaise;
    u.uBlink.value = frameState.blink;
    u.uBrowRaise.value = frameState.browRaise;

    // If lookAtCamera is enabled, subtly aim the head toward the camera lens so it makes natural eye contact!
    let effYaw = frameState.headYaw;
    let effPitch = frameState.headPitch;
    let camYaw = 0;
    let camPitch = 0;
    if (camera) {
      const toCam = camera.position.clone().sub(this.group.position);
      camYaw = Math.atan2(toCam.x, toCam.z);
      camPitch = Math.atan2(toCam.y - 0.25, Math.hypot(toCam.x, toCam.z));
      if (lookAtCamera) {
        // Turn head ~62% of the way toward camera (clamped to anatomical ±22deg), eyes cover the rest!
        effYaw += Math.max(-0.38, Math.min(0.38, camYaw * 0.62));
        effPitch += Math.max(-0.14, Math.min(0.14, -camPitch * 0.35));
      }
    }

    u.uHeadRot.value.set(effPitch, effYaw, frameState.headRoll);
    u.uGlitch.value = frameState.glitch;
    u.uAudioEnergy.value = frameState.energy;

    const neck = u.uNeckPivot.value;
    const eul = new THREE.Euler(effPitch, effYaw, frameState.headRoll, 'YXZ');
    const breathY = Math.sin(frameState.t * 1.45) * 0.009;

    // Keep dark mouth interior aligned with rotated head & opening jaw
    if (this.vocalCore) {
      this.vocalCore.material.uniforms.uOpen.value = frameState.jawOpen;
      this.vocalCore.material.uniforms.uFunnel.value = frameState.mouthFunnel;
      this.vocalCore.material.uniforms.uWide.value = frameState.mouthWide;
      const rel = this.vocalCoreBasePos.clone().sub(neck);
      rel.y -= frameState.jawOpen * 0.068;
      rel.applyEuler(eul).add(neck);
      rel.y += breathY;
      this.vocalCore.position.copy(rel);
      this.vocalCore.rotation.copy(eul);
    }

    if (this.mouthDepthBlocker) {
      const relOcc = this.mouthDepthBlockerBasePos.clone().sub(neck);
      relOcc.applyEuler(eul).add(neck);
      relOcc.y += breathY;
      this.mouthDepthBlocker.position.copy(relOcc);
      this.mouthDepthBlocker.rotation.copy(eul);
    }

    // Keep luminous holographic eyes aligned with rotated head + lock iris gaze straight onto camera lens!
    if (this.holoEyes && this.holoEyes.length > 0) {
      this.eyeUniforms.uBlink.value = frameState.blink;
      if (lookAtCamera && camera) {
        // Residual angle between where head faces and exact camera lens + tiny living micro-saccades
        const microX = Math.sin(frameState.t * 1.7) * 0.018 + Math.sin(frameState.t * 3.9) * 0.008;
        const microY = Math.cos(frameState.t * 1.3) * 0.012;
        const gazeX = (camYaw - effYaw) * 0.85 + microX;
        const gazeY = (camPitch + effPitch) * 0.65 + microY;
        this.eyeUniforms.uGaze.value.set(
          Math.max(-0.34, Math.min(0.34, gazeX)),
          Math.max(-0.22, Math.min(0.22, gazeY))
        );
      } else {
        this.eyeUniforms.uGaze.value.set(0, 0);
      }

      for (const eyeObj of this.holoEyes) {
        const rel = eyeObj.basePos.clone().sub(neck);
        rel.applyEuler(eul).add(neck);
        rel.y += breathY;
        eyeObj.mesh.position.copy(rel);
        eyeObj.mesh.rotation.copy(eul);
      }
    }
  }
}
