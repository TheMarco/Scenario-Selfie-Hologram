/**
 * HoloMe Custom GLSL Shaders:
 * - Shared anatomical GPU bust deformation (jaw TMJ rotation, lip visemes, blink, brow, head/neck pivot, glitch tear)
 * - Structured holographic point cloud shader (photo detail + edge boost + Fresnel + scanline + bottom dissolve)
 * - Translucent Fresnel inner hologram core shell
 * - Constellation wireframe shoulder/contour shader
 * - Volumetric emitter light beams shader
 * - Alpha-safe multi-scale HDR Bloom + Chromatic Glitch + Filmic Composite post-process shaders
 */

export const DEFORM_GLSL = /* glsl */ `
uniform vec3 uJawPivot;
uniform vec3 uNeckPivot;
uniform vec3 uMouthCenter;
uniform float uMouthHalfW;

uniform float uTime;
uniform float uJawOpen;
uniform float uMouthFunnel;
uniform float uMouthWide;
uniform float uMouthClose;
uniform float uMouthF;
uniform float uUpperLipRaise;
uniform float uBlink;
uniform float uBrowRaise;
uniform vec3 uHeadRot; // x = pitch, y = yaw, z = roll
uniform float uGlitch;
uniform float uAudioEnergy;
uniform float uHasDriving;

// Right-handed GLSL column-major rotation matrices matching THREE.Euler exactly!
mat3 rotX(float a) {
  float c = cos(a), s = sin(a);
  return mat3(1.0, 0.0, 0.0,  0.0, c, s,  0.0, -s, c);
}
mat3 rotY(float a) {
  float c = cos(a), s = sin(a);
  return mat3(c, 0.0, -s,  0.0, 1.0, 0.0,  s, 0.0, c);
}
mat3 rotZ(float a) {
  float c = cos(a), s = sin(a);
  return mat3(c, s, 0.0,  -s, c, 0.0,  0.0, 0.0, 1.0);
}

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

void deformBust(
  in vec3 basePos,
  in vec3 baseNorm,
  in float headW,
  in float jawW,
  in float lipSignedW,
  in float blinkW,
  in float browW,
  out vec3 outPos,
  out vec3 outNorm
) {
  vec3 p = basePos;
  vec3 n = baseNorm;

  // 1. Jaw rotation around anatomical TMJ pivot (+X right-handed rotation swings chin DOWN & slightly back!)
  float jawMult = mix(1.0, 0.32, uHasDriving);
  float jawAngle = uJawOpen * 0.25 * jawW * jawMult;
  if (abs(jawAngle) > 0.0001) {
    mat3 jr = rotX(jawAngle);
    p = uJawPivot + jr * (p - uJawPivot);
    n = jr * n;
    if (uHasDriving < 0.5) {
      float lipLowerMask = clamp(-lipSignedW, 0.0, 1.0);
      float lipUpperMask = clamp(lipSignedW, 0.0, 1.0);
      // Lower lip drops & rolls slightly outward; upper lip lifts gently to reveal 3D upper teeth
      p.y -= uJawOpen * 0.032 * lipLowerMask;
      p.z += uJawOpen * 0.008 * lipLowerMask;
      p.y += uJawOpen * 0.018 * lipUpperMask;
      // Subtle lower-cheek narrowing when jaw opens wide or funnels
      float narrow = (uJawOpen * 0.06 + uMouthFunnel * 0.14) * jawW;
      p.x -= (p.x - uMouthCenter.x) * narrow;
    }
  }

  // 2. Anatomical Multi-Muscle 3D Lip-Roll, Mustache-Flex & Cheek-Puff Rig (procedural mode)
  float lipAbs = abs(lipSignedW) * (1.0 - uHasDriving);
  if (lipAbs > 0.001) {
    vec3 relM = p - uMouthCenter;
    float sideSign = sign(relM.x);
    float normX = abs(relM.x) / max(uMouthHalfW, 0.04);
    float cornerFactor = smoothstep(0.18, 1.0, normX);
    float centerFactor = 1.0 - smoothstep(0.0, 0.95, normX);
    float upperOnly = clamp(lipSignedW, 0.0, 1.0);
    float lowerOnly = clamp(-lipSignedW, 0.0, 1.0);

    // Orbicularis Oris — Funnel / Pucker (O / U / W / R / SH):
    // Pull corners inward and protrude both lips boldly forward (+Z) with outward curl!
    p.x -= relM.x * (0.44 * uMouthFunnel * lipAbs);
    p.z += (0.088 * uMouthFunnel * lipAbs) * (0.45 + 0.55 * centerFactor);
    p.y += lipSignedW * (0.022 * uMouthFunnel);

    // Risorius & Zygomaticus — Wide Spread (E / I / AE / S):
    // Pull mouth corners outward, slightly up, and back along the cheek contour
    p.x += sideSign * cornerFactor * (0.072 * uMouthWide * lipAbs);
    p.y += cornerFactor * (0.022 * uMouthWide * lipAbs) + lipSignedW * (0.010 * uMouthWide);
    p.z -= cornerFactor * (0.024 * uMouthWide * lipAbs);

    // Bilabial Press & Inward Roll (M / B / P): lips roll slightly inward & snap shut cleanly
    p.y -= lipSignedW * (0.028 * uMouthClose);
    p.z -= (0.024 * uMouthClose * lipAbs) * (0.5 + 0.5 * centerFactor);

    // Labiodental Tuck (F / V): lower lip rolls inward & upward under upper teeth
    p.z -= 0.044 * uMouthF * lowerOnly;
    p.y += 0.028 * uMouthF * lowerOnly;
    p.y += 0.012 * uMouthF * upperOnly;

    // Levator Labii — Upper Lip & Mustache Arch Lift on open vowels
    p.y += 0.024 * uUpperLipRaise * upperOnly * (0.4 + 0.6 * centerFactor);
    p.z += 0.014 * uUpperLipRaise * upperOnly * centerFactor;
  }

  // 2b. Coupled Cheek / Nasolabial Muscle Puff on Wide Vowels (E/I) & Speech Emphasis
  if (uHasDriving < 0.5 && (uMouthWide > 0.02 || uJawOpen > 0.05)) {
    vec2 cheekL = (p.xy - vec2(uMouthCenter.x - 0.23, uMouthCenter.y + 0.14)) / vec2(0.15, 0.13);
    vec2 cheekR = (p.xy - vec2(uMouthCenter.x + 0.23, uMouthCenter.y + 0.14)) / vec2(0.15, 0.13);
    float cheekW = (exp(-dot(cheekL, cheekL)) + exp(-dot(cheekR, cheekR))) * clamp(baseNorm.z, 0.0, 1.0) * headW;
    p.y += cheekW * (0.022 * uMouthWide);
    p.z += cheekW * (0.018 * uMouthWide - 0.010 * uMouthFunnel);
  }

  // 3. Natural Eye Blink (procedural mode only — driving video has real blinking eyelids!)
  if (uHasDriving < 0.5 && blinkW > 0.001 && uBlink > 0.001) {
    p.y -= uBlink * 0.056 * blinkW;
    p.z += uBlink * 0.008 * blinkW;
  }

  // 4. Eyebrow Emphasis Raise
  if (uHasDriving < 0.5 && browW > 0.001 && uBrowRaise > 0.001) {
    p.y += uBrowRaise * 0.034 * browW;
    p.z += uBrowRaise * 0.006 * browW;
  }

  // 5. Head & Neck Natural Rotation (smoothly blended to anchored hoodie shoulders)
  // Subtle breathing float for whole bust + organic head rotation weighted by headW
  float breath = sin(uTime * 1.45) * 0.009;
  p.y += breath * (0.45 + 0.55 * headW);

  if (headW > 0.001) {
    mat3 hr = rotY(uHeadRot.y * headW) * rotX(uHeadRot.x * headW) * rotZ(uHeadRot.z * headW);
    p = uNeckPivot + hr * (p - uNeckPivot);
    n = normalize(hr * n);
  }

  // 6. Subtle Holographic Glitch Scan-Tear Displacement
  if (uGlitch > 0.02) {
    float band = floor(p.y * 22.0 + floor(uTime * 18.0) * 1.37);
    float rnd = hash11(band);
    float tearTrigger = step(1.0 - uGlitch * 0.22, rnd);
    float tearShift = (hash11(band + 19.7) - 0.5) * 0.14 * uGlitch * tearTrigger;
    p.x += tearShift;
  }

  outPos = p;
  outNorm = normalize(n);
}
`;

export const POINT_CLOUD_VERT = /* glsl */ `
attribute vec3 aNormal;
attribute float aLum;
attribute float aEdge;
attribute float aHeadW;
attribute float aJawW;
attribute float aLipW;
attribute float aBlinkW;
attribute float aBrowW;
attribute float aVNorm;
attribute float aRand;

uniform float uPixelRatio;
uniform float uResolutionY;
uniform float uPointScale;
uniform float uIntensity;
uniform float uDensity;
uniform sampler2D uDrivingTex;
uniform vec4 uDrivingBounds; // xMin, xMax, yMin, yMax

varying float vBrightness;
varying float vFresnel;
varying float vDissolve;
varying float vScanPulse;
varying float vEdge;
varying float vFacing;
varying float vHeadW;
varying float vGlitchHit;

${DEFORM_GLSL}

void main() {
  vec3 pos;
  vec3 norm;
  deformBust(position, aNormal, aHeadW, aJawW, aLipW, aBlinkW, aBrowW, pos, norm);

  // Subtle upward holographic particle drift near very bottom dissolve fringe
  float botFade = smoothstep(0.84, 0.995, aVNorm);
  if (botFade > 0.01) {
    float driftPhase = fract(uTime * 0.35 + aRand * 3.7);
    pos.y += botFade * driftPhase * 0.06 * step(0.65, aRand);
    pos.x += (aRand - 0.5) * botFade * 0.04;
  }

  vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPos;

  vec3 viewNormal = normalize(normalMatrix * norm + vec3(0.0, 0.0, 1e-5));
  vec3 viewDir = normalize(-mvPos.xyz + vec3(0.0, 0.0, 1e-5));
  float ndotv = dot(viewNormal, viewDir);
  float absNdotV = abs(ndotv);
  float fresnel = pow(max(1.0 - clamp(absNdotV, 0.0, 1.0), 0.0001), 2.15);

  // Strict front-hemisphere culling so back-of-head & back-of-bust-base NEVER bleed through the face!
  float frontFactor = smoothstep(-0.08, 0.16, ndotv);
  vFacing = frontFactor;

  // Horizontal holographic refresh scanline wave sweeping smoothly
  float scanWave1 = 0.5 + 0.5 * sin(pos.y * 55.0 - uTime * 3.2);
  float scanBand = smoothstep(0.92, 1.0, sin(pos.y * 3.4 - uTime * 1.7));
  vScanPulse = 0.90 + 0.10 * scanWave1 + 0.16 * scanBand;

  // Bottom organic dissolve (fades hoodie shoulders out cleanly ABOVE emitter lens!)
  float dissolveNoise = aRand * 0.13;
  vDissolve = 1.0 - smoothstep(0.77 - dissolveNoise, 0.945, aVNorm);

  // Density stochastic thinning if uDensity < 1.0
  if (aRand > uDensity) {
    vDissolve = 0.0;
  }

  // Luminous, high-detail 3D hologram lighting (inspired by cinema holo-projection reference!):
  // Directional 3D key light + specular ridge highlights carve out nose, brows, cheekbones, beard & hoodie folds,
  // while photo luminance + micro-edge detail keep dark hollows crisp and lit planes radiant electric cyan-white!
  vec3 keyDir = normalize(vec3(0.42, 0.52, 0.74));
  vec3 fillDir = normalize(vec3(-0.55, 0.15, 0.82));
  float keyDiff = max(dot(norm, keyDir), 0.0);
  float fillDiff = max(dot(norm, fillDir), 0.0);
  vec3 halfVec = normalize(keyDir + vec3(0.0, 0.0, 1.0));
  float spec = pow(max(dot(norm, halfVec), 0.0001), 14.0);

  // Sample live driving video face texture when active (projects real lips, teeth, eyes, blinks & cheeks onto 3D bust!)
  float effLum = aLum;
  float effEdge = aEdge;
  float darkCavity = 0.0;
  if (uHasDriving > 0.5) {
    float fu = (position.x - uDrivingBounds.x) / max(uDrivingBounds.y - uDrivingBounds.x, 0.001);
    // WebGL flipY=true maps image row 0 (top, yMax) to v=1.0, and bottom (yMin) to v=0.0
    float fv = (position.y - uDrivingBounds.z) / max(uDrivingBounds.w - uDrivingBounds.z, 0.001);
    if (fu > 0.01 && fu < 0.99 && fv > 0.01 && fv < 0.99 && aNormal.z > 0.15) {
      vec3 drv = texture2D(uDrivingTex, vec2(fu, fv)).rgb; // r=enhanced lum, g=sobel edge, b=raw gray
      // Smooth front-face projection feather (1.0 on eyes/nose/mouth/beard/cheeks, fades cleanly to 0 at ears/crown/hoodie)
      vec2 fRel = vec2(position.x / 0.44, (position.y - 0.04) / 0.56);
      float ellip = 1.0 - smoothstep(0.68, 1.05, length(fRel));
      float frontW = smoothstep(0.18, 0.52, aNormal.z) * aHeadW;
      float faceMask = clamp(ellip * frontW, 0.0, 1.0);

      // Protect pitch-dark open mouth interior & jet-black pupils while keeping sclera, teeth & lips luminous!
      float openDark = 1.0 - smoothstep(0.04, 0.16, drv.b);
      darkCavity = openDark * faceMask;

      effLum = mix(aLum, clamp(drv.r * 1.12, 0.0, 1.0), faceMask);
      effEdge = mix(aEdge, drv.g * 1.25, faceMask);
    }
  }

  // S-curve photo luminance: deep dark shadows stay dark cobalt, lit skin/beard/brows/lips shine bright!
  float lumC = pow(max(smoothstep(0.10, 0.88, clamp(effLum, 0.0, 1.0)), 0.0001), 1.25);
  float detailBoost = pow(max(clamp(effEdge, 0.0, 1.0), 0.0001), 0.85) * 1.15;
  // Hoodie (aHeadW near 0) is darker fabric — lift folds, drawstrings & shoulder rim cleanly!
  float hoodieEdgeLift = (1.0 - aHeadW) * (0.16 + effEdge * 1.05 + keyDiff * 0.45 + fresnel * 0.90);

  // Combine directional 3D studio key-lighting with photo luminance & fine edge detail!
  float litFace = (0.28 + keyDiff * 1.05 + fillDiff * 0.32 + spec * 0.85) * (0.26 + lumC * 1.22);
  // Suppress procedural lip/brow glow overrides in driving-video mode so real video contrast rules cleanly!
  float procMask = 1.0 - uHasDriving;
  float lipRimBoost = abs(aLipW) * (0.18 + uJawOpen * 0.38) * procMask;
  float eyeBrowBoost = (aBrowW * 0.34 + aBlinkW * (0.14 + uBlink * 0.25)) * procMask;

  float baseBright = (0.04 + litFace + detailBoost * (0.55 + keyDiff * 0.65) + hoodieEdgeLift + lipRimBoost + eyeBrowBoost) * (1.0 + fresnel * 0.92);
  // Deepen real open-mouth & pupil shadows dramatically so speech & gaze read razor-sharp!
  baseBright *= (1.0 - darkCavity * 0.86);
  baseBright *= (1.0 + uAudioEnergy * 0.08);

  vBrightness = baseBright * uIntensity;
  vFresnel = fresnel;
  vEdge = effEdge;
  vHeadW = aHeadW;

  // Occasional micro glitch color flicker
  float gRow = floor(pos.y * 28.0 + floor(uTime * 16.0));
  vGlitchHit = step(1.0 - uGlitch * 0.14, hash11(gRow + aRand * 0.1)) * uGlitch;

  // Crisp, radiant micro-dots (tight/tiny inside dark open mouth & shadows, luminous on lit lips/teeth/face!)
  float sizeBase = mix(0.85, 3.05, clamp(lumC * 0.55 + keyDiff * 0.35 + effEdge * 0.48 + fresnel * 0.40, 0.0, 1.0));
  sizeBase *= (1.0 - darkCavity * 0.55);
  float resScale = (uResolutionY / 1080.0) * uPixelRatio;
  float distScale = 2.65 / max(-mvPos.z, 0.5);
  gl_PointSize = clamp(sizeBase * uPointScale * resScale * distScale * vDissolve, 0.0, 15.0);
}
`;

export const POINT_CLOUD_FRAG = /* glsl */ `
uniform vec3 uColorPrimary;
uniform vec3 uColorHot;
uniform vec3 uColorDeep;

varying float vBrightness;
varying float vFresnel;
varying float vDissolve;
varying float vScanPulse;
varying float vEdge;
varying float vFacing;
varying float vHeadW;
varying float vGlitchHit;

void main() {
  if (vDissolve < 0.01 || vFacing < 0.01) discard;

  vec2 uv = gl_PointCoord - 0.5;
  float r2 = dot(uv, uv);
  if (r2 > 0.25) discard;

  // Crisp glowing pinpoint dot (distinct dots, never mushy white fog!)
  float core = exp(-r2 * 22.0);
  float halo = exp(-r2 * 7.5) * 0.32;
  float shape = core + halo;

  float lum = vBrightness * vScanPulse * vFacing;

  // Vivid electric cyan palette: deep sapphire shadows -> rich electric cyan midtones -> luminous ice-white highlights!
  vec3 col = mix(uColorDeep, uColorPrimary, clamp(lum * 0.85, 0.0, 1.0));
  float hotMix = smoothstep(1.05, 2.25, lum + vFresnel * 0.50 + vEdge * 0.35);
  col = mix(col, uColorHot, hotMix * 0.82);

  // Subtle chromatic glitch shift on glitch-triggered scan bands
  if (vGlitchHit > 0.05) {
    col = mix(col, vec3(0.45, 1.0, 0.90), vGlitchHit * 0.45);
  }

  float alpha = shape * vDissolve * clamp(lum * 0.84, 0.04, 1.0);
  gl_FragColor = vec4(col * shape * lum * 0.92 * vDissolve, alpha);
}
`;

export const INNER_SHELL_VERT = /* glsl */ `
attribute float aLum;
attribute float aEdge;
attribute float aHeadW;
attribute float aJawW;
attribute float aLipW;
attribute float aBlinkW;
attribute float aBrowW;
attribute float aVNorm;

varying vec3 vNormalView;
varying vec3 vViewDir;
varying float vLum;
varying float vVNorm;
varying vec3 vWorldPos;

${DEFORM_GLSL}

void main() {
  vec3 pos;
  vec3 norm;
  deformBust(position, normal, aHeadW, aJawW, aLipW, aBlinkW, aBrowW, pos, norm);

  // Pull inner shell slightly inward (~0.008) so front point cloud never z-fights!
  pos -= norm * 0.008;

  vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPos;

  vNormalView = normalize(normalMatrix * norm);
  vViewDir = normalize(-mvPos.xyz);
  vLum = aLum;
  vVNorm = aVNorm;
  vWorldPos = pos;
}
`;

export const INNER_SHELL_FRAG = /* glsl */ `
uniform vec3 uColorPrimary;
uniform vec3 uColorDeep;
uniform vec3 uMouthCenter;
uniform float uTime;
uniform float uIntensity;
uniform float uAudioEnergy;

varying vec3 vNormalView;
varying vec3 vViewDir;
varying float vLum;
varying float vVNorm;
varying vec3 vWorldPos;

void main() {
  float dissolve = 1.0 - smoothstep(0.79, 0.945, vVNorm);
  if (dissolve < 0.02) discard;

  float ndotv = clamp(dot(normalize(vNormalView + vec3(0.0, 0.0, 1e-5)), normalize(vViewDir + vec3(0.0, 0.0, 1e-5))), 0.0, 1.0);
  float fresnel = pow(max(1.0 - ndotv, 0.0001), 2.5);
  // Suppress bright Fresnel fold highlight on inner cheek creases right around the moving mouth!
  float mouthDist = length((vWorldPos.xy - uMouthCenter.xy) * vec2(0.75, 1.0));
  float mouthQuiet = smoothstep(0.36, 0.14, mouthDist);
  fresnel *= (1.0 - mouthQuiet * 0.92);

  // Fine holographic horizontal scanlines
  float scan = 0.78 + 0.22 * sin(vWorldPos.y * 145.0 - uTime * 4.0);
  // Deep dark volumetric core + crisp electric cyan Fresnel rim (properly occludes back of bust & emitter base!)
  vec3 col = mix(uColorDeep * 0.06, uColorPrimary * 1.25, fresnel) * scan;
  col *= (0.30 + 0.70 * vLum) * uIntensity * (1.0 + uAudioEnergy * 0.08);

  // Strong core opacity prevents back-of-bust / back-of-emitter from showing through the face & neck
  float alpha = clamp((0.76 + fresnel * 0.22) * dissolve, 0.0, 0.96);
  gl_FragColor = vec4(col, alpha);
}
`;

export const DEPTH_OCCLUDER_FRAG = /* glsl */ `
varying float vVNorm;
void main() {
  if (vVNorm > 0.94) discard;
  gl_FragColor = vec4(0.0);
}
`;

export const CONSTELLATION_VERT = /* glsl */ `
attribute vec3 aNormal;
attribute float aHeadW;
attribute float aJawW;
attribute float aLipW;
attribute float aBlinkW;
attribute float aBrowW;
attribute float aVNorm;
attribute float aLineStrength;

varying float vAlpha;
varying float vFresnel;

${DEFORM_GLSL}

void main() {
  vec3 pos;
  vec3 norm;
  deformBust(position, aNormal, aHeadW, aJawW, aLipW, aBlinkW, aBrowW, pos, norm);

  pos += norm * 0.005;

  vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPos;

  vec3 viewNormal = normalize(normalMatrix * norm + vec3(0.0, 0.0, 1e-5));
  vec3 viewDir = normalize(-mvPos.xyz + vec3(0.0, 0.0, 1e-5));
  float ndotv = dot(viewNormal, viewDir);
  float fresnel = pow(max(1.0 - clamp(abs(ndotv), 0.0, 1.0), 0.0001), 1.8);

  float dissolve = 1.0 - smoothstep(0.79, 0.95, aVNorm);
  // Strict front-face culling on constellation wireframe so back-of-shoulders never bleeds through!
  float facing = smoothstep(-0.04, 0.16, ndotv);

  // Prominent geometric constellation along hoodie shoulders/collar & outer rim
  float shoulderZone = 1.0 - aHeadW;
  float vis = max(shoulderZone * 1.15, fresnel * 0.65) * aLineStrength;
  vAlpha = vis * dissolve * facing;
  vFresnel = fresnel;
}
`;

export const CONSTELLATION_FRAG = /* glsl */ `
uniform vec3 uColorPrimary;
uniform vec3 uColorHot;
uniform float uIntensity;

varying float vAlpha;
varying float vFresnel;

void main() {
  if (vAlpha < 0.01) discard;
  vec3 col = mix(uColorPrimary, uColorHot, vFresnel * 0.55) * uIntensity * 1.15;
  gl_FragColor = vec4(col * vAlpha, vAlpha * 0.85);
}
`;

export const EMITTER_BEAM_VERT = /* glsl */ `
varying vec2 vUv;
varying vec3 vNormalView;
varying vec3 vViewDir;

void main() {
  vUv = uv;
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPos;
  vNormalView = normalize(normalMatrix * normal);
  vViewDir = normalize(-mvPos.xyz);
}
`;

export const EMITTER_BEAM_FRAG = /* glsl */ `
uniform vec3 uColorPrimary;
uniform vec3 uColorHot;
uniform float uTime;
uniform float uAudioEnergy;
uniform float uIntensity;

varying vec2 vUv;
varying vec3 vNormalView;
varying vec3 vViewDir;

void main() {
  // vUv.y = 0 at emitter lens, 1 at top of cone
  float yFade = pow(max(1.0 - vUv.y, 0.0001), 1.55) * smoothstep(0.0, 0.05, vUv.y);
  // Crisp radial volumetric light shafts shooting upward from emitter optics!
  float angle = vUv.x * 6.2831853;
  float rays = 0.48
    + 0.32 * sin(angle * 16.0 + uTime * 0.45)
    + 0.22 * sin(angle * 31.0 - uTime * 0.85)
    + 0.16 * sin(angle * 53.0 + uTime * 1.2);
  rays = pow(max(clamp(rays, 0.12, 1.35), 0.0001), 1.25);

  float edgeSoft = pow(max(abs(dot(normalize(vNormalView + vec3(0.0, 0.0, 1e-5)), normalize(vViewDir + vec3(0.0, 0.0, 1e-5)))), 0.0001), 0.72);
  float pulse = 0.92 + 0.24 * uAudioEnergy;

  float strength = yFade * rays * edgeSoft * pulse * uIntensity * 1.25;
  vec3 col = mix(uColorPrimary, uColorHot, pow(max(1.0 - vUv.y, 0.0001), 2.4) * 0.78);
  gl_FragColor = vec4(col * strength, clamp(strength, 0.0, 1.0));
}
`;

export const FULLSCREEN_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const BLOOM_EXTRACT_FRAG = /* glsl */ `
uniform sampler2D tDiffuse;
uniform float uThreshold;
varying vec2 vUv;

vec3 sanitize(vec3 c) {
  return clamp(vec3(
    isnan(c.r) || isinf(c.r) ? 0.0 : c.r,
    isnan(c.g) || isinf(c.g) ? 0.0 : c.g,
    isnan(c.b) || isinf(c.b) ? 0.0 : c.b
  ), 0.0, 16.0);
}

void main() {
  vec3 rgb = sanitize(texture2D(tDiffuse, vUv).rgb);
  float lum = max(rgb.r, max(rgb.g, rgb.b));
  float soft = smoothstep(uThreshold, uThreshold + 0.45, lum);
  gl_FragColor = vec4(rgb * soft, 1.0);
}
`;

export const BLUR_FRAG = /* glsl */ `
uniform sampler2D tDiffuse;
uniform vec2 uDirection;
uniform vec2 uResolution;
varying vec2 vUv;

void main() {
  vec2 px = uDirection / uResolution;
  vec3 sum = texture2D(tDiffuse, vUv).rgb * 0.227027;
  sum += texture2D(tDiffuse, vUv + px * 1.384615).rgb * 0.316216;
  sum += texture2D(tDiffuse, vUv - px * 1.384615).rgb * 0.316216;
  sum += texture2D(tDiffuse, vUv + px * 3.230769).rgb * 0.070270;
  sum += texture2D(tDiffuse, vUv - px * 3.230769).rgb * 0.070270;
  gl_FragColor = vec4(sum, 1.0);
}
`;

export const COMPOSITE_FRAG = /* glsl */ `
uniform sampler2D tScene;
uniform sampler2D tBloomTight;
uniform sampler2D tBloomWide;
uniform float uBloomStrength;
uniform float uGlitch;
uniform float uTime;
uniform int uBgMode; // 0 = black, 1 = transparent, 2 = dark lab backdrop
uniform vec3 uColorPrimary;
uniform vec2 uResolution;

varying vec2 vUv;

float hash(float n) { return fract(sin(n) * 43758.5453123); }

// Smooth ACES-inspired highlight rolloff (keeps intense cyan peaks turning luminous ice-white!)
vec3 tonemapHolo(vec3 x) {
  return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
}

void main() {
  vec2 uv = vUv;

  // Subtle chromatic RGB shift during glitch bursts
  float ca = (0.0006 + uGlitch * 0.0045);
  float rowNoise = hash(floor(uv.y * 90.0) + floor(uTime * 24.0));
  if (uGlitch > 0.05 && rowNoise > 1.0 - uGlitch * 0.10) {
    uv.x += (hash(rowNoise * 13.7) - 0.5) * 0.018 * uGlitch;
  }

  vec4 sCenter = texture2D(tScene, uv);
  float rCh = texture2D(tScene, uv + vec2(ca, 0.0)).r;
  float bCh = texture2D(tScene, uv - vec2(ca, 0.0)).b;
  vec3 sceneRgb = vec3(rCh, sCenter.g, bCh);
  float sceneA = sCenter.a;

  vec3 bloomT = texture2D(tBloomTight, uv).rgb;
  vec3 bloomW = texture2D(tBloomWide, uv).rgb;

  // Combine crisp scene + tight halo + wide atmospheric volumetric bloom
  vec3 hdr = sceneRgb + (bloomT * 0.88 + bloomW * 0.72) * uBloomStrength;

  // Subtle high-frequency CRT/holo projection scanline texture on illuminated pixels
  float scanline = 0.96 + 0.04 * sin(uv.y * uResolution.y * 1.15);
  hdr *= scanline;

  if (uBgMode == 2) {
    // Moody dark volumetric sci-fi lab backdrop with subtle bokeh/vignette
    vec2 centered = (uv - 0.5) * vec2(uResolution.x / uResolution.y, 1.0);
    float r = length(centered);
    vec3 bg = mix(vec3(0.018, 0.045, 0.075), vec3(0.004, 0.009, 0.018), smoothstep(0.0, 0.95, r));
    // Soft ambient cyan haze behind hologram center
    bg += uColorPrimary * exp(-r * r * 3.8) * 0.075;
    vec3 finalRgb = tonemapHolo(bg + hdr);
    gl_FragColor = vec4(finalRgb, 1.0);
  } else if (uBgMode == 1) {
    // TRANSPARENT ALPHA OUTPUT (ProRes 4444 / WebM VP9)
    // Exact optical straight-alpha decomposition: straightRgb * outAlpha == mapped
    // Guarantees zero dark fringes over any video footage AND exact black-bg match!
    vec3 mapped = tonemapHolo(hdr);
    float glowLum = max(mapped.r, max(mapped.g, mapped.b));
    float outAlpha = clamp(max(sceneA, glowLum), 0.0, 1.0);
    vec3 straightRgb = outAlpha > 0.002 ? clamp(mapped / outAlpha, 0.0, 1.0) : vec3(0.0);
    gl_FragColor = vec4(straightRgb, outAlpha);
  } else {
    // PURE BLACK BACKGROUND (H.264 MP4 for Screen / Add blend mode!)
    vec3 mapped = tonemapHolo(hdr);
    gl_FragColor = vec4(mapped, 1.0);
  }
}
`;
