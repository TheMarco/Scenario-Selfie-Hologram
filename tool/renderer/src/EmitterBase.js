import * as THREE from 'three';
import { EMITTER_BEAM_VERT, EMITTER_BEAM_FRAG } from './shaders.js';

/**
 * Toggleable Sci-Fi Holographic Emitter Pad + Volumetric Projection Light Rays.
 */
export class EmitterBase {
  constructor(palette, options = {}) {
    this.group = new THREE.Group();
    this.group.position.set(0, -1.26, 0.02);

    // 1. Sleek machined metallic circular housing (LatheGeometry)
    const profile = [
      new THREE.Vector2(0.0, 0.0),
      new THREE.Vector2(0.76, 0.0),
      new THREE.Vector2(0.79, 0.045),
      new THREE.Vector2(0.73, 0.075),
      new THREE.Vector2(0.68, 0.13),
      new THREE.Vector2(0.58, 0.165),
      new THREE.Vector2(0.50, 0.165),
      new THREE.Vector2(0.45, 0.125),
      new THREE.Vector2(0.0, 0.125),
    ];
    const baseGeo = new THREE.LatheGeometry(profile, 64);
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0x1e2833,
      metalness: 0.85,
      roughness: 0.28,
      emissive: palette.deep.clone().multiplyScalar(0.18),
    });
    const baseMesh = new THREE.Mesh(baseGeo, baseMat);
    baseMesh.renderOrder = 1;
    this.group.add(baseMesh);

    // Radial machined perimeter fins around the emitter pad
    const finGeo = new THREE.BoxGeometry(0.11, 0.07, 0.14);
    const finMat = new THREE.MeshStandardMaterial({
      color: 0x2b3948,
      metalness: 0.88,
      roughness: 0.24,
    });
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      const fin = new THREE.Mesh(finGeo, finMat);
      fin.position.set(Math.cos(a) * 0.67, 0.075, Math.sin(a) * 0.67);
      fin.rotation.y = -a;
      this.group.add(fin);
    }

    // 2. Glowing concentric cyan optics rings + lens core
    const lensGeo = new THREE.CircleGeometry(0.46, 64);
    lensGeo.rotateX(-Math.PI / 2);
    this.lensMat = new THREE.ShaderMaterial({
      uniforms: {
        uColorPrimary: { value: palette.primary.clone() },
        uColorHot: { value: palette.hot.clone() },
        uTime: { value: 0 },
        uAudioEnergy: { value: 0 },
        uIntensity: { value: options.intensity ?? 1.0 },
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
        uniform float uTime;
        uniform float uAudioEnergy;
        uniform float uIntensity;
        varying vec2 vUv;
        void main() {
          vec2 p = (vUv - 0.5) * 2.0;
          float r = length(p);
          float core = exp(-r * r * 2.8);
          float ring1 = smoothstep(0.06, 0.0, abs(r - 0.72));
          float ring2 = smoothstep(0.04, 0.0, abs(r - 0.90));
          float pulse = 0.92 + 0.18 * uAudioEnergy + 0.06 * sin(uTime * 3.0);
          float val = (core * 1.1 + ring1 * 1.2 + ring2 * 0.9) * pulse * uIntensity;
          vec3 col = mix(uColorPrimary, uColorHot, clamp(core + ring1 * 0.6, 0.0, 1.0));
          gl_FragColor = vec4(col * val, clamp(val, 0.0, 1.0));
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const lens = new THREE.Mesh(lensGeo, this.lensMat);
    lens.position.y = 0.132;
    lens.renderOrder = 2;
    this.group.add(lens);

    // Subtle rim lights illuminating the metallic emitter pad
    const cyanPoint = new THREE.PointLight(palette.primary.getHex(), 4.5, 4.0);
    cyanPoint.position.set(0, 0.45, 0.2);
    this.group.add(cyanPoint);

    const rimLight = new THREE.DirectionalLight(0x88ccff, 1.4);
    rimLight.position.set(1.5, 2.0, 2.2);
    this.group.add(rimLight);

    const backFill = new THREE.DirectionalLight(0x224466, 0.9);
    backFill.position.set(-1.5, 1.0, -1.5);
    this.group.add(backFill);

    // 3. Upward Volumetric Projection Light Cones (layered inner + outer beams!)
    this.beamUniforms = {
      uColorPrimary: { value: palette.primary.clone() },
      uColorHot: { value: palette.hot.clone() },
      uTime: { value: 0 },
      uAudioEnergy: { value: 0 },
      uIntensity: { value: options.intensity ?? 1.0 },
    };

    const outerConeGeo = new THREE.CylinderGeometry(1.12, 0.40, 0.95, 64, 1, true);
    outerConeGeo.translate(0, 0.475, 0);
    const beamMat = new THREE.ShaderMaterial({
      vertexShader: EMITTER_BEAM_VERT,
      fragmentShader: EMITTER_BEAM_FRAG,
      uniforms: this.beamUniforms,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    const outerCone = new THREE.Mesh(outerConeGeo, beamMat);
    outerCone.position.y = 0.13;
    outerCone.renderOrder = 3;
    this.group.add(outerCone);

    const innerConeGeo = new THREE.CylinderGeometry(0.68, 0.24, 0.72, 48, 1, true);
    innerConeGeo.translate(0, 0.36, 0);
    const innerCone = new THREE.Mesh(innerConeGeo, beamMat);
    innerCone.position.y = 0.13;
    innerCone.renderOrder = 3;
    this.group.add(innerCone);
  }

  update(frame) {
    this.lensMat.uniforms.uTime.value = frame.t;
    this.lensMat.uniforms.uAudioEnergy.value = frame.energy;
    this.beamUniforms.uTime.value = frame.t;
    this.beamUniforms.uAudioEnergy.value = frame.energy;
  }
}
