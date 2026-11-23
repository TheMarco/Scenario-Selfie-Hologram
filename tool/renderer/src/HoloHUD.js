import * as THREE from 'three';

/**
 * 3D Floating Sci-Fi Holographic HUD Overlay.
 * Features left & right floating coordinate grids, rotating concentric circular reticles,
 * live audio-reactive numeric telemetry, 8-band spectrum ticks, and thin RGB glitch bars.
 */
export class HoloHUD {
  constructor(palette, options = {}) {
    this.group = new THREE.Group();
    this.palette = palette;
    this.isPortrait = !!options.isPortrait;
    this.primaryHex = '#' + palette.primary.getHexString();
    this.hotHex = '#' + palette.hot.getHexString();

    if (this.isPortrait) {
      // Portrait 9:16 social layout: Top sci-fi telemetry banner + Bottom live voice equalizer/waveform!
      this.topCanvas = document.createElement('canvas');
      this.topCanvas.width = 960;
      this.topCanvas.height = 360;
      this.topCtx = this.topCanvas.getContext('2d');
      this.topTex = new THREE.CanvasTexture(this.topCanvas);
      this.topTex.minFilter = THREE.LinearFilter;
      this.topTex.magFilter = THREE.LinearFilter;

      const topMat = new THREE.MeshBasicMaterial({
        map: this.topTex,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      this.topPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.75, 0.66), topMat);
      this.topPlane.position.set(0.0, 1.38, 0.08);
      this.topPlane.renderOrder = 6;
      this.group.add(this.topPlane);

      this.botCanvas = document.createElement('canvas');
      this.botCanvas.width = 960;
      this.botCanvas.height = 320;
      this.botCtx = this.botCanvas.getContext('2d');
      this.botTex = new THREE.CanvasTexture(this.botCanvas);
      this.botTex.minFilter = THREE.LinearFilter;
      this.botTex.magFilter = THREE.LinearFilter;

      const botMat = new THREE.MeshBasicMaterial({
        map: this.botTex,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      this.botPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.75, 0.58), botMat);
      this.botPlane.position.set(0.0, -1.74, 0.12);
      this.botPlane.renderOrder = 6;
      this.group.add(this.botPlane);
      return;
    }

    // Landscape 16:9 layout: Left & Right floating 3D planes
    this.rightCanvas = document.createElement('canvas');
    this.rightCanvas.width = 640;
    this.rightCanvas.height = 840;
    this.rightCtx = this.rightCanvas.getContext('2d');
    this.rightTex = new THREE.CanvasTexture(this.rightCanvas);
    this.rightTex.minFilter = THREE.LinearFilter;
    this.rightTex.magFilter = THREE.LinearFilter;

    const rightMat = new THREE.MeshBasicMaterial({
      map: this.rightTex,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    this.rightPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.18, 1.55), rightMat);
    this.rightPlane.position.set(1.02, 0.08, 0.10);
    this.rightPlane.rotation.y = -0.22;
    this.rightPlane.renderOrder = 6;
    this.group.add(this.rightPlane);

    // Left HUD canvas plane (numeric header + scale ruler + spectrum bars + circular node + glitch lines)
    this.leftCanvas = document.createElement('canvas');
    this.leftCanvas.width = 640;
    this.leftCanvas.height = 840;
    this.leftCtx = this.leftCanvas.getContext('2d');
    this.leftTex = new THREE.CanvasTexture(this.leftCanvas);
    this.leftTex.minFilter = THREE.LinearFilter;
    this.leftTex.magFilter = THREE.LinearFilter;

    const leftMat = new THREE.MeshBasicMaterial({
      map: this.leftTex,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    this.leftPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.18, 1.55), leftMat);
    this.leftPlane.position.set(-1.02, 0.06, 0.14);
    this.leftPlane.rotation.y = 0.22;
    this.leftPlane.renderOrder = 6;
    this.group.add(this.leftPlane);
  }

  _rgba(hex, alpha) {
    const c = new THREE.Color(hex);
    return `rgba(${Math.round(c.r * 255)}, ${Math.round(c.g * 255)}, ${Math.round(c.b * 255)}, ${alpha})`;
  }

  _drawGrid(ctx, x, y, w, h, step, alpha) {
    ctx.save();
    ctx.strokeStyle = this._rgba(this.primaryHex, alpha);
    ctx.lineWidth = 1;
    // Soft radial fade using clip + gradient mask effect via per-line alpha
    for (let gx = 0; gx <= w; gx += step) {
      const fx = 1.0 - Math.abs((gx / w) - 0.5) * 1.4;
      ctx.strokeStyle = this._rgba(this.primaryHex, Math.max(0.03, alpha * fx));
      ctx.beginPath();
      ctx.moveTo(x + gx, y);
      ctx.lineTo(x + gx, y + h);
      ctx.stroke();
    }
    for (let gy = 0; gy <= h; gy += step) {
      const fy = 1.0 - Math.abs((gy / h) - 0.5) * 1.4;
      ctx.strokeStyle = this._rgba(this.primaryHex, Math.max(0.03, alpha * fy));
      ctx.beginPath();
      ctx.moveTo(x, y + gy);
      ctx.lineTo(x + w, y + gy);
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawReticle(ctx, cx, cy, r, t, energy) {
    ctx.save();
    ctx.translate(cx, cy);

    // Outer dashed thin ring
    ctx.rotate(t * 0.28);
    ctx.strokeStyle = this._rgba(this.primaryHex, 0.45);
    ctx.lineWidth = 1.5;
    ctx.setLineDash([12, 8, 4, 8]);
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Active segmented heavy arc reacting to audio
    ctx.rotate(-t * 0.65);
    ctx.strokeStyle = this._rgba(this.hotHex, 0.72 + energy * 0.25);
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.78, 0.2, Math.PI * (0.9 + energy * 0.45));
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.78, Math.PI * 1.25, Math.PI * 1.72);
    ctx.stroke();

    // Inner precision ring with compass ticks
    ctx.strokeStyle = this._rgba(this.primaryHex, 0.52);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.56, 0, Math.PI * 2);
    ctx.stroke();

    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const len = i % 3 === 0 ? 9 : 4;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * (r * 0.56), Math.sin(a) * (r * 0.56));
      ctx.lineTo(Math.cos(a) * (r * 0.56 + len), Math.sin(a) * (r * 0.56 + len));
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawGlitchBar(ctx, x, y, w, h, glitchBoost) {
    // Crisp cyan + tiny magenta/emerald RGB chromatic sliver like the reference image!
    ctx.save();
    ctx.fillStyle = this._rgba(this.hotHex, 0.62 + glitchBoost * 0.3);
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = `rgba(0, 255, 165, ${0.45 + glitchBoost * 0.35})`;
    ctx.fillRect(x + w * 0.35, y - 1, w * 0.28, h);
    ctx.fillStyle = `rgba(255, 48, 108, ${0.38 + glitchBoost * 0.40})`;
    ctx.fillRect(x - 10, y + 1, w * 0.22, h);
    ctx.restore();
  }

  update(frame) {
    const t = frame.t;
    const energy = frame.energy;
    const glitch = frame.glitch;
    const spec = frame.spectrum || [0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1];

    if (this.isPortrait) {
      this._updatePortrait(t, energy, glitch, spec);
      return;
    }

    // Subtle organic floating parallax
    this.rightPlane.position.y = 0.08 + Math.sin(t * 0.9) * 0.015;
    this.leftPlane.position.y = 0.06 + Math.cos(t * 0.85) * 0.015;

    // --- Render RIGHT HUD ---
    const rc = this.rightCtx;
    rc.clearRect(0, 0, 640, 840);

    // Upper coordinate grid
    this._drawGrid(rc, 285, 65, 280, 220, 28, 0.44);

    // Upper rotating circular reticle
    this._drawReticle(rc, 168, 175, 86, t, energy);

    // Tiny micro telemetry label above reticle
    rc.font = '600 13px "SF Mono", "Menlo", monospace';
    rc.fillStyle = this._rgba(this.hotHex, 0.62);
    rc.fillText('SYN//LOCK', 70, 74);
    rc.fillStyle = this._rgba(this.primaryHex, 0.48);
    rc.fillText(`HZ ${(124.4 + energy * 68.2).toFixed(2)}`, 70, 92);

    // Mid-right numeric readout + ring node (echoing "83053003" from reference)
    this._drawReticle(rc, 276, 390, 42, -t * 1.1, energy * 0.8);
    rc.strokeStyle = this._rgba(this.primaryHex, 0.55);
    rc.lineWidth = 1.2;
    rc.beginPath();
    rc.moveTo(318, 390);
    rc.lineTo(356, 390);
    rc.stroke();

    const codeNum = Math.floor(83053000 + Math.floor(t * 12) + energy * 420);
    rc.font = '500 17px "SF Mono", "Menlo", monospace';
    rc.fillStyle = this._rgba(this.hotHex, 0.78);
    rc.fillText(String(codeNum), 366, 396);

    // Crisp horizontal accent / glitch bars
    this._drawGlitchBar(rc, 210 + Math.sin(t * 5) * 6, 326, 220, 3, glitch);
    this._drawGlitchBar(rc, 320, 444, 150, 2.5, glitch * 0.8);

    // Lower coordinate grid + subtle live audio waveform inside it
    this._drawGrid(rc, 270, 505, 255, 195, 25, 0.38);
    rc.save();
    rc.strokeStyle = this._rgba(this.hotHex, 0.70 + energy * 0.25);
    rc.lineWidth = 1.8;
    rc.beginPath();
    for (let i = 0; i < 48; i++) {
      const px = 280 + (i / 47) * 235;
      const bIdx = Math.min(7, Math.floor((i / 48) * 8));
      const amp = spec[bIdx] * (16 + energy * 24);
      const py = 602 + Math.sin(i * 0.75 + t * 14.0) * amp;
      if (i === 0) rc.moveTo(px, py);
      else rc.lineTo(px, py);
    }
    rc.stroke();
    rc.restore();

    this.rightTex.needsUpdate = true;

    // --- Render LEFT HUD ---
    const lc = this.leftCtx;
    lc.clearRect(0, 0, 640, 840);

    // Top-left floating coordinate & readout (echoing "128.57621" from reference)
    const metricVal = (128.57621 + Math.sin(t * 0.9) * 0.14 + energy * 1.42).toFixed(5);
    lc.font = '500 19px "SF Mono", "Menlo", monospace';
    lc.fillStyle = this._rgba(this.hotHex, 0.78);
    lc.fillText(metricVal, 175, 174);

    // Underline + diagonal leader
    lc.strokeStyle = this._rgba(this.primaryHex, 0.50);
    lc.lineWidth = 1.2;
    lc.beginPath();
    lc.moveTo(165, 184);
    lc.lineTo(395, 184);
    lc.lineTo(430, 208);
    lc.stroke();

    // Small decorative glitch slivers on left
    this._drawGlitchBar(lc, 120, 96, 130, 2.5, glitch);
    this._drawGlitchBar(lc, 240, 286, 195, 3, glitch);
    this._drawGlitchBar(lc, 105, 366, 165, 2, glitch * 0.7);

    // Mid-left floating target reticle near cheek/shoulder
    this._drawReticle(lc, 248, 520, 48, t * 0.85, energy);

    // Vertical scale & live 8-band voice equalizer (echoing "G8" vertical scale in reference!)
    lc.font = '600 17px "SF Mono", "Menlo", monospace';
    lc.fillStyle = this._rgba(this.hotHex, 0.74);
    lc.fillText('G8', 112, 558);

    for (let b = 0; b < 8; b++) {
      const by = 582 + b * 22;
      const bw = 14 + spec[b] * 78;
      lc.fillStyle = this._rgba(this.primaryHex, 0.35 + spec[b] * 0.55);
      lc.fillRect(126, by, bw, 5);
      lc.fillStyle = this._rgba(this.hotHex, 0.85);
      lc.fillRect(126 + bw + 4, by, 4, 5);
    }

    // Fine vertical graduation ruler ticks
    lc.strokeStyle = this._rgba(this.primaryHex, 0.42);
    lc.lineWidth = 1;
    lc.beginPath();
    lc.moveTo(116, 576);
    lc.lineTo(116, 760);
    lc.stroke();
    for (let k = 0; k <= 12; k++) {
      const ty = 576 + k * 15;
      lc.beginPath();
      lc.moveTo(110, ty);
      lc.lineTo(116, ty);
      lc.stroke();
    }

    this.leftTex.needsUpdate = true;
  }

  _updatePortrait(t, energy, glitch, spec) {
    this.topPlane.position.y = 1.38 + Math.sin(t * 0.85) * 0.012;
    this.botPlane.position.y = -1.74 + Math.cos(t * 0.85) * 0.010;

    // --- TOP PORTRAIT HUD (left telemetry readout + center subtle grid + right rotating reticle) ---
    const tc = this.topCtx;
    tc.clearRect(0, 0, 960, 360);

    // Subtle coordinate grid on top-right
    this._drawGrid(tc, 640, 48, 250, 180, 26, 0.36);

    // Top-right rotating scanner reticle
    this._drawReticle(tc, 765, 138, 66, t, energy);

    // Top-left live telemetry header & glitch accent bars
    const metricVal = (128.57621 + Math.sin(t * 0.9) * 0.14 + energy * 1.42).toFixed(5);
    tc.font = '600 18px "SF Mono", "Menlo", monospace';
    tc.fillStyle = this._rgba(this.hotHex, 0.78);
    tc.fillText('SYN//HOLO-LINK', 74, 92);
    tc.font = '500 22px "SF Mono", "Menlo", monospace';
    tc.fillStyle = this._rgba(this.primaryHex, 0.75);
    tc.fillText(metricVal, 74, 126);

    tc.strokeStyle = this._rgba(this.primaryHex, 0.48);
    tc.lineWidth = 1.4;
    tc.beginPath();
    tc.moveTo(74, 140);
    tc.lineTo(310, 140);
    tc.lineTo(340, 166);
    tc.stroke();

    this._drawGlitchBar(tc, 74, 162, 160, 3, glitch);
    this._drawGlitchBar(tc, 520 + Math.sin(t * 4.5) * 5, 258, 175, 2.5, glitch * 0.9);

    const codeNum = Math.floor(83053000 + Math.floor(t * 12) + energy * 420);
    tc.font = '500 16px "SF Mono", "Menlo", monospace';
    tc.fillStyle = this._rgba(this.hotHex, 0.68);
    tc.fillText(`HZ ${(124.4 + energy * 68.2).toFixed(2)}  //  ${codeNum}`, 74, 212);

    this.topTex.needsUpdate = true;

    // --- BOTTOM PORTRAIT HUD (live audio waveform + 8-band spectrum equalizer) ---
    const bc = this.botCtx;
    bc.clearRect(0, 0, 960, 320);

    // Centered subtle grid + live oscillating voice waveform
    this._drawGrid(bc, 110, 42, 430, 150, 26, 0.30);
    bc.save();
    bc.strokeStyle = this._rgba(this.hotHex, 0.78 + energy * 0.20);
    bc.lineWidth = 2.2;
    bc.beginPath();
    for (let i = 0; i < 64; i++) {
      const px = 122 + (i / 63) * 406;
      const bIdx = Math.min(7, Math.floor((i / 64) * 8));
      const amp = spec[bIdx] * (16 + energy * 34);
      const py = 117 + Math.sin(i * 0.65 + t * 14.0) * amp;
      if (i === 0) bc.moveTo(px, py);
      else bc.lineTo(px, py);
    }
    bc.stroke();
    bc.restore();

    // Right-side 8-band vertical voice spectrum bars
    bc.font = '600 16px "SF Mono", "Menlo", monospace';
    bc.fillStyle = this._rgba(this.hotHex, 0.72);
    bc.fillText('VOCAL // G8', 600, 62);
    for (let b = 0; b < 8; b++) {
      const bx = 602 + b * 32;
      const bh = 12 + spec[b] * 92;
      bc.fillStyle = this._rgba(this.primaryHex, 0.35 + spec[b] * 0.55);
      bc.fillRect(bx, 188 - bh, 18, bh);
      bc.fillStyle = this._rgba(this.hotHex, 0.88);
      bc.fillRect(bx, 188 - bh - 5, 18, 3);
    }

    this._drawGlitchBar(bc, 110, 214, 210, 2.5, glitch);
    this._drawGlitchBar(bc, 600, 214, 150, 2.5, glitch * 0.8);

    this.botTex.needsUpdate = true;
  }
}
