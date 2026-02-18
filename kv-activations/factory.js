// KV Activations in Transformers — Interactive Diagrams
// 4 sections: autoregressive, unrolled, KV cache, backprop

// ── Configurable token sequence ──
// Change this array to visualize any autoregressive chain.
// SEQUENCE[0] is the prompt; each subsequent token is what the model predicts.
const SEQUENCE = ['The', 'cat', 'sat', 'on', 'the', 'mat'];

const FULL_CHAIN = SEQUENCE;
const TOKENS = SEQUENCE.slice(0, -1);
const PREDS  = SEQUENCE.slice(1);
const N = TOKENS.length;
const L = 4;

const COL = {
  cyan: '#00d4ff', amber: '#ffb020', green: '#40ff90',
  magenta: '#ff3080', red: '#ff4060', white: '#e2e8f0',
  dim: '#4a5568', bg: '#000000',
};

// ── Utilities ──

function lerp(a, b, t) { return a + (b - a) * t; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function easeOut(t) { return 1 - (1 - t) ** 3; }
function easeInOut(t) { return t < 0.5 ? 4*t*t*t : 1 - (-2*t+2)**3/2; }

function rgba(hex, a) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

function pop(t, activation, riseMs, fallMs) {
  if (activation < 0) return 0;
  const e = t - activation;
  if (e < 0) return 0;
  const rise = riseMs || 0.1;
  const fall = fallMs || 0.45;
  if (e < rise) return e / rise;
  return Math.max(0, 1 - (e - rise) / fall);
}

// ── Drawing primitives ──

function drawBg(ctx, W, H) {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);
  const sp = 28;
  ctx.fillStyle = 'rgba(0,200,255,0.03)';
  for (let x = sp; x < W; x += sp)
    for (let y = sp; y < H; y += sp)
      ctx.fillRect(x - 0.5, y - 0.5, 1, 1);
}

function glowRect(ctx, x, y, w, h, color, intensity) {
  if (intensity <= 0) return;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.fillStyle = rgba(color, intensity * 0.05);
  ctx.fillRect(x - 12, y - 12, w + 24, h + 24);
  ctx.fillStyle = rgba(color, intensity * 0.12);
  ctx.fillRect(x - 6, y - 6, w + 12, h + 12);
  ctx.fillStyle = rgba(color, intensity * 0.2);
  ctx.fillRect(x - 2, y - 2, w + 4, h + 4);
  ctx.restore();
}

function glowLine(ctx, x1, y1, x2, y2, color, intensity, w) {
  if (intensity <= 0) return;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'butt';
  ctx.strokeStyle = rgba(color, intensity * 0.06);
  ctx.lineWidth = w + 6;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  ctx.strokeStyle = rgba(color, intensity * 0.14);
  ctx.lineWidth = w + 2;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  ctx.restore();
  ctx.strokeStyle = rgba(color, intensity);
  ctx.lineWidth = w;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}

// Directional glow: draws from (x1,y1) toward (x2,y2), clipped at `progress` (0..1)
function glowLineDir(ctx, x1, y1, x2, y2, color, intensity, w, progress) {
  if (intensity <= 0 || progress <= 0) return;
  const p = Math.min(progress, 1);
  glowLine(ctx, x1, y1, lerp(x1, x2, p), lerp(y1, y2, p), color, intensity, w);
}

function rRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r, y); ctx.lineTo(x+w-r, y);
  ctx.arcTo(x+w, y, x+w, y+r, r); ctx.lineTo(x+w, y+h-r);
  ctx.arcTo(x+w, y+h, x+w-r, y+h, r); ctx.lineTo(x+r, y+h);
  ctx.arcTo(x, y+h, x, y+h-r, r); ctx.lineTo(x, y+r);
  ctx.arcTo(x, y, x+r, y, r); ctx.closePath();
}

function drawLayerNode(ctx, cx, cy, sz, label, a, glow, color) {
  color = color || COL.cyan;
  const x = cx - sz/2, y = cy - sz/2;
  if (glow > 0) glowRect(ctx, x, y, sz, sz, color, a * (0.6 + glow * 0.6));
  ctx.fillStyle = rgba('#050a18', a * 0.95);
  rRect(ctx, x, y, sz, sz, 3); ctx.fill();
  ctx.strokeStyle = rgba(color, a * (0.5 + glow * 0.5));
  ctx.lineWidth = 1.5 + glow;
  rRect(ctx, x, y, sz, sz, 3); ctx.stroke();
  ctx.fillStyle = rgba(color, a * (0.4 + glow * 0.3));
  ctx.font = `500 ${Math.round(sz * 0.36)}px 'IBM Plex Mono', monospace`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, cx, cy + 1);
}

function drawActNode(ctx, cx, cy, sz, color, a, glow) {
  const x = cx - sz/2, y = cy - sz/2;
  if (glow > 0) glowRect(ctx, x, y, sz, sz, color, a * glow * 0.8);
  ctx.fillStyle = rgba('#050a18', a * 0.9);
  rRect(ctx, x, y, sz, sz, 2); ctx.fill();
  ctx.strokeStyle = rgba(color, a * (0.4 + glow * 0.5));
  ctx.lineWidth = 1 + glow * 0.5;
  rRect(ctx, x, y, sz, sz, 2); ctx.stroke();
}

function drawTokenPill(ctx, cx, cy, text, fontSize, a, glow, color) {
  color = color || COL.white;
  ctx.font = `500 ${fontSize}px 'IBM Plex Mono', monospace`;
  const tw = ctx.measureText(text).width + fontSize * 0.9;
  const th = fontSize * 1.6;
  const x = cx - tw/2, y = cy - th/2;
  if (glow > 0) glowRect(ctx, x, y, tw, th, color, a * glow * 0.5);
  ctx.fillStyle = rgba('#050a18', a * 0.8);
  rRect(ctx, x, y, tw, th, th/2); ctx.fill();
  ctx.strokeStyle = rgba(color, a * (0.3 + glow * 0.4));
  ctx.lineWidth = 1;
  rRect(ctx, x, y, tw, th, th/2); ctx.stroke();
  ctx.fillStyle = rgba(color, a * (0.7 + glow * 0.3));
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, cx, cy);
  return tw;
}

function drawPredBar(ctx, cx, cy, text, a, glow) {
  const bw = 4, bg = 1.5, nb = 5;
  const heights = [0.8, 0.35, 0.16, 0.08, 0.03];
  const maxH = 14;
  const tw = nb * bw + (nb-1) * bg;
  const sx = cx - tw/2;
  if (glow > 0) glowRect(ctx, sx - 4, cy - maxH - 2, tw + 8, maxH + 20, COL.cyan, a * glow * 0.25);
  for (let i = 0; i < nb; i++) {
    const bh = heights[i] * maxH;
    ctx.fillStyle = rgba(COL.cyan, a * (i === 0 ? 0.7 + glow * 0.3 : 0.15));
    ctx.fillRect(sx + i*(bw+bg), cy - bh, bw, bh);
  }
  ctx.fillStyle = rgba(COL.white, a * 0.85);
  ctx.font = `500 ${10}px 'IBM Plex Mono', monospace`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText('"' + text + '"', cx, cy + 3);
}

// Measures token pill positions for a centered row
function measurePills(ctx, tokens, fontSize, gap) {
  ctx.font = `500 ${fontSize}px 'IBM Plex Mono', monospace`;
  const widths = tokens.map(tok => ctx.measureText(tok).width + fontSize * 0.9);
  const totalW = widths.reduce((s, w) => s + w, 0) + (tokens.length - 1) * gap;
  return { widths, totalW };
}

// ── Section Manager ──

function createSection(sectionEl, drawFn, duration) {
  const canvas = sectionEl.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const playBtn = sectionEl.querySelector('.btn-play');
  const resetBtn = sectionEl.querySelector('.btn-reset');
  let W = 0, H = 0, t = 0, playing = false, playStartWall = 0, tAtPlay = 0;

  function resize() {
    const wrap = canvas.parentElement;
    W = Math.floor(wrap.getBoundingClientRect().width);
    H = Math.round(W * 0.32);
    canvas.width = W; canvas.height = H;
  }

  function frame(now) {
    if (sectionEl.dataset.seek) {
      t = parseFloat(sectionEl.dataset.seek);
      playing = false;
      playBtn.textContent = 'play';
      delete sectionEl.dataset.seek;
    } else if (playing) {
      t = Math.min(tAtPlay + (now - playStartWall) / 1000, duration);
      if (t >= duration) { playing = false; playBtn.textContent = 'play'; }
    }
    sectionEl.dataset.t = t.toFixed(3);
    drawFn(ctx, W, H, t);
    requestAnimationFrame(frame);
  }

  playBtn.onclick = () => {
    if (t >= duration) t = 0;
    playing = !playing;
    if (playing) { playStartWall = performance.now(); tAtPlay = t; }
    playBtn.textContent = playing ? 'pause' : 'play';
  };
  resetBtn.onclick = () => {
    t = 0; playing = false;
    playBtn.textContent = 'play';
  };

  resize();
  window.addEventListener('resize', resize);
  requestAnimationFrame(frame);
}

// ════════════════════════════════════════════════════════════
//  SECTION 1 — Autoregressive Generation
//  Single model, growing token chain with per-token embeddings
// ════════════════════════════════════════════════════════════

function drawSec1(ctx, W, H, t) {
  drawBg(ctx, W, H);
  const s = W / 1100;
  const cx = W / 2;
  const lSz = Math.round(32 * s);
  const lY = [0.30, 0.43, 0.56, 0.69].map(f => Math.round(f * H));
  const tokY = Math.round(0.10 * H);
  const embY = Math.round(0.19 * H);
  const predY = Math.round(0.84 * H);
  const embSz = Math.round(6 * s);
  const resSz = Math.round(8 * s);

  const CYCLE = 2.0;
  // 6 cycles: 0-4 are forward passes, cycle 5 just appends "mat"
  const cycle = Math.min(Math.floor(t / CYCLE), N);
  const ct = clamp((t - cycle * CYCLE) / CYCLE, 0, 1);

  // Chain includes predicted tokens from previous cycles
  const chainTokens = cycle <= N - 1
    ? FULL_CHAIN.slice(0, cycle + 1)
    : FULL_CHAIN; // cycle 5: full chain with "mat"

  const pillFs = Math.round(12 * s);
  const pillGap = 4 * s;
  const { widths: pillWidths, totalW: chainW } = measurePills(ctx, chainTokens, pillFs, pillGap);

  // Compute pill X positions (centered)
  const pillXs = [];
  let drawX = cx - chainW / 2;
  for (let i = 0; i < chainTokens.length; i++) {
    pillXs.push(drawX + pillWidths[i] / 2);
    drawX += pillWidths[i] + pillGap;
  }

  // Vertical model connections (dim base + directional pulse)
  const fwdStart = 0.20;
  const connA = 0.12;
  glowLine(ctx, cx, embY + embSz, cx, lY[0] - lSz/2, COL.cyan, connA, 1.5);
  for (let l = 0; l < L-1; l++)
    glowLine(ctx, cx, lY[l] + lSz/2, cx, lY[l+1] - lSz/2, COL.cyan, connA, 1.5);
  glowLine(ctx, cx, lY[L-1] + lSz/2, cx, predY - 10*s, COL.cyan, connA, 1.5);

  if (cycle < N) {
    const segDur = 0.06;
    const segS0 = fwdStart;
    glowLineDir(ctx, cx, embY + embSz, cx, lY[0] - lSz/2, COL.cyan,
      pop(ct, segS0, 0.03, 0.12) * 0.5, 2.5, clamp((ct - segS0) / segDur, 0, 1));
    for (let l = 0; l < L - 1; l++) {
      const segS = fwdStart + (l + 1) * 0.08;
      glowLineDir(ctx, cx, lY[l] + lSz/2, cx, lY[l+1] - lSz/2, COL.cyan,
        pop(ct, segS, 0.03, 0.12) * 0.5, 2.5, clamp((ct - segS) / segDur, 0, 1));
    }
    const segSL = fwdStart + L * 0.08;
    glowLineDir(ctx, cx, lY[L-1] + lSz/2, cx, predY - 10*s, COL.cyan,
      pop(ct, segSL, 0.03, 0.12) * 0.5, 2.5, clamp((ct - segSL) / segDur, 0, 1));
  }

  // Residual nodes between layers (below embedding convergence)
  const resPositions = [
    ...Array.from({length: L-1}, (_, l) => (lY[l] + lSz/2 + lY[l+1] - lSz/2) / 2),
    (lY[L-1] + lSz/2 + predY - 10*s) / 2,
  ];
  for (let r = 0; r < resPositions.length; r++) {
    const rAct = (cycle < N) ? fwdStart + (r + 2) * 0.07 : -1;
    const rGlow = pop(ct, rAct);
    drawActNode(ctx, cx, resPositions[r], resSz, COL.cyan, 0.6 + rGlow * 0.4, rGlow);
  }

  // Draw token pills
  for (let i = 0; i < chainTokens.length; i++) {
    const isNew = (i === cycle && cycle > 0 && cycle <= N);
    if (isNew && ct < 0.12) {
      // Slide predicted token from predY up to its position in chain
      const slideT = easeOut(ct / 0.12);
      const animCx = lerp(cx, pillXs[i], slideT);
      const animCy = lerp(predY, tokY, slideT);
      drawTokenPill(ctx, animCx, animCy, chainTokens[i], pillFs, 1, pop(ct, 0), COL.cyan);
    } else {
      const pg = (i === 0 && cycle === 0 && ct < 0.08) ? 1 - ct/0.08 : 0;
      drawTokenPill(ctx, pillXs[i], tokY, chainTokens[i], pillFs, 1, pg);
    }
  }

  // Embedding nodes: one per token, with lines from token → embedding → converge to center
  if (cycle < N || ct < 0.5) {
    const numEmb = Math.min(chainTokens.length, cycle < N ? cycle + 1 : chainTokens.length);
    for (let i = 0; i < numEmb; i++) {
      const ex = pillXs[i];
      const isNew = (i === cycle && cycle > 0 && ct < 0.12);
      if (isNew) continue; // skip during slide animation

      // Line from token down to embedding node
      glowLine(ctx, ex, tokY + 10*s, ex, embY - embSz/2, COL.cyan, 0.15, 1);

      // Embedding node
      const embAct = (cycle < N) ? fwdStart + 0.01 : -1;
      const eGlow = pop(ct, embAct);
      drawActNode(ctx, ex, embY, embSz, COL.cyan, 0.5 + eGlow * 0.5, eGlow);

      // Line from embedding node converging to center above L1
      if (Math.abs(ex - cx) > 3) {
        glowLine(ctx, ex, embY + embSz/2, cx, lY[0] - lSz/2, COL.cyan, 0.10, 1);
      }
    }
  }

  // Layer nodes with forward-pass pop-glow
  for (let l = 0; l < L; l++) {
    const lAct = (cycle < N) ? fwdStart + (l + 1) * 0.08 + 0.04 : -1;
    drawLayerNode(ctx, cx, lY[l], lSz, 'L'+(l+1), 1, pop(ct, lAct), COL.cyan);
  }

  // Prediction
  if (cycle < N) {
    const predAct = fwdStart + L * 0.10 + 0.12;
    if (ct > predAct) {
      const pa = clamp((ct - predAct) / 0.04, 0, 1);
      drawPredBar(ctx, cx, predY, PREDS[cycle], pa, pop(ct, predAct));
    }
  }
}

// ════════════════════════════════════════════════════════════
//  SECTION 2 — Model Cloning → Unrolled View
//  Each forward pass creates a new model copy.
//  Only latest token shown per column, but pulse shows all embeddings.
//  Output copies to next column's input.
// ════════════════════════════════════════════════════════════

function drawSec2(ctx, W, H, t) {
  drawBg(ctx, W, H);
  const s = W / 1100;
  const lSz = Math.round(28 * s);
  const colSp = Math.round(Math.min(170, 850 / Math.max(N - 1, 1)) * s);
  const totalW = (N - 1) * colSp;
  const leftPad = (W - totalW) / 2;
  const colX = Array.from({length: N}, (_, i) => leftPad + i * colSp);

  const tokY = Math.round(0.10 * H);
  const embY = Math.round(0.18 * H);
  const lY = [0.28, 0.41, 0.54, 0.67].map(f => Math.round(f * H));
  const predY = Math.round(0.83 * H);
  const resSz = Math.round(7 * s);
  const embSz = Math.round(5 * s);

  const CYCLE = 2.0;
  const cycle = Math.min(Math.floor(t / CYCLE), N);
  const drawCycle = Math.min(cycle, N - 1);
  const ct = cycle < N ? clamp((t - cycle * CYCLE) / CYCLE, 0, 1) : 1;

  for (let pos = 0; pos <= drawCycle; pos++) {
    const cx = colX[pos];
    const isCur = pos === drawCycle && cycle < N;
    const matAlpha = isCur ? clamp(ct / 0.08, 0, 1) : 1;
    const fwdS = 0.12;

    // Vertical connections (dim base + directional pulse)
    glowLine(ctx, cx, embY + embSz, cx, lY[0] - lSz/2, COL.cyan, matAlpha * 0.12, 1.5);
    for (let l = 0; l < L-1; l++)
      glowLine(ctx, cx, lY[l] + lSz/2, cx, lY[l+1] - lSz/2, COL.cyan, matAlpha * 0.12, 1.5);
    glowLine(ctx, cx, lY[L-1] + lSz/2, cx, predY - 10*s, COL.cyan, matAlpha * 0.12, 1.5);

    if (isCur) {
      const segDur = 0.05;
      glowLineDir(ctx, cx, embY + embSz, cx, lY[0] - lSz/2, COL.cyan,
        pop(ct, fwdS, 0.03, 0.10) * 0.5, 2.5, clamp((ct - fwdS) / segDur, 0, 1));
      for (let l = 0; l < L - 1; l++) {
        const segS = fwdS + (l + 1) * 0.07;
        glowLineDir(ctx, cx, lY[l] + lSz/2, cx, lY[l+1] - lSz/2, COL.cyan,
          pop(ct, segS, 0.03, 0.10) * 0.5, 2.5, clamp((ct - segS) / segDur, 0, 1));
      }
      const segSL = fwdS + L * 0.07;
      glowLineDir(ctx, cx, lY[L-1] + lSz/2, cx, predY - 10*s, COL.cyan,
        pop(ct, segSL, 0.03, 0.10) * 0.5, 2.5, clamp((ct - segSL) / segDur, 0, 1));
    }

    // Residual nodes between layers
    const resPs = [
      ...Array.from({length: L-1}, (_, l) => (lY[l] + lSz/2 + lY[l+1] - lSz/2) / 2),
      (lY[L-1] + lSz/2 + predY - 10*s) / 2,
    ];
    for (let r = 0; r < resPs.length; r++) {
      const rAct = isCur ? fwdS + (r + 1) * 0.06 : -1;
      const rG = isCur ? pop(ct, rAct) : 0;
      drawActNode(ctx, cx, resPs[r], resSz, COL.cyan, matAlpha * (0.5 + rG * 0.5), rG);
    }

    // Main token pill — only the LATEST token for this position
    const fs = Math.round(10 * s);

    // Sliding animation: predicted token from previous column arrives at start of this cycle
    if (isCur && pos > 0 && ct < 0.10) {
      const slideT = easeOut(ct / 0.10);
      const fromX = colX[pos - 1];
      const animCx = lerp(fromX, cx, slideT);
      const animCy = lerp(predY, tokY, slideT);
      drawTokenPill(ctx, animCx, animCy, TOKENS[pos], fs, 1, pop(ct, 0), COL.cyan);
    } else {
      // All tokens glow briefly at each cycle start, not just the current one
      const allGlow = cycle < N ? pop(ct, 0.01, 0.05, 0.20) : 0;
      const tokGlow = isCur ? pop(ct, 0.02) : allGlow;
      drawTokenPill(ctx, cx, tokY, TOKENS[pos], fs, matAlpha, tokGlow);
    }

    // Embedding nodes for ALL accumulated tokens (shows model processes all of them)
    // Brief flash during forward pass start, then fade
    const numEmb = pos + 1;
    const embSpread = Math.round(12 * s);
    const embTotalW = (numEmb - 1) * embSpread;
    const embStartX = cx - embTotalW / 2;
    for (let ei = 0; ei < numEmb; ei++) {
      const ex = embStartX + ei * embSpread;
      const embAct = isCur ? fwdS : -1;
      const eGlow = isCur ? pop(ct, embAct, 0.06, 0.3) : 0;
      const eAlpha = isCur ? matAlpha * (0.3 + eGlow * 0.7) : 0.2;
      drawActNode(ctx, ex, embY, embSz, COL.cyan, eAlpha, eGlow);
      // Line from embedding to center (converge to model)
      if (Math.abs(ex - cx) > 2)
        glowLine(ctx, ex, embY + embSz/2, cx, lY[0] - lSz/2, COL.cyan, eAlpha * 0.4, 0.8);
    }
    // Center embedding line down
    glowLine(ctx, cx, embY + embSz/2, cx, lY[0] - lSz/2, COL.cyan, matAlpha * 0.15, 1);

    // Layer nodes
    for (let l = 0; l < L; l++) {
      const lAct = isCur ? fwdS + (l+1) * 0.07 + 0.02 : -1;
      drawLayerNode(ctx, cx, lY[l], lSz, 'L'+(l+1), matAlpha, isCur ? pop(ct, lAct) : 0, COL.cyan);
    }

    // Prediction
    const predAct = fwdS + L * 0.09 + 0.08;
    const showPred = isCur ? ct > predAct : true;
    if (showPred) {
      const pa = isCur ? clamp((ct - predAct) / 0.04, 0, 1) : 1;
      drawPredBar(ctx, cx, predY, PREDS[pos], pa * matAlpha, isCur ? pop(ct, predAct) : 0);
    }

    // Position label
    ctx.fillStyle = rgba(COL.dim, matAlpha * 0.35);
    ctx.font = `${Math.round(9*s)}px 'IBM Plex Mono', monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('pos ' + pos, cx, predY + 20*s);
  }

  // Epilogue: final predicted token flies up to become input
  if (cycle >= N) {
    const ect = clamp((t - N * CYCLE) / CYCLE, 0, 1);
    const lastCx = colX[N - 1];
    const newCx = lastCx + colSp;
    const fs = Math.round(10 * s);
    if (ect < 0.15) {
      const slideT = easeOut(ect / 0.15);
      drawTokenPill(ctx, lerp(lastCx, newCx, slideT), lerp(predY, tokY, slideT),
        PREDS[N - 1], fs, 1, pop(ect, 0), COL.cyan);
    } else {
      drawTokenPill(ctx, newCx, tokY, PREDS[N - 1], fs, 1, pop(ect, 0.15, 0.05, 0.3));
    }
  }
}

// ════════════════════════════════════════════════════════════
//  SECTION 3 — KV Cache
//  Only latest token enters; KV from previous positions
//  feeds into each layer. KV glows BEFORE the layer it feeds.
//  Order: residual → KV read → layer → KV output
// ════════════════════════════════════════════════════════════

function drawSec3(ctx, W, H, t) {
  drawBg(ctx, W, H);
  const s = W / 1100;
  const lSz = Math.round(28 * s);
  const kvSz = Math.round(14 * s);
  const resSz = Math.round(8 * s);
  const kvOff = Math.round(40 * s);
  const colSp = Math.round(Math.min(175, 900 / Math.max(N - 1, 1)) * s);
  const totalW = (N-1) * colSp;
  const leftPad = (W - totalW) / 2;
  const colX = Array.from({length: N}, (_, i) => leftPad + i * colSp);

  const tokY = Math.round(0.10 * H);
  const lY = [0.28, 0.41, 0.54, 0.67].map(f => Math.round(f * H));
  const predY = Math.round(0.83 * H);

  const CYCLE = 2.4;
  const cycle = Math.min(Math.floor(t / CYCLE), N);
  const drawCycle = Math.min(cycle, N - 1);
  const ct = cycle < N ? clamp((t - cycle * CYCLE) / CYCLE, 0, 1) : 1;

  // KV bus lines
  const busOff = Math.round(18 * s);
  const busVisible = drawCycle > 0 || ct > 0.2;
  if (busVisible) {
    const busAlpha = cycle === 0 ? clamp((ct - 0.2) / 0.15, 0, 0.08) : 0.08;
    for (let l = 0; l < L; l++) {
      const busY = lY[l] - busOff;
      const x1 = colX[0] - 15 * s;
      const lastVisPos = Math.min(drawCycle, N - 1);
      const x2 = colX[lastVisPos] + kvOff + kvSz/2 + 10 * s;
      glowLine(ctx, x1, busY, x2, busY, COL.amber, busAlpha, 1);
    }
  }

  // Per-layer timing: stepTime controls spacing between layers
  const fwdS = 0.10;
  const stepTime = 0.10;
  // For each layer l:
  //   KV read (supply rails):  fwdS + l * stepTime
  //   Layer fires:             fwdS + l * stepTime + 0.04
  //   KV output:               fwdS + l * stepTime + 0.07

  for (let pos = 0; pos <= drawCycle; pos++) {
    const cx = colX[pos];
    const isCur = pos === drawCycle && cycle < N;
    const matA = isCur ? clamp(ct / 0.08, 0, 1) : 1;

    // Vertical connections (dim base + directional pulse)
    glowLine(ctx, cx, tokY + 10*s, cx, lY[0] - lSz/2, COL.cyan, matA * 0.12, 1.5);
    for (let l = 0; l < L-1; l++)
      glowLine(ctx, cx, lY[l] + lSz/2, cx, lY[l+1] - lSz/2, COL.cyan, matA * 0.12, 1.5);
    glowLine(ctx, cx, lY[L-1] + lSz/2, cx, predY - 10*s, COL.cyan, matA * 0.12, 1.5);

    if (isCur) {
      const segDur = 0.05;
      glowLineDir(ctx, cx, tokY + 10*s, cx, lY[0] - lSz/2, COL.cyan,
        pop(ct, fwdS, 0.03, 0.10) * 0.5, 2.5, clamp((ct - fwdS) / segDur, 0, 1));
      for (let l = 0; l < L - 1; l++) {
        const segS = fwdS + (l + 1) * stepTime;
        glowLineDir(ctx, cx, lY[l] + lSz/2, cx, lY[l+1] - lSz/2, COL.cyan,
          pop(ct, segS, 0.03, 0.10) * 0.5, 2.5, clamp((ct - segS) / segDur, 0, 1));
      }
      const segSL = fwdS + L * stepTime;
      glowLineDir(ctx, cx, lY[L-1] + lSz/2, cx, predY - 10*s, COL.cyan,
        pop(ct, segSL, 0.03, 0.10) * 0.5, 2.5, clamp((ct - segSL) / segDur, 0, 1));
    }

    // Residual nodes between layers
    const resPs = [
      (tokY + 10*s + lY[0] - lSz/2) / 2,
      ...Array.from({length: L-1}, (_, l) => (lY[l] + lSz/2 + lY[l+1] - lSz/2) / 2),
      (lY[L-1] + lSz/2 + predY - 10*s) / 2,
    ];
    for (let r = 0; r < resPs.length; r++) {
      const rAct = isCur ? fwdS + r * stepTime - 0.01 : -1;
      const rG = isCur ? pop(ct, rAct) : 0;
      drawActNode(ctx, cx, resPs[r], resSz, COL.cyan, matA * (0.5 + rG * 0.5), rG);
    }

    // Latest token — with fly-up animation from previous column's prediction
    const fs = Math.round(11 * s);
    if (isCur && pos > 0 && ct < 0.10) {
      const slideT = easeOut(ct / 0.10);
      const fromX = colX[pos - 1];
      const animCx = lerp(fromX, cx, slideT);
      const animCy = lerp(predY, tokY, slideT);
      drawTokenPill(ctx, animCx, animCy, TOKENS[pos], fs, 1, pop(ct, 0), COL.cyan);
    } else {
      const tokGlow = isCur ? pop(ct, 0.02) : 0;
      drawTokenPill(ctx, cx, tokY, TOKENS[pos], fs, matA, tokGlow);
    }

    // KV supply rails from previous positions — glow BEFORE layer fires
    if (pos > 0) {
      for (let l = 0; l < L; l++) {
        const busY = lY[l] - busOff;
        const kvReadT = fwdS + l * stepTime;
        for (let src = 0; src < pos; src++) {
          const srcKvX = colX[src] + kvOff;
          const railGlow = isCur ? pop(ct, kvReadT, 0.06, 0.2) : 0;

          // Source KV box glows green when being read
          if (isCur && railGlow > 0) {
            const kx = srcKvX - kvSz/2, ky = lY[l] - kvSz/2;
            glowRect(ctx, kx, ky, kvSz, kvSz, COL.green, railGlow * 0.7);
          }

          // Directional rail: travels from source KV toward current station
          const railProgress = isCur ? clamp((ct - kvReadT) / 0.10, 0, 1) : 1;
          const railA = isCur ? clamp((ct - kvReadT) / 0.08, 0, 1) * 0.3 : 0.15;
          glowLineDir(ctx, srcKvX, busY, cx, busY, COL.green,
            matA * (railA + railGlow * 0.15), 1.2, railProgress);
        }
        // Directional read tick from bus down to station
        const tickProgress = isCur ? clamp((ct - kvReadT - 0.06) / 0.04, 0, 1) : 1;
        const tickA = isCur ? clamp((ct - kvReadT) / 0.08, 0, 1) * 0.25 : 0.12;
        glowLineDir(ctx, cx, busY, cx, lY[l] - lSz/2, COL.green, matA * tickA, 1, tickProgress);
      }
    }

    // Layer nodes — fire AFTER KV read
    for (let l = 0; l < L; l++) {
      const lAct = isCur ? fwdS + l * stepTime + 0.04 : -1;
      drawLayerNode(ctx, cx, lY[l], lSz, 'L'+(l+1), matA, isCur ? pop(ct, lAct) : 0, COL.cyan);
    }

    // KV output nodes — appear AFTER layer fires, directional write
    for (let l = 0; l < L; l++) {
      const kvAct = isCur ? fwdS + l * stepTime + 0.07 : -1;
      const kvG = isCur ? pop(ct, kvAct) : 0;
      const kvCx = cx + kvOff;
      const busY = lY[l] - busOff;

      // Directional: layer → KV box (rightward)
      const kvWriteP = isCur ? clamp((ct - (fwdS + l * stepTime + 0.05)) / 0.04, 0, 1) : 1;
      glowLineDir(ctx, cx + lSz/2 + 1, lY[l], kvCx - kvSz/2 - 1, lY[l],
        COL.amber, matA * (0.2 + kvG * 0.15), 1, kvWriteP);
      // Directional: KV box → bus (upward)
      const kvBusP = isCur ? clamp((ct - (fwdS + l * stepTime + 0.08)) / 0.03, 0, 1) : 1;
      glowLineDir(ctx, kvCx, lY[l] - kvSz/2, kvCx, busY,
        COL.amber, matA * (0.15 + kvG * 0.1), 1, kvBusP);
      drawActNode(ctx, kvCx, lY[l], kvSz, COL.amber, matA, kvG);
      ctx.fillStyle = rgba(COL.amber, matA * (0.35 + kvG * 0.3));
      ctx.font = `600 ${Math.round(kvSz * 0.5)}px 'IBM Plex Mono', monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('KV', kvCx, lY[l] + 0.5);
    }

    // Prediction
    const predAct = fwdS + (L - 1) * stepTime + 0.12;
    const showPred = isCur ? ct > predAct : true;
    if (showPred) {
      const pa = isCur ? clamp((ct - predAct) / 0.04, 0, 1) : 1;
      drawPredBar(ctx, cx, predY, PREDS[pos], pa * matA, isCur ? pop(ct, predAct) : 0);
    }

    // Position label
    ctx.fillStyle = rgba(COL.dim, matA * 0.35);
    ctx.font = `${Math.round(9*s)}px 'IBM Plex Mono', monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('pos ' + pos, cx, predY + 20*s);
  }

  // Epilogue: final predicted token flies up to become input
  if (cycle >= N) {
    const ect = clamp((t - N * CYCLE) / CYCLE, 0, 1);
    const lastCx = colX[N - 1];
    const newCx = lastCx + colSp;
    const fs = Math.round(11 * s);
    if (ect < 0.15) {
      const slideT = easeOut(ect / 0.15);
      drawTokenPill(ctx, lerp(lastCx, newCx, slideT), lerp(predY, tokY, slideT),
        PREDS[N - 1], fs, 1, pop(ect, 0), COL.cyan);
    } else {
      drawTokenPill(ctx, newCx, tokY, PREDS[N - 1], fs, 1, pop(ect, 0.15, 0.05, 0.3));
    }
  }
}

// ════════════════════════════════════════════════════════════
//  SECTION 4 — Backpropagation
//  Gradient flows from loss backward through layers and KV.
//  Diagonal wavefront from bottom-right to top-left.
//  Prominent loss indicator with visible tokens/predictions.
// ════════════════════════════════════════════════════════════

function drawSec4(ctx, W, H, t) {
  drawBg(ctx, W, H);
  const s = W / 1100;
  const lSz = Math.round(28 * s);
  const kvSz = Math.round(14 * s);
  const resSz = Math.round(8 * s);
  const kvOff = Math.round(40 * s);
  const colSp = Math.round(Math.min(175, 900 / Math.max(N - 1, 1)) * s);
  const totalW = (N-1) * colSp;
  const leftPad = (W - totalW) / 2;
  const colX = Array.from({length: N}, (_, i) => leftPad + i * colSp);

  const tokY = Math.round(0.10 * H);
  const lY = [0.28, 0.41, 0.54, 0.67].map(f => Math.round(f * H));
  const predY = Math.round(0.83 * H);
  const busOff = Math.round(18 * s);

  // Static base: draw the full KV-cached unrolled view (dimmed)
  // KV bus lines
  for (let l = 0; l < L; l++) {
    const busY = lY[l] - busOff;
    const x1 = colX[0] - 15 * s;
    const x2 = colX[N-1] + kvOff + kvSz/2 + 10 * s;
    glowLine(ctx, x1, busY, x2, busY, COL.amber, 0.06, 1);
  }

  for (let pos = 0; pos < N; pos++) {
    const cx = colX[pos];

    // Vertical connections (dimmed)
    glowLine(ctx, cx, tokY + 10*s, cx, lY[0] - lSz/2, COL.cyan, 0.07, 1);
    for (let l = 0; l < L-1; l++)
      glowLine(ctx, cx, lY[l] + lSz/2, cx, lY[l+1] - lSz/2, COL.cyan, 0.07, 1);
    glowLine(ctx, cx, lY[L-1] + lSz/2, cx, predY - 10*s, COL.cyan, 0.07, 1);

    // Residual nodes
    const resPs = [
      (tokY + 10*s + lY[0] - lSz/2) / 2,
      ...Array.from({length: L-1}, (_, l) => (lY[l] + lSz/2 + lY[l+1] - lSz/2) / 2),
      (lY[L-1] + lSz/2 + predY - 10*s) / 2,
    ];
    for (let r = 0; r < resPs.length; r++)
      drawActNode(ctx, cx, resPs[r], resSz, COL.cyan, 0.25, 0);

    // Tokens — more visible
    drawTokenPill(ctx, cx, tokY, TOKENS[pos], Math.round(11*s), 0.65, 0);

    // Layers (dimmed)
    for (let l = 0; l < L; l++)
      drawLayerNode(ctx, cx, lY[l], lSz, 'L'+(l+1), 0.3, 0, COL.cyan);

    // KV nodes (dimmed)
    for (let l = 0; l < L; l++) {
      const kvCx = cx + kvOff;
      const busY = lY[l] - busOff;
      glowLine(ctx, cx + lSz/2 + 1, lY[l], kvCx - kvSz/2 - 1, lY[l], COL.amber, 0.08, 1);
      glowLine(ctx, kvCx, lY[l] - kvSz/2, kvCx, busY, COL.amber, 0.06, 1);
      drawActNode(ctx, kvCx, lY[l], kvSz, COL.amber, 0.3, 0);
    }

    // Supply rails (dim)
    if (pos > 0) {
      for (let l = 0; l < L; l++) {
        const busY = lY[l] - busOff;
        for (let src = 0; src < pos; src++)
          glowLine(ctx, colX[src] + kvOff, busY, cx, busY, COL.green, 0.04, 1);
        glowLine(ctx, cx, busY, cx, lY[l] - lSz/2, COL.green, 0.03, 1);
      }
    }

    // Predictions — more visible
    drawPredBar(ctx, cx, predY, PREDS[pos], 0.55, 0);

    // Position label
    ctx.fillStyle = rgba(COL.dim, 0.3);
    ctx.font = `${Math.round(9*s)}px 'IBM Plex Mono', monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('pos ' + pos, cx, predY + 20*s);
  }

  // ── Prominent loss indicator at last position ──
  const lossAppear = 0.3;
  if (t > lossAppear) {
    const la = clamp((t - lossAppear) / 0.3, 0, 1);
    const lcx = colX[N-1];
    const lossGlow = pop(t, lossAppear, 0.2, 0.8);

    // Red glow zone around prediction area
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = rgba(COL.red, la * (0.04 + lossGlow * 0.08));
    ctx.fillRect(lcx - 40*s, predY - 22, 80*s, 36);
    ctx.restore();

    // Actual token label
    ctx.fillStyle = rgba(COL.white, la * 0.8);
    ctx.font = `500 ${Math.round(10*s)}px 'IBM Plex Mono', monospace`;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText('actual:', lcx - 28*s, predY - 14);

    ctx.fillStyle = rgba(COL.green, la * 0.9);
    ctx.textAlign = 'left';
    ctx.fillText('"' + FULL_CHAIN[FULL_CHAIN.length - 1] + '"', lcx - 24*s, predY - 14);

    // Loss label with glow
    ctx.fillStyle = rgba(COL.red, la * (0.7 + lossGlow * 0.3));
    ctx.font = `600 ${Math.round(11*s)}px 'IBM Plex Mono', monospace`;
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('← loss', lcx + 30*s, predY - 4);

    // Delta arrow from prediction up to model
    if (lossGlow > 0.3) {
      const arrowA = la * 0.4;
      ctx.save();
      ctx.strokeStyle = rgba(COL.red, arrowA);
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(lcx, predY - 20);
      ctx.lineTo(lcx, predY - 10*s);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }
  }

  // ── Gradient wavefront ──
  const gradStart = 0.8;
  const waveSpeed = 0.55;

  for (let pos = N - 1; pos >= 0; pos--) {
    const cx = colX[pos];
    const posFromRight = N - 1 - pos;

    for (let l = L - 1; l >= 0; l--) {
      const layerFromBottom = L - 1 - l;
      const diag = posFromRight + layerFromBottom;
      const activationT = gradStart + diag * waveSpeed;
      const glow = pop(t, activationT, 0.12, 0.6);

      if (t < activationT) continue;

      const ga = clamp((t - activationT) / 0.15, 0, 1);
      const steadyGlow = ga * 0.5;
      const totalGlow = Math.max(glow, steadyGlow);
      drawLayerNode(ctx, cx, lY[l], lSz, 'L'+(l+1), 0.5 + ga * 0.5, totalGlow, COL.magenta);

      // Gradient on vertical connection (directional: upward from this layer)
      if (l > 0) {
        const upA = ga * 0.35;
        const upProgress = clamp((t - activationT) / 0.25, 0, 1);
        glowLineDir(ctx, cx, lY[l] - lSz/2, cx, lY[l-1] + lSz/2, COL.magenta, upA, 2, upProgress);
      } else {
        // L1 → embedding: gradient continues upward to token embedding
        const upA = ga * 0.35;
        const upProgress = clamp((t - activationT) / 0.25, 0, 1);
        glowLineDir(ctx, cx, lY[0] - lSz/2, cx, tokY + 10*s, COL.magenta, upA, 2, upProgress);
      }

      // Gradient on KV (travels progressively leftward)
      if (pos > 0) {
        const kvCx = cx + kvOff;
        const busY = lY[l] - busOff;
        const busA = ga * 0.15;
        const elapsed = t - activationT;
        for (let src = pos - 1; src >= 0; src--) {
          const dist = pos - src;
          const travelProgress = clamp(elapsed / (0.4 * dist), 0, 1);
          if (travelProgress <= 0) continue;
          const srcKvX = colX[src] + kvOff;
          const endX = lerp(cx, srcKvX, travelProgress);
          glowLine(ctx, cx, busY, endX, busY, COL.magenta, busA, 1.5);
        }
        glowLine(ctx, cx + lSz/2, lY[l], kvCx - kvSz/2, lY[l], COL.magenta, busA, 1);
        glowLine(ctx, kvCx, lY[l] - kvSz/2, kvCx, busY, COL.magenta, busA, 1);
      }

      // KV node glow
      const kvCx = cx + kvOff;
      drawActNode(ctx, kvCx, lY[l], kvSz, COL.magenta, 0.5 + ga * 0.5, Math.max(glow * 0.8, steadyGlow * 0.6));
    }

    // Residual glow
    const resPs = [
      (tokY + 10*s + lY[0] - lSz/2) / 2,
      ...Array.from({length: L-1}, (_, l) => (lY[l] + lSz/2 + lY[l+1] - lSz/2) / 2),
      (lY[L-1] + lSz/2 + predY - 10*s) / 2,
    ];
    for (let r = 0; r < resPs.length; r++) {
      // Unembedding residual only gets gradient at loss source position
      if (r === L && pos !== N - 1) continue;
      const layerFromBottom = L - 1 - Math.min(r, L - 1);
      const diag = posFromRight + layerFromBottom;
      const activationT = gradStart + diag * waveSpeed;
      const rGlow = pop(t, activationT);
      if (t >= activationT) {
        const rA = clamp((t - activationT) / 0.1, 0, 1);
        drawActNode(ctx, cx, resPs[r], resSz, COL.magenta, 0.4 + rA * 0.6, rGlow);
      }
    }
  }

  // Gradient arrow from loss upward
  if (t > gradStart - 0.2) {
    const aa = clamp((t - gradStart + 0.2) / 0.2, 0, 1);
    const lcx = colX[N-1];
    glowLine(ctx, lcx, predY - 10*s, lcx, lY[L-1] + lSz/2, COL.magenta, aa * 0.5, 2);
  }
}

// ── Init ──

document.addEventListener('DOMContentLoaded', () => {
  const SEC1_CYCLE = 2.0, SEC2_CYCLE = 2.0, SEC3_CYCLE = 2.4;
  const sec4GradStart = 0.8, sec4WaveSpeed = 0.55;
  createSection(document.getElementById('sec1'), drawSec1, (N + 1) * SEC1_CYCLE);
  createSection(document.getElementById('sec2'), drawSec2, (N + 1) * SEC2_CYCLE);
  createSection(document.getElementById('sec3'), drawSec3, (N + 1) * SEC3_CYCLE);
  createSection(document.getElementById('sec4'), drawSec4,
    sec4GradStart + (N + L - 2) * sec4WaveSpeed + 2.0);
});
