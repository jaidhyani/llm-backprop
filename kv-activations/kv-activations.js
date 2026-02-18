// Beyond Next-Token Prediction — KV Activations Visualization
// Interactive step-through: how KV cache implements anticipatory state

// ============================================================
//  CONFIGURATION
// ============================================================

const TOKENS = ['The', 'cat', 'sat', 'on', 'the'];
const PREDS  = ['cat', 'sat', 'on', 'the', 'mat'];
const NUM_POS = 5;
const NUM_LAYERS = 4;
const TOTAL_STEPS = 9;

const C = {
  canvasBg:     [12, 16, 34],

  // model layers
  layerFill:    [22, 34, 60],
  layerLight:   [34, 50, 82],
  layerStroke:  [42, 60, 95],
  layerLabel:   [78, 98, 135],

  // KV activations
  kv:           [196, 127, 10],
  kvBright:     [245, 180, 55],

  // forward flow
  fwd:          [45, 212, 191],

  // gradient / backprop
  grad:         [248, 113, 113],

  // attention
  attn:         [100, 170, 255],

  // text
  text:         [226, 232, 240],
  textDim:      [100, 118, 145],
  textMuted:    [50, 62, 88],

  // loss
  loss:         [255, 120, 120],

  white:        [255, 255, 255],
};

// ============================================================
//  STEP DEFINITIONS
// ============================================================

const STEPS = [
  {
    title: 'One Forward Pass',
    html: 'A transformer takes a token, passes it through <em>a series of layers</em>, and outputs a probability distribution over possible next tokens. Each layer transforms the representation, building richer understanding as it goes deeper.',
    dur: 1.2,
  },
  {
    title: 'The Unrolled View',
    html: 'During generation, this happens for <em>every token in sequence</em>. Each position runs the same model with the same weights &mdash; laid out side by side, it looks like a chain of independent predictions.',
    dur: 1.0,
  },
  {
    title: 'The Second Output',
    html: 'But each forward pass produces more than just a prediction. At every layer, the model generates <span class="kv">key-value (K,V) activations</span> &mdash; rich, structured representations that persist in memory. The prediction is consumed and forgotten. The <span class="kv">K,V endures</span>.',
    dur: 2.0,
  },
  {
    title: 'Where Information Comes From',
    html: 'These <span class="kv">K,V activations</span> aren\'t just stored &mdash; they\'re <em>consumed by every future position</em>. At each layer, the current token combines its own upstream representation with all previous positions\' K,V at that layer. <em>Click any layer</em> to see where its information comes from.',
    dur: 1.2,
  },
  {
    title: 'The Training Signal',
    html: 'During training, the model\'s predicted distribution is compared against the actual next token. The gap between prediction and reality is the <span class="grad">loss</span> &mdash; and there is <em>always</em> a loss signal, even when the top prediction is correct.',
    dur: 0.8,
  },
  {
    title: 'Local Backpropagation',
    html: 'The <span class="grad">gradient</span> flows upward through the model\'s layers at this position &mdash; standard backpropagation. Each layer\'s weights receive a signal about how to adjust. But this is only the beginning.',
    dur: 1.5,
  },
  {
    title: 'The Gradient Cascade',
    html: 'The <span class="grad">gradient</span> doesn\'t stop here. It flows backward through the <span class="fwd">attention connections</span> to every previous position\'s <span class="kv">K,V</span>. When it arrives, the chain rule continues &mdash; cascading down through each position\'s layers, with <em>additional K,V gradient joining at each layer</em>.',
    dur: 2.8,
  },
  {
    title: 'Gradient Accumulation',
    html: 'This happens from <em>every</em> future position\'s loss. The <span class="kv">K,V</span> at early positions receive <span class="grad">gradient</span> from the entire future. Each forward pass deposits K,V optimized for <em>everything that will ever read it</em>. Watch as gradient accumulates from each source.',
    dur: 2.5,
  },
  {
    title: 'Anticipatory State',
    html: 'Consider <span class="kv">&ldquo;The&rdquo;</span>. Its K,V activations have been shaped by <span class="grad">gradient</span> from thousands of training documents &mdash; every context where &ldquo;The&rdquo; appeared, the loss from <em>every future token</em> refined what its K,V should represent. In a conversation with an LLM, <em>every forward pass generates K,V activations that anticipate the rest of the conversation</em>. The KV cache isn\'t a performance trick. It\'s the model\'s persistent, layered prediction of everything that follows.',
    dur: 0.8,
  },
];

// ============================================================
//  STATE
// ============================================================

let canvasWidth  = 400;
let drawHeight   = 420;
let currentStep  = 0;
let animT        = 0;
let animDuration = 1.2;
let animStartTime = 0;
let hoveredPos   = -1;
let selectedPos  = -1;
let selectedLayer = -1;
let sourceAnimT  = 0;
let sourceAnimStart = 0;
let L = {};

// ============================================================
//  LAYOUT
// ============================================================

function calcLayout() {
  const w = canvasWidth;
  const h = drawHeight;
  const mx = Math.max(50, w * 0.055);

  const modelW  = constrain(w * 0.085, 75, 110);
  const layerH  = constrain(h * 0.088, 32, 46);
  const layerGap = 5;
  const modelH  = NUM_LAYERS * layerH + (NUM_LAYERS - 1) * layerGap;

  const kvW   = constrain(modelW * 0.28, 18, 32);
  const kvGap = constrain(modelW * 0.1, 7, 14);

  const tokenY   = 38;
  const modelTop = 86;
  const modelBot = modelTop + modelH;
  const predY    = modelBot + 48;

  // account for KV stacks extending right of the rightmost position
  const leftPad  = modelW / 2 + 12;
  const rightPad = modelW / 2 + kvGap + kvW + 12;
  const usable   = w - leftPad - rightPad;
  const spacing  = usable / (NUM_POS - 1);
  const posX = [];
  for (let i = 0; i < NUM_POS; i++) posX.push(leftPad + i * spacing);

  const layerY = [];
  for (let l = 0; l < NUM_LAYERS; l++) {
    layerY.push(modelTop + l * (layerH + layerGap));
  }

  L = {
    w, h, mx, modelW, layerH, layerGap, modelH, kvW, kvGap,
    tokenY, modelTop, modelBot, predY,
    posX, spacing, layerY,
    cx: w / 2,
  };
}

function getPosX(i) {
  // step 0: single model centered
  if (currentStep === 0) return L.cx;
  // step 1: position 0 slides from center to its final spot, others fade in
  if (currentStep === 1 && animT < 1) {
    if (i === 0) return lerp(L.cx, L.posX[0], easeOut(animT));
    return L.posX[i];
  }
  return L.posX[i];
}

// ============================================================
//  EASING
// ============================================================

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
function easeInOut(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// ============================================================
//  DRAWING UTILITIES
// ============================================================

function sf(c, a) { fill(c[0], c[1], c[2], a !== undefined ? a : 255); }
function ss(c, a) { stroke(c[0], c[1], c[2], a !== undefined ? a : 255); }

function drawArrow(x1, y1, x2, y2, col, weight, alpha) {
  const a = alpha !== undefined ? alpha : 255;
  const w = weight || 1.5;
  ss(col, a);
  strokeWeight(w);
  line(x1, y1, x2, y2);
  // arrowhead
  const ang = atan2(y2 - y1, x2 - x1);
  const sz = Math.max(6, w * 3);
  push();
  translate(x2, y2);
  rotate(ang);
  noStroke();
  sf(col, a);
  triangle(0, 0, -sz, -sz * 0.38, -sz, sz * 0.38);
  pop();
}

function drawCurvedArrow(x1, y1, x2, y2, curveAmt, col, weight, alpha) {
  const a = alpha !== undefined ? alpha : 255;
  const w = weight || 1.5;
  ss(col, a);
  strokeWeight(w);
  noFill();
  const cx1 = lerp(x1, x2, 0.3);
  const cy1 = y1 + curveAmt * 0.7;
  const cx2 = lerp(x1, x2, 0.7);
  const cy2 = y2 + curveAmt * 0.7;
  bezier(x1, y1, cx1, cy1, cx2, cy2, x2, y2);
  // arrowhead tangent at endpoint
  const dx = x2 - cx2;
  const dy = y2 - cy2;
  const ang = atan2(dy, dx);
  const sz = Math.max(6, w * 3);
  push();
  translate(x2, y2);
  rotate(ang);
  noStroke();
  sf(col, a);
  triangle(0, 0, -sz, -sz * 0.38, -sz, sz * 0.38);
  pop();
}

function drawGlow(x, y, w, h, col, intensity) {
  noStroke();
  sf(col, intensity * 15);
  rect(x - 7, y - 7, w + 14, h + 14, 8);
  sf(col, intensity * 30);
  rect(x - 3, y - 3, w + 6, h + 6, 5);
}

// ============================================================
//  BACKGROUND
// ============================================================

function drawBackground() {
  background(C.canvasBg);
  const sp = 30;
  noStroke();
  for (let gx = sp; gx < L.w; gx += sp) {
    for (let gy = sp; gy < L.h; gy += sp) {
      const fade = Math.min(gx, L.w - gx, gy, L.h - gy) / 50;
      sf(C.textMuted, Math.min(1, fade) * 14);
      circle(gx, gy, 1.5);
    }
  }
}

// ============================================================
//  COMPONENT: MODEL COLUMN
// ============================================================

function drawModelColumn(posIdx, alpha) {
  const x = getPosX(posIdx);
  const a = alpha !== undefined ? alpha : 255;
  const hw = L.modelW / 2;

  for (let l = 0; l < NUM_LAYERS; l++) {
    const lx = x - hw;
    const ly = L.layerY[l];
    const lh = L.layerH;

    // shadow
    noStroke();
    sf(C.canvasBg, a * 0.4);
    rect(lx + 2, ly + 2, L.modelW, lh, 5);

    // main layer
    sf(C.layerFill, a);
    ss(C.layerStroke, a * 0.5);
    strokeWeight(1);
    rect(lx, ly, L.modelW, lh, 5);

    // top gradient highlight
    noStroke();
    sf(C.layerLight, a * 0.4);
    rect(lx + 1, ly + 1, L.modelW - 2, lh * 0.38, 4, 4, 0, 0);

    // label
    sf(C.layerLabel, a * 0.65);
    textAlign(CENTER, CENTER);
    textSize(11);
    textFont('monospace');
    text('L' + (l + 1), x, ly + lh / 2 + 1);
  }

  // thin connecting lines between layers
  for (let l = 0; l < NUM_LAYERS - 1; l++) {
    ss(C.layerStroke, a * 0.2);
    strokeWeight(1);
    line(x, L.layerY[l] + L.layerH, x, L.layerY[l + 1]);
  }
}

// ============================================================
//  COMPONENT: TOKEN BADGE
// ============================================================

function drawToken(posIdx, alpha) {
  const x = getPosX(posIdx);
  const a = alpha !== undefined ? alpha : 255;
  const tok = TOKENS[posIdx];

  textSize(15);
  textFont('monospace');
  const tw = textWidth(tok) + 18;
  const th = 26;

  // badge background
  noStroke();
  sf(C.layerFill, a * 0.7);
  rect(x - tw / 2, L.tokenY - th / 2, tw, th, 13);

  // badge border
  ss(C.layerStroke, a * 0.35);
  strokeWeight(1);
  noFill();
  rect(x - tw / 2, L.tokenY - th / 2, tw, th, 13);

  // text
  noStroke();
  sf(C.text, a * 0.95);
  textAlign(CENTER, CENTER);
  textSize(15);
  text(tok, x, L.tokenY);

  // arrow to model
  drawArrow(x, L.tokenY + th / 2 + 3, x, L.modelTop - 5, C.fwd, 1.5, a * 0.35);
}

// ============================================================
//  COMPONENT: PREDICTION
// ============================================================

function drawPrediction(posIdx, alpha, showLoss) {
  const x = getPosX(posIdx);
  const a = alpha !== undefined ? alpha : 255;

  // arrow from model bottom
  drawArrow(x, L.modelBot + 3, x, L.predY - 8, C.fwd, 1.5, a * 0.35);

  // distribution bars
  const barW = 7;
  const barGap = 2;
  const nBars = 5;
  const heights = [0.82, 0.38, 0.18, 0.09, 0.04];
  const maxH = 26;
  const baseY = L.predY + 6;
  const totalW = nBars * barW + (nBars - 1) * barGap;
  const sx = x - totalW / 2;

  for (let b = 0; b < nBars; b++) {
    const bh = heights[b] * maxH;
    const bx = sx + b * (barW + barGap);
    noStroke();
    sf(C.fwd, a * (b === 0 ? 0.85 : Math.max(0.08, 0.35 - b * 0.08)));
    rect(bx, baseY - bh, barW, bh, 2, 2, 0, 0);
  }

  // predicted token
  noStroke();
  sf(C.textDim, a * 0.8);
  textAlign(CENTER, TOP);
  textSize(12);
  textFont('monospace');
  text('"' + PREDS[posIdx] + '"', x, baseY + 5);

  // loss indicator
  if (showLoss) {
    // dashed target line at 100% probability
    drawingContext.save();
    ss(C.loss, a * 0.35);
    strokeWeight(1);
    drawingContext.setLineDash([3, 3]);
    line(sx, baseY - maxH, sx + totalW, baseY - maxH);
    drawingContext.setLineDash([]);
    drawingContext.restore();

    // shaded gap between top prediction and perfect
    noStroke();
    sf(C.loss, a * 0.12);
    const topBarH = heights[0] * maxH;
    rect(sx, baseY - maxH, barW, maxH - topBarH);

    // loss label
    sf(C.loss, a * 0.75);
    textAlign(LEFT, CENTER);
    textSize(9);
    textFont('monospace');
    text('loss', sx + totalW + 6, baseY - maxH * 0.85);
  }
}

// ============================================================
//  COMPONENT: K,V STACKS
// ============================================================

function drawKVStack(posIdx, progress, alpha, glowIntensity) {
  const x = getPosX(posIdx);
  const kvLeft = x + L.modelW / 2 + L.kvGap;
  const a = alpha !== undefined ? alpha : 255;
  const glow = glowIntensity || 0;

  for (let l = 0; l < NUM_LAYERS; l++) {
    // stagger: each layer appears slightly after the previous
    const lp = constrain((progress - l * 0.18) / 0.28, 0, 1);
    if (lp <= 0) continue;

    const ly = L.layerY[l];
    const la = a * easeOut(lp);

    // glow halo (behind block)
    if (glow > 0) {
      drawGlow(kvLeft, ly, L.kvW, L.layerH, C.kv, glow * easeOut(lp));
    }

    // shadow
    noStroke();
    sf([0, 0, 0], la * 0.15);
    rect(kvLeft + 1, ly + 1, L.kvW, L.layerH, 3);

    // main golden block
    sf(C.kv, la * 0.8);
    rect(kvLeft, ly, L.kvW, L.layerH, 3);

    // top highlight
    sf(C.kvBright, la * 0.3);
    rect(kvLeft, ly, L.kvW, L.layerH * 0.35, 3, 3, 0, 0);

    // label on first layer only
    if (l === 0 && lp > 0.5) {
      sf(C.kvBright, la * 0.65);
      textAlign(CENTER, BOTTOM);
      textSize(9);
      textFont('monospace');
      text('K,V', kvLeft + L.kvW / 2, ly - 3);
    }
  }

  // deposition arrows (model layer → KV) during animation
  if (progress > 0 && progress < 0.95) {
    for (let l = 0; l < NUM_LAYERS; l++) {
      const lp = constrain((progress - l * 0.18) / 0.28, 0, 1);
      if (lp > 0.15 && lp < 0.85) {
        const ly = L.layerY[l] + L.layerH / 2;
        const fromX = x + L.modelW / 2 + 2;
        const toX = kvLeft - 2;
        ss(C.kv, 75 * lp);
        strokeWeight(1);
        line(fromX, ly, toX, ly);
        noStroke();
        sf(C.kv, 75 * lp);
        triangle(toX, ly, toX - 4, ly - 2.5, toX - 4, ly + 2.5);
      }
    }
  }

  // persistent connectors (visible once KV is fully shown)
  if (progress >= 0.95) {
    for (let l = 0; l < NUM_LAYERS; l++) {
      const ly = L.layerY[l] + L.layerH / 2;
      const fromX = x + L.modelW / 2 + 2;
      const toX = kvLeft - 2;
      ss(C.kv, a * 0.18);
      strokeWeight(1);
      line(fromX, ly, toX, ly);
      noStroke();
      sf(C.kv, a * 0.25);
      circle(toX, ly, 3);
    }
  }
}

// ============================================================
//  COMPOSITE: ATTENTION ARROWS
// ============================================================

function drawAttentionForPos(targetPos, alpha) {
  if (targetPos <= 0) return;
  const a = alpha !== undefined ? alpha : 160;

  for (let src = 0; src < targetPos; src++) {
    const srcRight = getPosX(src) + L.modelW / 2 + L.kvGap + L.kvW + 2;
    const tgtLeft  = getPosX(targetPos) - L.modelW / 2 - 2;

    for (let l = 0; l < NUM_LAYERS; l++) {
      const ly   = L.layerY[l] + L.layerH / 2;
      const dist = targetPos - src;
      const fade = map(dist, 1, NUM_POS, 1, 0.3);
      const curve = -(8 + dist * 5);
      drawCurvedArrow(srcRight, ly, tgtLeft, ly, curve, C.attn, 1.2, a * fade);
    }
  }
}

// ============================================================
//  COMPOSITE: CONNECTOR GRADIENT (layer↔KV during backprop)
// ============================================================

function drawConnectorGradient(posIdx, intensity) {
  const x = getPosX(posIdx);
  const kvLeft = x + L.modelW / 2 + L.kvGap;
  for (let l = 0; l < NUM_LAYERS; l++) {
    const ly = L.layerY[l] + L.layerH / 2;
    const fromX = x + L.modelW / 2 + 2;
    const toX = kvLeft - 2;
    const pulse = 0.7 + 0.3 * sin(millis() * 0.004 + l * 0.5);
    ss(C.grad, intensity * pulse * 255);
    strokeWeight(1.5);
    line(fromX, ly, toX, ly);
  }
}

// ============================================================
//  COMPOSITE: SOURCE HIGHLIGHTING (generative path)
// ============================================================

function getLayerAtY(y) {
  for (let l = 0; l < NUM_LAYERS; l++) {
    if (y >= L.layerY[l] - 3 && y <= L.layerY[l] + L.layerH + 3) return l;
  }
  return -1;
}

function drawSourceHighlighting(posIdx, layerIdx, progress) {
  const x = getPosX(posIdx);
  const upstreamCount = layerIdx;

  // Phase 1: upstream layers at same position (teal glow, staggered upward)
  for (let l = layerIdx - 1; l >= 0; l--) {
    const fromTarget = layerIdx - l;
    const lt = constrain((progress - fromTarget * 0.06) / 0.2, 0, 1);
    if (lt <= 0) continue;
    drawGlow(x - L.modelW / 2, L.layerY[l], L.modelW, L.layerH, C.fwd, easeOut(lt) * 0.5);
  }

  // input token glow
  const tokenLt = constrain((progress - upstreamCount * 0.06) / 0.2, 0, 1);
  if (tokenLt > 0) {
    noStroke();
    textSize(15);
    textFont('monospace');
    const tw = textWidth(TOKENS[posIdx]) + 18;
    sf(C.fwd, 45 * easeOut(tokenLt));
    rect(x - tw / 2 - 4, L.tokenY - 17, tw + 8, 34, 16);
  }

  // Phase 2: KV blocks from earlier positions at this layer (gold glow, sweep left)
  const phase2Start = 0.3;
  for (let src = posIdx - 1; src >= 0; src--) {
    const dist = posIdx - src;
    const kt = constrain((progress - phase2Start - dist * 0.08) / 0.2, 0, 1);
    if (kt <= 0) continue;

    const srcX = getPosX(src);
    const kvLeft = srcX + L.modelW / 2 + L.kvGap;
    const ly = L.layerY[layerIdx];

    // glow on source KV block at this layer
    drawGlow(kvLeft, ly, L.kvW, L.layerH, C.kv, easeOut(kt) * 0.8);

    // thin line from source KV to target layer
    const tgtLeft = x - L.modelW / 2 - 2;
    const srcRight = kvLeft + L.kvW + 2;
    const layerMidY = ly + L.layerH / 2;
    ss(C.attn, 120 * easeOut(kt));
    strokeWeight(1);
    line(srcRight, layerMidY, tgtLeft, layerMidY);
  }

  // Phase 3: convergence diamond at (posIdx, layerIdx)
  const diamondT = constrain((progress - 0.6) / 0.25, 0, 1);
  if (diamondT > 0) {
    const ly = L.layerY[layerIdx] + L.layerH / 2;
    const pulse = 0.75 + 0.25 * sin(millis() * 0.005);
    const da = 255 * easeOut(diamondT) * pulse;
    const sz = 8;
    push();
    translate(x, ly);
    rotate(PI / 4);
    noStroke();
    sf(C.white, da * 0.15);
    rect(-sz * 1.5, -sz * 1.5, sz * 3, sz * 3, 2);
    sf(C.kvBright, da * 0.6);
    rect(-sz, -sz, sz * 2, sz * 2, 2);
    sf(C.white, da * 0.4);
    rect(-sz * 0.5, -sz * 0.5, sz, sz, 1);
    pop();
  }
}

// ============================================================
//  COMPOSITE: LOCAL GRADIENT (upward through layers)
// ============================================================

function drawLocalGradient(posIdx, progress, alpha) {
  const x = getPosX(posIdx);
  const a = alpha || 220;

  // gradient glows bottom-to-top: L4 first, L1 last
  for (let l = NUM_LAYERS - 1; l >= 0; l--) {
    const fromBot = NUM_LAYERS - 1 - l;
    const lt = constrain((progress - fromBot * 0.12) / 0.35, 0, 1);
    if (lt <= 0) continue;

    const ly = L.layerY[l];
    const la = a * easeOut(lt);

    drawGlow(x - L.modelW / 2, ly, L.modelW, L.layerH, C.grad, easeOut(lt) * 0.6);

    // connecting gradient line between layers
    if (l < NUM_LAYERS - 1) {
      ss(C.grad, la * 0.45);
      strokeWeight(2.5);
      line(x, L.layerY[l + 1], x, ly + L.layerH);
    }
  }

  // upward arrow from below the model (gradient arriving)
  if (progress > 0.2) {
    const aa = a * easeOut(constrain((progress - 0.2) / 0.3, 0, 1));
    drawArrow(x, L.modelBot + 22, x, L.modelBot + 3, C.grad, 2, aa * 0.5);
  }
}

// ============================================================
//  COMPOSITE: KV GRADIENT CASCADE (across positions)
// ============================================================

function drawKVGradient(sourcePos, targetPositions, progress, alpha) {
  const a = alpha || 200;

  for (let ti = 0; ti < targetPositions.length; ti++) {
    const tgtPos = targetPositions[ti];
    const dist   = sourcePos - tgtPos;
    const arrowT = constrain((progress - dist * 0.08) / 0.35, 0, 1);
    if (arrowT <= 0) continue;

    const la       = a * easeOut(arrowT);
    const srcX     = getPosX(sourcePos) - L.modelW / 2 - 2;
    const tgtRight = getPosX(tgtPos) + L.modelW / 2 + L.kvGap + L.kvW + 2;

    // gradient arrows (reversed attention path) at each layer
    for (let l = 0; l < NUM_LAYERS; l++) {
      const ly    = L.layerY[l] + L.layerH / 2;
      const curve = 8 + dist * 5;
      drawCurvedArrow(srcX, ly, tgtRight, ly, curve, C.grad, 1.5, la * 0.45);
    }

    // cascade: gradient enters target position's layers and through connectors
    const cascadeT = constrain((arrowT - 0.25) / 0.65, 0, 1);
    if (cascadeT > 0) {
      drawLocalGradient(tgtPos, cascadeT, la * 0.6);
      drawConnectorGradient(tgtPos, cascadeT * 0.5);
      const kvLeft = getPosX(tgtPos) + L.modelW / 2 + L.kvGap;
      for (let l = 0; l < NUM_LAYERS; l++) {
        drawGlow(kvLeft, L.layerY[l], L.kvW, L.layerH, C.kv, cascadeT * 0.5);
      }
    }
  }
}

// ============================================================
//  COMPOSITE: INFERENCE GLOW (pulsing KV)
// ============================================================

function drawInferenceGlow(progress) {
  for (let i = 0; i < NUM_POS; i++) {
    const x = getPosX(i);
    const kvLeft = x + L.modelW / 2 + L.kvGap;
    for (let l = 0; l < NUM_LAYERS; l++) {
      const ly = L.layerY[l];
      const pulse = 0.55 + 0.45 * sin(millis() * 0.002 + i * 1.2 + l * 0.8);
      noStroke();
      // outer glow
      sf(C.kv, 18 * progress * pulse);
      rect(kvLeft - 10, ly - 10, L.kvW + 20, L.layerH + 20, 8);
      // mid glow
      sf(C.kv, 35 * progress * pulse);
      rect(kvLeft - 5, ly - 5, L.kvW + 10, L.layerH + 10, 5);
      // inner bright
      sf(C.kvBright, 20 * progress * pulse);
      rect(kvLeft - 2, ly - 2, L.kvW + 4, L.layerH + 4, 4);
    }
  }
}

// ============================================================
//  FORWARD PASS PULSE (step 0 animation)
// ============================================================

function drawForwardPulse(posIdx, progress) {
  const x = getPosX(posIdx);
  for (let l = 0; l < NUM_LAYERS; l++) {
    const lt = constrain((progress - l * 0.15) / 0.25, 0, 1);
    if (lt <= 0 || lt >= 1) continue;
    const ly = L.layerY[l];
    const pulse = sin(lt * PI);
    noStroke();
    sf(C.fwd, 50 * pulse);
    rect(x - L.modelW / 2, ly, L.modelW, L.layerH, 5);
  }
}

// ============================================================
//  POSITION LABELS
// ============================================================

function drawPosLabels(alpha) {
  const a = alpha !== undefined ? alpha : 255;
  noStroke();
  sf(C.textMuted, a * 0.55);
  textAlign(CENTER, TOP);
  textSize(9);
  textFont('monospace');
  for (let i = 0; i < NUM_POS; i++) {
    text('pos ' + i, getPosX(i), L.predY + 40);
  }
}

// ============================================================
//  KV GLOW INTENSITY (for gradient steps)
// ============================================================

function getKVGlow(posIdx) {
  let count;
  if (currentStep === 7) {
    const seqT = easeInOut(animT);
    count = 1;
    if (seqT > 0.0)  count++;
    if (seqT > 0.25) count++;
    if (seqT > 0.50) count++;
  } else {
    count = 4;
  }
  let sources = 0;
  for (let fp = posIdx + 1; fp < NUM_POS; fp++) {
    if (fp - posIdx <= count) sources++;
  }
  return Math.min(1, sources * 0.25);
}

// ============================================================
//  STEP ORCHESTRATOR
// ============================================================

function drawStep() {
  const s = currentStep;
  const t = animT;

  drawBackground();

  // which positions are visible
  const visPos     = (s === 0) ? [0] : [0, 1, 2, 3, 4];
  const otherAlpha = (s === 1) ? 255 * easeOut(t) : 255;

  // position labels (unrolled view only)
  if (s >= 1) drawPosLabels(s === 1 ? otherAlpha : 255);

  // --- model columns, tokens, predictions ---
  for (const i of visPos) {
    const a = (i === 0 || s >= 2) ? 255 : otherAlpha;
    drawToken(i, a);
    drawModelColumn(i, a);
    const showLoss = (s >= 4 && s < 8 && i === NUM_POS - 1);
    drawPrediction(i, a, showLoss);
  }

  // --- forward pass pulse (step 0) ---
  if (s === 0) drawForwardPulse(0, t);

  // --- K,V stacks ---
  if (s === 2) {
    // animating appearance
    for (const i of visPos) drawKVStack(i, easeInOut(t), 255, 0);
  } else if (s >= 3) {
    // fully shown, with glow during gradient steps
    for (const i of visPos) {
      const glow = (s >= 6 && s < 8) ? getKVGlow(i) : 0;
      drawKVStack(i, 1, 255, glow);
    }
  }

  // --- attention / source highlighting ---
  if (s === 3) {
    if (selectedPos >= 0 && selectedLayer >= 0) {
      sourceAnimT = constrain((millis() / 1000 - sourceAnimStart) / 1.2, 0, 1);
      drawSourceHighlighting(selectedPos, selectedLayer, easeInOut(sourceAnimT));
    } else if (t > 0.55) {
      // gentle default hint on (pos 4, layer 2)
      const hintAlpha = map(t, 0.55, 1, 0, 0.6);
      drawSourceHighlighting(NUM_POS - 1, 2, hintAlpha);
    }
  } else if (s >= 4 && s < 8) {
    const showFor = selectedPos >= 0 ? selectedPos : NUM_POS - 1;
    drawAttentionForPos(showFor, 130);
  }

  // --- local gradient at last position (step 5) ---
  if (s === 5) {
    drawLocalGradient(NUM_POS - 1, easeInOut(t), 220);
  } else if (s >= 6 && s < 8) {
    drawLocalGradient(NUM_POS - 1, 1, 220);
  }

  // --- gradient cascade from last position (step 6) ---
  if (s === 6) {
    drawKVGradient(NUM_POS - 1, [3, 2, 1, 0], easeInOut(t), 200);
  } else if (s === 7) {
    drawKVGradient(NUM_POS - 1, [3, 2, 1, 0], 1, 180);
  }

  // --- gradient accumulation from other positions (step 7, auto-animated) ---
  if (s === 7) {
    const seqT = easeInOut(t);
    const src3T = constrain(seqT / 0.33, 0, 1);
    const src2T = constrain((seqT - 0.25) / 0.33, 0, 1);
    const src1T = constrain((seqT - 0.50) / 0.33, 0, 1);
    if (src3T > 0) drawKVGradient(3, [2, 1, 0], src3T, 120);
    if (src2T > 0) drawKVGradient(2, [1, 0], src2T, 85);
    if (src1T > 0) drawKVGradient(1, [0], src1T, 60);
  }

  // --- inference view: focus on "The" (step 8) ---
  if (s === 8) {
    const fadeIn = easeInOut(t);

    // dim positions 1-4
    for (let i = 1; i < NUM_POS; i++) {
      const px = getPosX(i);
      noStroke();
      sf(C.canvasBg, 160 * fadeIn);
      rect(px - L.modelW / 2 - 20, L.tokenY - 20,
           L.modelW + L.kvGap + L.kvW + 40,
           L.predY + 50 - L.tokenY + 20, 8);
    }

    // prominent glow on "The"'s KV blocks
    const theX = getPosX(0);
    const kvLeft = theX + L.modelW / 2 + L.kvGap;
    for (let l = 0; l < NUM_LAYERS; l++) {
      const ly = L.layerY[l];
      const pulse = 0.6 + 0.4 * sin(millis() * 0.002 + l * 0.8);
      const intensity = fadeIn * pulse;
      noStroke();
      sf(C.kv, 25 * intensity);
      rect(kvLeft - 14, ly - 14, L.kvW + 28, L.layerH + 28, 10);
      sf(C.kv, 45 * intensity);
      rect(kvLeft - 8, ly - 8, L.kvW + 16, L.layerH + 16, 7);
      sf(C.kvBright, 30 * intensity);
      rect(kvLeft - 3, ly - 3, L.kvW + 6, L.layerH + 6, 4);
    }

    // "many documents" converging indicators
    if (fadeIn > 0.4) {
      const docAlpha = easeOut(constrain((fadeIn - 0.4) / 0.4, 0, 1));
      const baseX = theX - L.modelW / 2 - 30;
      for (let d = 0; d < 4; d++) {
        const dy = L.modelTop + d * (L.modelH / 3);
        const fadeD = (0.6 - d * 0.12) * docAlpha;
        ss(C.kv, 50 * fadeD);
        strokeWeight(1);
        const sx = baseX - 15 - d * 8;
        const sy = dy - 10 + d * 5;
        line(sx, sy, baseX + 10, dy);
      }
    }
  }

  // --- hover highlight ---
  if (hoveredPos >= 0 && s >= 3 && s < 8) {
    const hx = getPosX(hoveredPos);
    noFill();
    ss(C.white, 22);
    strokeWeight(1);
    const bw = L.modelW + L.kvGap + L.kvW + 14;
    rect(hx - L.modelW / 2 - 7, L.modelTop - 7, bw, L.modelH + 14, 6);
  }
}

// ============================================================
//  NAVIGATION
// ============================================================

function goToNextStep() {
  if (currentStep >= TOTAL_STEPS - 1) return;
  currentStep++;
  animT = 0;
  animStartTime = millis() / 1000;
  animDuration = STEPS[currentStep].dur;
  selectedPos = -1;
  selectedLayer = -1;
  updateUI();
}

function goToPrevStep() {
  if (currentStep <= 0) return;
  currentStep--;
  animT = 1;
  selectedPos = -1;
  selectedLayer = -1;
  updateUI();
}

function updateUI() {
  document.getElementById('step-title').textContent = STEPS[currentStep].title;
  document.getElementById('step-text').innerHTML = STEPS[currentStep].html;

  document.getElementById('btn-prev').disabled = (currentStep === 0);
  document.getElementById('btn-next').disabled = (currentStep === TOTAL_STEPS - 1);

  document.querySelectorAll('.dot').forEach((d, i) => {
    d.className = 'dot';
    if (i === currentStep) d.classList.add('active');
    else if (i < currentStep) d.classList.add('visited');
  });

  document.getElementById('step-counter').textContent =
    (currentStep + 1) + ' / ' + TOTAL_STEPS;

}

// ============================================================
//  INTERACTION
// ============================================================

function mouseMoved() {
  if (currentStep < 3 || currentStep >= 8) { hoveredPos = -1; return; }
  hoveredPos = -1;
  for (let i = 0; i < NUM_POS; i++) {
    const x = getPosX(i);
    if (Math.abs(mouseX - x) < L.modelW * 0.8 &&
        mouseY > L.modelTop - 12 && mouseY < L.modelBot + 12) {
      hoveredPos = i;
      break;
    }
  }
  cursor(hoveredPos >= 0 ? HAND : ARROW);
}

function mousePressed() {
  if (hoveredPos >= 0 && currentStep >= 3 && currentStep < 8) {
    if (currentStep === 3) {
      const clickedLayer = getLayerAtY(mouseY);
      if (selectedPos === hoveredPos && selectedLayer === clickedLayer) {
        selectedPos = -1;
        selectedLayer = -1;
      } else {
        selectedPos = hoveredPos;
        selectedLayer = clickedLayer >= 0 ? clickedLayer : 0;
        sourceAnimT = 0;
        sourceAnimStart = millis() / 1000;
      }
    } else {
      selectedPos = (selectedPos === hoveredPos) ? -1 : hoveredPos;
    }
  }
}

// ============================================================
//  P5.JS LIFECYCLE
// ============================================================

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, drawHeight);
  canvas.parent(document.querySelector('main'));

  // progress dots
  const dotsEl = document.getElementById('progress-dots');
  for (let i = 0; i < TOTAL_STEPS; i++) {
    const d = document.createElement('div');
    d.className = 'dot' + (i === 0 ? ' active' : '');
    dotsEl.appendChild(d);
  }

  calcLayout();
  updateUI();

  // trigger step 0 animation
  animT = 0;
  animStartTime = millis() / 1000;
  animDuration = STEPS[0].dur;
}

function draw() {
  updateCanvasSize();

  if (animT < 1) {
    animT = constrain((millis() / 1000 - animStartTime) / animDuration, 0, 1);
  }

  drawStep();
}

function windowResized() {
  updateCanvasSize();
  calcLayout();
  resizeCanvas(canvasWidth, drawHeight);
}

function updateCanvasSize() {
  const el = document.querySelector('main');
  if (!el) return;
  canvasWidth = Math.floor(el.getBoundingClientRect().width);
  calcLayout();
}

function keyPressed() {
  if (keyCode === RIGHT_ARROW) goToNextStep();
  if (keyCode === LEFT_ARROW) goToPrevStep();
}
