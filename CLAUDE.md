# llm-backprop

Interactive pedagogical visualizations explaining how LLM transformers work internally, with emphasis on KV activations and gradient flow.

## Project Structure

```
kv-activations/
  index.html      # HTML shell: 4 canvas sections with play/reset controls
  factory.js      # All visualization logic, raw Canvas 2D (no libraries)
```

No build step. Serve with any static HTTP server (e.g. `python -m http.server 8766`) and open `kv-activations/index.html`.

## Architecture

`factory.js` is a single-file Canvas 2D animation engine. Key concepts:

- **`createSection(el, drawFn, duration)`** — Binds a draw function to a canvas with wall-clock timing via `requestAnimationFrame`. Handles play/pause/reset. Supports seek via `canvas.parentElement.parentElement.dataset.seek` for testing.
- **4 draw functions** (`drawSec1`–`drawSec4`), each receives `(ctx, W, H, t)` where `t` is normalized 0..1 progress through the animation.
- **`pop(t, activation, riseMs, fallMs)`** — Glow flash curve: quick rise, slow exponential fall. Used everywhere for activation highlights.
- **`glowLineDir(…, progress)`** — Directional glow: draws a partial line from source toward destination, clipped at `progress` (0..1). Creates traveling-signal effect.
- **Additive glow** via `globalCompositeOperation: 'lighter'` throughout.

## The 4 Sections

1. **Autoregressive generation** — Single model column, tokens fed one at a time, predictions fly up to become next input
2. **Unrolled view** — Same process shown as N parallel columns (one per position), forward pass pulses downward through each
3. **KV cache** — KV boxes materialize at each layer, supply rails show attention reads from earlier positions, directional glow emphasizes information flow
4. **Backpropagation** — Diagonal gradient wavefront propagates from loss (bottom-right) upward and leftward through layers and KV nodes

## Configuring the Token Sequence

Change `SEQUENCE` at the top of `factory.js`:

```js
const SEQUENCE = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
```

Everything (column count, spacing, durations, labels) derives from this array. `SEQUENCE[0]` is the prompt; each subsequent token is a prediction target.

## Color Palette

| Role | Hex |
|------|-----|
| Forward pass / stations | `#00d4ff` (cyan) |
| KV storage | `#ffb020` (amber) |
| Attention reads | `#40ff90` (green) |
| Gradient / backprop | `#ff3080` (magenta) |
| Loss | `#ff4060` (red) |

## Dev Notes

- No dependencies. No build. Just HTML + JS.
- Animations use wall-clock `performance.now()`, not frame counting. Playwright's accelerated clock makes animations complete near-instantly — use the `dataset.seek` mechanism for frame-precise testing.
- `L = 4` (layer count) is a constant alongside the token sequence. Changing it should work but is less tested than changing the token sequence.
