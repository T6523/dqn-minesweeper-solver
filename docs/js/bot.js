// The bot: board view -> 7 feature channels -> ONNX model -> a cell, chosen by difficulty.
// encodeBoard must match the training code exactly (see test_vectors.json and tests/parity.mjs).
import { SIZE, CELLS, MINES, MINE, NEIGHBOURS } from "./game.js";

const CHANNELS = 7;

const neighbourSum = (a) => Float64Array.from(a, (_, i) => NEIGHBOURS[i].reduce((s, n) => s + a[n], 0));

/** board view -> { obs: Float32Array(7*36) in channel-major order, mask: bool[36] (true = hidden = legal) } */
export function encodeBoard(view) {
  const hidden = new Float64Array(CELLS), bomb = new Float64Array(CELLS), clue = new Float64Array(CELLS);
  view.forEach((row, r) => row.forEach((cell, c) => {
    const i = r * SIZE + c;
    if (cell === null) hidden[i] = 1;
    else if (cell === MINE) bomb[i] = 1;
    else clue[i] = cell;
  }));
  const bombNear = neighbourSum(bomb), hiddenNear = neighbourSum(hidden);
  const hiddenCount = hidden.reduce((s, v) => s + v, 0), bombCount = bomb.reduce((s, v) => s + v, 0);
  const density = (MINES - bombCount) / Math.max(hiddenCount, 1);
  const obs = new Float32Array(CHANNELS * CELLS);
  for (let i = 0; i < CELLS; i++) {
    const shown = (1 - hidden[i]) * (1 - bomb[i]);
    const remaining = Math.max(clue[i] - bombNear[i], 0) * shown; // bombs this clue still has to account for
    const ratio = hiddenNear[i] > 0 ? Math.min(remaining / Math.max(hiddenNear[i], 1), 1) : 0;
    obs[0 * CELLS + i] = hidden[i];
    obs[1 * CELLS + i] = bomb[i];
    obs[2 * CELLS + i] = clue[i] / 8;
    obs[3 * CELLS + i] = remaining / 8;
    obs[4 * CELLS + i] = hiddenNear[i] / 8;
    obs[5 * CELLS + i] = ratio;
    obs[6 * CELLS + i] = density;
  }
  return { obs, mask: Array.from(hidden, (v) => v === 1) };
}

/** Choose among legal cells. T = 0: the best cell (ties broken at random). T > 0: sample softmax(q / T). */
export function pickCell(q, mask, T, rng = Math.random) {
  const legal = [];
  mask.forEach((ok, i) => ok && legal.push(i));
  if (T <= 0) {
    const best = Math.max(...legal.map((i) => q[i]));
    const ties = legal.filter((i) => q[i] >= best - 1e-5);
    return ties[Math.floor(rng() * ties.length)];
  }
  const top = Math.max(...legal.map((i) => q[i]));
  const w = legal.map((i) => Math.exp((q[i] - top) / T));
  let r = rng() * w.reduce((s, v) => s + v, 0);
  for (let k = 0; k < legal.length; k++) { r -= w[k]; if (r <= 0) return legal[k]; }
  return legal[legal.length - 1];
}

/** `ort` is onnxruntime-web, `model` a URL or bytes, `levels` maps level name -> temperature. */
export async function createBot(ort, model, levels, rng = Math.random) {
  const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  async function qValues(view) {
    const { obs, mask } = encodeBoard(view);
    const out = await session.run({ observation: new ort.Tensor("float32", obs, [1, CHANNELS, SIZE, SIZE]) });
    return { q: out.q_values.data, mask };
  }
  return {
    levels,
    qValues,
    async choose(view, level) {
      const { q, mask } = await qValues(view);
      return pickCell(q, mask, levels[level], rng);
    },
  };
}
