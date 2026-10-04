// Node checks for the JavaScript port (not needed to run the site):
//   1. encodeBoard reproduces the Python observation / mask for every vector in test_vectors.json
//   2. the ONNX model gives the same values and the same greedy move
//   3. the rules behave (turn passing, scoring, winner) over many random games
//   4. the bot at each level averages the turns it was calibrated to
// Run:  python test.py vectors   (writes tests/test_vectors.json)
//       npm install onnxruntime-web && node docs/tests/parity.mjs
//       (or ORT=/path/to/onnxruntime-web/dist/ort.node.min.mjs node docs/tests/parity.mjs)
import fs from "node:fs";
import { pathToFileURL } from "node:url";
import { Game, SIZE, CELLS, MINES } from "../js/game.js";
import { encodeBoard, createBot } from "../js/bot.js";
import { MODEL_URL, LEVELS } from "../js/config.js";

const ort = await import(process.env.ORT ? pathToFileURL(process.env.ORT).href : "onnxruntime-web");
const dir = new URL("..", import.meta.url).pathname;
const vectors = JSON.parse(fs.readFileSync(process.env.VECTORS ?? new URL("test_vectors.json", import.meta.url).pathname)).vectors;
const bot = await createBot(ort.default ?? ort, fs.readFileSync(dir + MODEL_URL), LEVELS);
let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? "pass" : "FAIL"}  ${msg}`); if (!ok) failures++; };

// 1 + 2: Python test vectors
let obsDiff = 0, maskOk = true, qDiff = 0, moveOk = 0;
for (const v of vectors) {
  const { obs, mask } = encodeBoard(v.board);
  const flat = v.observation.flat(2);
  obs.forEach((x, i) => { obsDiff = Math.max(obsDiff, Math.abs(x - flat[i])); });
  maskOk &&= mask.every((m, i) => m === v.legal_mask[i]);
  const { q } = await bot.qValues(v.board);
  q.forEach((x, i) => { qDiff = Math.max(qDiff, Math.abs(x - v.q_values[i])); });
  const best = Math.max(...mask.map((m, i) => (m ? q[i] : -Infinity)));
  moveOk += q[v.greedy_move] >= best - 1e-5;
}
check(obsDiff < 1e-6, `observation matches Python on ${vectors.length} boards (max diff ${obsDiff.toExponential(1)})`);
check(maskOk, "legal mask matches Python");
check(qDiff < 1e-4, `model values match Python (max diff ${qDiff.toExponential(1)})`);
check(moveOk === vectors.length, `Python's greedy move is still optimal in JS on ${moveOk}/${vectors.length} boards`);

// 3: rules
{
  let ok = true;
  for (let g = 0; g < 500 && ok; g++) {
    const game = new Game(), first = g % 2;
    game.reset(first);
    let guard = 0;
    ok &&= game.mines.size === MINES && game.current === first;
    while (!game.done && guard++ < 100) {
      const seat = game.current, free = [...Array(CELLS).keys()].filter((i) => !game.revealed.has(i));
      const before = game.scores[seat], r = game.pick(free[Math.floor(Math.random() * free.length)]);
      ok &&= r.ok && (r.isMine ? game.scores[seat] === before + 1 && (game.done || game.current === seat)
                              : game.scores[seat] === before && game.current !== seat);
      ok &&= !game.pick([...game.revealed.keys()][0]).ok; // re-picking is refused
    }
    ok &&= game.done && game.scores[0] + game.scores[1] === MINES && game.winner === (game.scores[0] > game.scores[1] ? 0 : 1);
    const view = game.boardView();
    ok &&= view.flat().filter((c) => c === "bomb").length === MINES && view.length === SIZE;
  }
  check(ok, "rules: scoring, turn passing, refusing repeats, winner (500 random games)");
}

// 4: bot strength per level (solo, fresh random boards): the turns each level was calibrated to, give or take noise
const expected = { easy: 17.5, medium: 12.5, hard: 10.3 };
for (const level of ["easy", "medium", "hard"]) {
  const N = Number(process.env.GAMES ?? 150);
  let turns = 0;
  for (let g = 0; g < N; g++) {
    const game = new Game();
    let picks = 0;
    while (!game.done) { game.pick(await bot.choose(game.boardView(), level)); picks++; }
    turns += picks - MINES;
  }
  const avg = turns / N;
  check(Math.abs(avg - expected[level]) < 1.0, `bot ${level} (T=${LEVELS[level]}): ${avg.toFixed(2)} turns over ${N} games (calibrated to about ${expected[level]})`);
}
console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
