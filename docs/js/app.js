import { Game, SIZE, CELLS, MINE, MINES } from "./game.js";
import { createBot } from "./bot.js";
import { MODEL_URL, LEVELS, TURN_MS, BOT_DELAY_MS } from "./config.js";

const HUMAN = 0, BOT = 1;
const LEVEL_NAMES = { easy: "Easy bot", medium: "Medium bot", hard: "Hard bot" };
const LEVEL_BLURBS = { easy: "Makes plenty of slips.", medium: "Plays like a careful person.", hard: "Reads every clue." };

const $ = (sel) => document.querySelector(sel);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const store = {
  get(key, fallback) { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode: settings just won't persist */ } },
};
const cellName = (i) => `${"abcdef"[i % SIZE]}${Math.floor(i / SIZE) + 1}`;

const MINE_SVG = (seat) => `
  <svg viewBox="0 0 40 40" aria-hidden="true" class="mine mine-${seat === HUMAN ? "you" : "bot"}">
    <g stroke="currentColor" stroke-width="3.2" stroke-linecap="round">
      <path d="M20 4v7M20 29v7M4 20h7M29 20h7M8.7 8.7l5 5M26.3 26.3l5 5M31.3 8.7l-5 5M13.7 26.3l-5 5"/>
    </g>
    <circle cx="20" cy="20" r="11" fill="currentColor"/>
    ${seat === HUMAN ? '<circle cx="16.5" cy="16.5" r="3" fill="#fff" opacity=".4"/>'
                     : '<circle cx="20" cy="20" r="6.2" fill="none" stroke="var(--shore)" stroke-width="2.4"/>'}
  </svg>`;

const state = {
  game: new Game(),
  bot: null,
  level: store.get("fmm.level", "medium"),
  timerOn: store.get("fmm.timer", "1") === "1",
  token: 0,            // bumps on every new game so a bot move that was in flight can tell it is stale
  turnStart: 0,
  clock: null,
  status: "",
  timeoutNote: false,     // your clock ran out last turn
  botOutOfTime: false,    // the bot's turn ended because its next move would not fit in 10 s
  botMoves: 0,            // moves the bot has made in its current turn
};
if (!LEVEL_NAMES[state.level]) state.level = "medium";

// ---------- build the page ----------
const cells = [];
function buildBoard() {
  $(".axis-cols").innerHTML = [..."abcdef"].map((l) => `<span>${l}</span>`).join("");
  $(".axis-rows").innerHTML = Array.from({ length: SIZE }, (_, i) => `<span>${i + 1}</span>`).join("");
  const grid = $("#grid");
  for (let i = 0; i < CELLS; i++) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "cell";
    b.dataset.idx = i;
    b.tabIndex = i === 0 ? 0 : -1;
    b.addEventListener("click", () => humanPick(i));
    grid.append(b);
    cells.push(b);
  }
  grid.addEventListener("keydown", (e) => {
    const i = Number(document.activeElement?.dataset?.idx);
    const moves = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -SIZE, ArrowDown: SIZE };
    if (Number.isNaN(i) || !(e.key in moves)) return;
    const col = i % SIZE, d = moves[e.key];
    if ((e.key === "ArrowLeft" && col === 0) || (e.key === "ArrowRight" && col === SIZE - 1)) return;
    const next = i + d;
    if (next < 0 || next >= CELLS) return;
    e.preventDefault();
    cells[i].tabIndex = -1;
    cells[next].tabIndex = 0;
    cells[next].focus();
  });
}

// ---------- drawing ----------
function setStatus(text) {
  state.status = text;
  $("#status").textContent = text || " ";
}

function render() {
  const g = state.game, human = g.current === HUMAN && !g.done;
  const canPlay = human && state.bot && !state.busy;
  cells.forEach((el, i) => {
    const v = g.revealed.get(i);
    const last = g.lastMove?.idx === i;
    let cls = "cell", label = `${cellName(i)}, covered`, html = "";
    if (v === undefined) {
      if (g.done && g.mines.has(i)) { cls += " missed"; html = MINE_SVG(-1).replace("mine-bot", "mine-missed"); label = `${cellName(i)}, mine nobody found`; }
      else cls += canPlay ? " open" : "";
    } else if (v === MINE) {
      const seat = g.owner.get(i);
      cls += ` shown mined mined-${seat === HUMAN ? "you" : "bot"}`;
      html = MINE_SVG(seat);
      label = `${cellName(i)}, mine found by ${seat === HUMAN ? "you" : "the bot"}`;
    } else {
      cls += ` shown depth-${Math.min(v, 5)}`;
      html = `<span class="sounding">${v}</span>`;
      label = `${cellName(i)}, ${v} ${v === 1 ? "mine" : "mines"} next to it`;
    }
    if (last) cls += g.lastMove.seat === HUMAN ? " last last-you" : " last last-bot";
    el.className = cls;
    el.innerHTML = html;
    el.setAttribute("aria-label", label);
    el.disabled = v !== undefined || !canPlay;
    el.setAttribute("aria-disabled", String(el.disabled));
  });
  $("#score-you").textContent = g.scores[HUMAN];
  $("#score-bot").textContent = g.scores[BOT];
  $("#bot-name").textContent = LEVEL_NAMES[state.level];
  $("#level-blurb").textContent = LEVEL_BLURBS[state.level];
  $("#card-you").classList.toggle("turn", g.current === HUMAN && !g.done);
  $("#card-bot").classList.toggle("turn", g.current === BOT && !g.done);
  $("#timer-toggle").checked = state.timerOn;
  document.querySelectorAll('input[name="level"]').forEach((r) => { r.checked = r.value === state.level; });
  const res = $("#result");
  res.hidden = !g.done;
  $("#rematch").hidden = !g.done;
  $(".toggle").hidden = g.done;
  if (g.done) {
    const you = g.winner === HUMAN;
    $("#result-line").textContent = `${you ? "You win" : "The bot wins"}, ${g.scores[g.winner]} to ${g.scores[1 - g.winner]}.`;
    res.classList.toggle("won", you);
    res.classList.toggle("lost", !you);
  }
}

// ---------- the turn clock (10 s per turn, not reset by finding a mine) ----------
function stopClock() {
  clearInterval(state.clock);
  state.clock = null;
  const bar = $("#clock-bar");
  bar.style.transform = "scaleX(0)";
  bar.parentElement.classList.remove("running", "low", "bot");
}
function startClock(seat = HUMAN) {
  stopClock();
  if (seat === HUMAN && !state.timerOn) return; // your clock is optional; the bot's is part of its difficulty
  state.turnStart = performance.now();
  const bar = $("#clock-bar"), box = bar.parentElement;
  box.classList.add("running");
  box.classList.toggle("bot", seat === BOT);
  const tick = () => {
    const left = Math.max(0, TURN_MS - (performance.now() - state.turnStart));
    bar.style.transform = `scaleX(${left / TURN_MS})`;
    box.classList.toggle("low", seat === HUMAN && left < 3000);
    if (left <= 0 && seat === HUMAN) timeUp();
  };
  tick();
  state.clock = setInterval(tick, 100);
}
function timeUp() {
  stopClock();
  if (state.game.current !== HUMAN || state.game.done) return;
  state.game.passTurn();
  state.timeoutNote = true;
  beginTurn();
}

// ---------- turn flow ----------
function beginTurn() {
  render();
  const g = state.game;
  if (g.done) return finish();
  if (g.current === HUMAN) {
    setStatus(state.timeoutNote ? "Time ran out on your last turn. Your move."
            : state.botOutOfTime ? "The bot ran out of time. Your move." : "Your move.");
    state.timeoutNote = state.botOutOfTime = false;
    startClock(HUMAN);
  } else {
    state.botMoves = 0;
    startClock(BOT);
    botTurn();
  }
}

function finish() {
  stopClock();
  setStatus("");
  render();
  $("#rematch").focus({ preventScroll: true });
}

function humanPick(i) {
  const g = state.game;
  if (g.current !== HUMAN || g.done || !state.bot || state.busy) return;
  const r = g.pick(i);
  if (!r.ok) return;
  if (r.done) return finish();
  if (r.isMine) { setStatus("Mine. Go again."); render(); }
  else { stopClock(); setStatus(""); beginTurn(); }
}

async function botTurn() {
  const g = state.game, delay = BOT_DELAY_MS[state.level];
  // The budget counts only the delay per move, not inference time (that happens inside the delay).
  // A move is allowed if the total is TURN_MS or less, so one landing exactly on 10 s still counts.
  if ((state.botMoves + 1) * delay > TURN_MS) {
    state.botOutOfTime = true;
    g.passTurn();
    return beginTurn();
  }
  const token = ++state.token;
  state.busy = true;
  render();
  setStatus(state.timeoutNote ? "Time ran out. The bot is thinking…" : "The bot is thinking…");
  state.timeoutNote = false;
  let move;
  try {
    [move] = await Promise.all([state.bot.choose(g.boardView(), state.level), sleep(delay)]);
  } catch (err) {
    state.busy = false;
    console.error(err);
    return setStatus("The bot could not move. Reload the page to try again.");
  }
  if (token !== state.token) return; // a new game started while the bot was thinking
  state.busy = false;
  state.botMoves++;
  const r = g.pick(move);
  if (r.done) return finish();
  if (r.isMine) {
    setStatus("The bot found a mine and goes again.");
    render();
    return botTurn();
  }
  beginTurn();
}

// ---------- starting games ----------
function newGame(first) {
  state.token++;
  state.busy = false;
  stopClock();
  state.game.reset(first);
  state.timeoutNote = state.botOutOfTime = false;
  state.botMoves = 0;
  beginTurn();
}
const randomSeat = () => (Math.random() < 0.5 ? HUMAN : BOT);

function wire() {
  $("#new-game").addEventListener("click", () => newGame(randomSeat()));
  $("#rematch").addEventListener("click", () => newGame(state.game.winner ?? randomSeat()));
  $("#timer-toggle").addEventListener("change", (e) => {
    state.timerOn = e.target.checked;
    store.set("fmm.timer", state.timerOn ? "1" : "0");
    if (state.game.current === HUMAN && !state.game.done && state.bot) startClock(); else stopClock();
  });
  document.querySelectorAll('input[name="level"]').forEach((r) =>
    r.addEventListener("change", () => {
      state.level = r.value;
      store.set("fmm.level", state.level);
      newGame(randomSeat());
    }));
}

// ---------- boot ----------
async function boot() {
  buildBoard();
  wire();
  render();
  const loading = $("#loading");
  try {
    if (!window.ort) throw new Error("ONNX Runtime Web did not load (check your connection).");
    window.ort.env.wasm.numThreads = 1;
    state.bot = await createBot(window.ort, MODEL_URL, LEVELS);
    loading.hidden = true;
    document.body.classList.add("ready");
    newGame(randomSeat());
  } catch (err) {
    console.error(err);
    loading.classList.add("error");
    loading.textContent = location.protocol === "file:"
      ? "Browsers block loading the model from a file. Run `python -m http.server` in this folder and open http://localhost:8000."
      : `Could not load the model: ${err.message}`;
  }
}
const THEME_COLORS = { light: "#f4f7fb", dark: "#0b1016" };
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#theme-toggle").setAttribute("aria-pressed", theme === "dark");
  $("#theme-toggle").textContent = theme === "dark" ? "Light mode" : "Dark mode";
  $('meta[name="theme-color"]').content = THEME_COLORS[theme];
}
applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
$("#theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  store.set("fmm.theme", next);
  applyTheme(next);
});

boot();

// a handle for tests and tinkering in the console
window.fmm = { state, humanPick, newGame, cellName };
