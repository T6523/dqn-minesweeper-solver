// Find My Mines rules. Mirrors game.py: a bomb lets you go again, an empty cell shows its
// neighbour count and passes the turn, the match ends when every bomb has been found.
export const SIZE = 6;
export const CELLS = SIZE * SIZE;
export const MINES = 11;
export const MINE = "bomb"; // what a revealed mine looks like in a board view

export const NEIGHBOURS = Array.from({ length: CELLS }, (_, i) => {
  const r = Math.floor(i / SIZE), c = i % SIZE, out = [];
  for (let dr = -1; dr <= 1; dr++)
    for (let dc = -1; dc <= 1; dc++) {
      const rr = r + dr, cc = c + dc;
      if ((dr || dc) && rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE) out.push(rr * SIZE + cc);
    }
  return out;
});

export class Game {
  constructor(rng = Math.random) {
    this.rng = rng;
    this.reset(0);
  }

  /** Deal a new board. `first` is the seat (0 or 1) that moves first. */
  reset(first = 0) {
    const cells = [...Array(CELLS).keys()];
    for (let i = CELLS - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [cells[i], cells[j]] = [cells[j], cells[i]];
    }
    this.mines = new Set(cells.slice(0, MINES));
    this.adjacent = NEIGHBOURS.map((ns) => ns.filter((n) => this.mines.has(n)).length);
    this.revealed = new Map(); // cell -> MINE or a count 0-8
    this.owner = new Map();    // found mine -> seat that found it
    this.scores = [0, 0];
    this.current = first;
    this.lastMove = null;      // { idx, seat }
  }

  get done() { return this.scores[0] + this.scores[1] >= MINES; }
  get winner() { return this.done ? (this.scores[0] > this.scores[1] ? 0 : 1) : null; } // 11 is odd: no ties

  pick(idx) {
    if (this.done || !(idx >= 0 && idx < CELLS) || this.revealed.has(idx)) return { ok: false };
    const seat = this.current, isMine = this.mines.has(idx);
    this.revealed.set(idx, isMine ? MINE : this.adjacent[idx]);
    this.lastMove = { idx, seat };
    if (isMine) {
      this.scores[seat]++;
      this.owner.set(idx, seat);
    } else {
      this.passTurn();
    }
    return { ok: true, seat, isMine, done: this.done, turnChanged: this.current !== seat };
  }

  passTurn() { if (!this.done) this.current = 1 - this.current; }

  /** All a player (or the bot) may see: null = hidden, "bomb" = found mine, 0-8 = clue. */
  boardView() {
    return Array.from({ length: SIZE }, (_, r) =>
      Array.from({ length: SIZE }, (_, c) => this.revealed.get(r * SIZE + c) ?? null));
  }
}
