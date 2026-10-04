// Settings you may want to change. Everything else in js/ is game logic.

// The exported network (see export_onnx.py). The path is relative to index.html.
export const MODEL_URL = "ddqn.onnx";

// Difficulty. The bot gets a value for every covered cell and samples a cell with probability proportional
// to exp(value / T). T = 0 always takes the best cell; a larger T makes more slips.
// These were calibrated for this particular model so that, playing alone on a fresh board, each level averages
// about this many turns (a bomb lets you go again, so only empty picks count):
//   easy  T = 0.526  about 17.5 turns     medium  T = 0.172  about 12.5 turns     hard  T = 0  about 10.3 turns
// To make a level easier, raise its T. If you export a new model, calibrate again: the right T depends on the model.
export const LEVELS = { easy: 0.526, medium: 0.172, hard: 0 };

// Rules from the assignment: 10 seconds per turn (the player can switch the clock off on the page).
export const TURN_MS = 10000;

// How long the bot takes over each move, in milliseconds. A bot turn is also limited to TURN_MS, like yours, so this sets
// a hard ceiling on how many moves it can chain after finding mines: floor(TURN_MS / delay).
// A move is allowed if the turn's delays add up to TURN_MS or less (a move landing exactly on 10 s counts).
// Only this delay is counted, not the time the model takes to think: that runs inside the delay.
// With 10 seconds: easy 2.5 s gives at most 4 moves per turn, medium 2 s gives 5, hard 1.5 s gives 6.
export const BOT_DELAY_MS = { easy: 2500, medium: 2000, hard: 1500 };
