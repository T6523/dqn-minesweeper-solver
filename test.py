"""Experiments behind the README: ablation study, reference players, difficulty levels.

    python test.py train ab_final,ab_double --seeds 0,1,2 --workers 2   train variants; per-board results go to results/
    python test.py report                                               ablation tables with 95% intervals
    python test.py deployed [checkpoint]                                the deployed model on fresh boards
    python test.py levels [model.onnx]                                  the three difficulty levels, solo and head to head
    python test.py calibrate [checkpoint]                               temperatures that give Easy and Medium their target strength
    python test.py vectors                                              test data for docs/tests/parity.mjs

Score: "turns" = picks that landed on an empty cell before all mines were found, alone on a fresh board (lower is better).
Keep --workers at 2 or fewer on a single GPU.
"""
import concurrent.futures
import glob
import json
import multiprocessing
import os
import random
import re
import sys
from math import comb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn import BOMB_COUNT, CELL_COUNT, GRID_SIZE, NEGATIVE_INF, ReplayBuffer, Symmetric, Transition, environment, select_action
from export_onnx import Wrapper, load_model
from game import BOMB, Game

# Pinned here so the study does not change when dqn.py is edited.
GAMMA, LEARNING_RATE, BATCH_SIZE, BUFFER_CAPACITY = 0.3, 1e-3, 64, 20000
LEARNING_STARTS, SYNC_EVERY, CLIP_NORM = 1000, 500, 10
STEPS, EPSILON_END = 30000, 0.05          # epsilon falls linearly from 1 to EPSILON_END over the first third of the run
EVAL_SEED, EVAL_GAMES = 12345, 2000       # fixed evaluation boards: seeds EVAL_SEED + i
FRESH_SEED = 8675309                      # boards that nothing was tuned on
RESULTS_DIR = "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False    # True reserves large scratch memory each time the batch size changes
if DEVICE.type == "cuda":                 # hard per-process cap: a mistake kills one process, not the GPU
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRACTION", "0.25")))


# ---- models and environments ------------------------------------------------------------------------------------
class FullyConvDQN(nn.Module):
    """Padded 3x3 conv stack and a 1x1 head: one rule shared by every cell."""

    def __init__(self, input_channels, channels=64, layers=4):
        super().__init__()
        blocks, c = [], input_channels
        for _ in range(layers):
            blocks += [nn.Conv2d(c, channels, 3, padding=1), nn.ReLU()]
            c = channels
        self.network = nn.Sequential(*blocks, nn.Conv2d(c, 1, 1), nn.Flatten())

    def forward(self, x):
        return self.network(x)


class LegacyDQN(nn.Module):
    """Two convolutions, then flatten and fully connected layers: a separate set of weights for every output cell."""

    def __init__(self, input_channels, cells=CELL_COUNT):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * cells, 128), nn.ReLU(), nn.Linear(128, cells))

    def forward(self, x):
        return self.network(x)


class ThreeChannelEnv(environment):
    """Only the three raw channels (covered, found, clue / 8): no engineered features."""

    def observe(self):
        obs, mask = super().observe()
        return obs[:3].copy(), mask


def variant(env=environment, net=lambda: FullyConvDQN(7), double=True, aug=False):
    return dict(env=env, net=net, double=double, aug=aug)


# Each version is trained for STEPS steps with the settings above.
VARIANTS = {
    # ingredients added one at a time
    "ab_raw":         variant(env=ThreeChannelEnv, net=lambda: LegacyDQN(3), double=False),
    "ab_features":    variant(net=lambda: LegacyDQN(7), double=False),
    "ab_fcn":         variant(double=False),
    "ab_double":      variant(),
    "ab_final":       variant(aug=True),
    # one ingredient left out of the final recipe (without augmentation is ab_double)
    "ab_no_double":   variant(double=False, aug=True),
    "ab_no_features": variant(env=ThreeChannelEnv, net=lambda: FullyConvDQN(3), aug=True),
    "ab_no_fcn":      variant(net=lambda: LegacyDQN(7), aug=True),
}


# ---- training ---------------------------------------------------------------------------------------------------
class Learner:
    """One DQN update (Double DQN or plain). On CUDA the whole step is captured as a CUDA graph, which removes
    launch overhead and gives the same result."""

    def __init__(self, main, target, optimizer, double):
        self.main, self.target, self.optimizer, self.double, self.graph = main, target, optimizer, double, None

    def _step(self):
        obs, action, reward, next_obs, next_mask, done = self.batch
        n = obs.shape[0]
        q_all = self.main(torch.cat([obs, next_obs]))               # current and next boards in one pass
        q = q_all[:n].gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            target_q = self.target(next_obs)
            if self.double:                                         # main network picks the legal action, target scores it
                best = torch.where(next_mask, q_all[n:], NEGATIVE_INF).argmax(1, keepdim=True)
                value = target_q.gather(1, best).squeeze(1)
            else:
                value = torch.where(next_mask, target_q, NEGATIVE_INF).max(1).values
            y = reward + GAMMA * value * (1 - done)
        loss = F.smooth_l1_loss(q, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main.parameters(), CLIP_NORM, foreach=True)
        self.optimizer.step()
        return loss.detach()

    def _capture(self):
        saved = {k: v.clone() for k, v in self.main.state_dict().items()}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step()                                        # warm-up creates the Adam state the graph reuses
        torch.cuda.current_stream().wait_stream(side)
        self.main.load_state_dict(saved)                            # undo the warm-up steps
        for state in self.optimizer.state.values():
            for v in state.values():
                v.zero_()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = self._step()

    def update(self, batch):
        if DEVICE.type != "cuda":
            self.batch = batch
            return self._step()
        if self.graph is None:
            self.batch = [t.to(DEVICE) for t in batch]
            self._capture()
        for dst, src in zip(self.batch, batch):
            dst.copy_(src)
        self.graph.replay()
        return self.loss


class GraphedActor:
    """Epsilon-greedy action for one board, with the greedy forward pass as a CUDA graph."""

    def __init__(self, network):
        self.network, self.graph = network, None

    def _capture(self, obs, mask):
        self.obs = torch.from_numpy(obs).to(DEVICE)[None].clone()
        self.mask = torch.from_numpy(mask).to(DEVICE)[None].clone()
        with torch.no_grad():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.network(self.obs)
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.action = torch.where(self.mask, self.network(self.obs), NEGATIVE_INF).argmax(1)

    def act(self, obs, mask, epsilon):
        if random.random() < epsilon:
            return random.choice(np.flatnonzero(mask).tolist())
        if self.graph is None:
            self._capture(obs, mask)
        self.obs.copy_(torch.from_numpy(obs)[None])
        self.mask.copy_(torch.from_numpy(mask)[None])
        self.graph.replay()
        return int(self.action.item())


def turns_used(env):
    return env.step_count - sum(env.game.scores)


def play_lockstep(envs, policy):
    """Play every env to the end together, one batched policy call per move."""
    state = [env.reset() for env in envs]
    active = list(range(len(envs)))
    while active:
        actions = policy(np.stack([state[i][0] for i in active]), np.stack([state[i][1] for i in active]))
        still = []
        for i, action in zip(active, actions):
            state[i] = envs[i].step(action)[:2]
            if not envs[i].game.done:
                still.append(i)
        active = still


def batched_greedy(model, device):
    def policy(obs, mask):
        with torch.no_grad():
            q = model(torch.from_numpy(obs).to(device))
            return torch.where(torch.from_numpy(mask).to(device), q, NEGATIVE_INF).argmax(1).tolist()
    return policy


def per_game_turns(model, env_cls, device, games=EVAL_GAMES, seed=EVAL_SEED, chunk=250):
    """Turns of a greedy model on `games` fixed boards (seeds seed + i). Chunked to keep GPU memory small."""
    out = []
    for start in range(0, games, chunk):
        envs = [env_cls() for _ in range(min(chunk, games - start))]
        for i, env in enumerate(envs):
            env.game.rng = random.Random(seed + start + i)
        play_lockstep(envs, batched_greedy(model, device))
        out += [turns_used(env) for env in envs]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.array(out)


def run(name, seed):
    v = VARIANTS[name]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    env = v["env"]()
    env.game.rng = random.Random(seed)
    main, target = v["net"]().to(DEVICE), v["net"]().to(DEVICE)
    target.load_state_dict(main.state_dict())
    cuda = DEVICE.type == "cuda"
    learner = Learner(main, target, optim.Adam(main.parameters(), lr=LEARNING_RATE, fused=cuda, capturable=cuda), v["double"])
    actor = GraphedActor(main) if cuda else None
    buffer = ReplayBuffer(BUFFER_CAPACITY, Symmetric(GRID_SIZE) if v["aug"] else None)
    obs, mask = env.reset()
    for step in range(STEPS):
        epsilon = max(EPSILON_END, 1 - (1 - EPSILON_END) * step / (STEPS // 3))
        action = actor.act(obs, mask, epsilon) if actor else select_action(main, obs, mask, epsilon, DEVICE)
        next_obs, next_mask, reward, done = env.step(action)
        buffer.push(Transition(obs, action, reward, next_obs, next_mask, done))
        obs, mask = next_obs, next_mask
        if len(buffer) >= LEARNING_STARTS:
            learner.update(buffer.sample(BATCH_SIZE))
            if step % SYNC_EVERY == 0:
                target.load_state_dict(main.state_dict())
        if done:
            obs, mask = env.reset()
    plain = per_game_turns(main, v["env"], DEVICE)
    symmetric = per_game_turns(Wrapper(main).to(DEVICE).eval(), v["env"], DEVICE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(f"{RESULTS_DIR}/{name}_s{seed}.npz", plain=plain, symmetric=symmetric)
    print(f"{name} seed {seed}: {plain.mean():.2f} turns plain, {symmetric.mean():.2f} symmetry-averaged", flush=True)


def run_job(job):
    run(*job)


# ---- reference players: random, hand-written heuristic, exact solver ---------------------------------------------
_OBSERVER = environment()


def observe(game):
    """The training observation (7 x 6 x 6 array, legal-move mask) of any Game."""
    _OBSERVER.game = game
    return _OBSERVER.observe()


def random_policy(game):
    return random.choice(game.valid_moves())


def heuristic_policy(game):
    """Stand-in for an intuitive player: pick the cell next to the clue with the highest mine share."""
    obs, mask = observe(game)
    n, found, ratio = GRID_SIZE, obs[1] > .5, obs[5]
    score = np.full((n, n), obs[6][0, 0])
    for r in range(n):
        for c in range(n):
            if obs[0][r, c] > .5:
                near = [ratio[a, b] for a in range(max(r - 1, 0), min(r + 2, n)) for b in range(max(c - 1, 0), min(c + 2, n))
                        if obs[0][a, b] < .5 and not found[a, b] and obs[4][a, b] > 0]
                if near:
                    score[r, c] = max(near)
    return divmod(int(np.where(mask, score.flatten() + np.random.rand(n * n) * 1e-6, -1).argmax()), n)


def exact_probs(game):
    """Exact P(mine) of every covered cell: enumerate all mine layouts consistent with the clues and the mine count."""
    n, rev = GRID_SIZE, game.revealed
    nb = lambda r, c: [(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and 0 <= r + a < n and 0 <= c + b < n]
    covered = [(r, c) for r in range(n) for c in range(n) if (r, c) not in rev]
    left = game.bomb_count - sum(1 for v in rev.values() if v == BOMB)
    clues = []
    for (r, c), v in rev.items():
        if v != BOMB:
            hn = [x for x in nb(r, c) if x not in rev]
            if hn:
                clues.append((hn, v - sum(1 for x in nb(r, c) if rev.get(x) == BOMB)))
    frontier = sorted({x for hn, _ in clues for x in hn})
    interior = len(covered) - len(frontier)
    idx = {x: i for i, x in enumerate(frontier)}
    cl = [([idx[x] for x in hn], need) for hn, need in clues]
    by_cell = [[] for _ in frontier]
    for j, (cells, _) in enumerate(cl):
        for i in cells:
            by_cell[i].append(j)
    need, free = [k for _, k in cl], [len(cells) for cells, _ in cl]
    cnt, cellcnt, assign = {}, {}, [0] * len(frontier)

    def rec(i, b):
        if i == len(frontier):
            cnt[b] = cnt.get(b, 0) + 1
            arr = cellcnt.setdefault(b, [0] * len(frontier))
            for k in range(len(frontier)):
                arr[k] += assign[k]
            return
        for val in (0, 1):
            for j in by_cell[i]:
                need[j] -= val
                free[j] -= 1
            if all(0 <= need[j] <= free[j] for j in by_cell[i]):
                assign[i] = val
                rec(i + 1, b + val)
            for j in by_cell[i]:
                need[j] += val
                free[j] += 1
        assign[i] = 0

    rec(0, 0)
    z, fsum, isum = 0, [0] * len(frontier), 0
    for b, c in cnt.items():
        k = left - b
        if 0 <= k <= interior:
            w = comb(interior, k)
            z += c * w
            isum += c * w * k
            for t in range(len(frontier)):
                fsum[t] += cellcnt[b][t] * w
    return {x: (fsum[idx[x]] / z if x in idx else (isum / z / interior if interior else 0)) for x in covered}


def exact_policy(game):
    p = exact_probs(game)
    best = max(p.values())
    return random.choice([c for c, v in p.items() if v >= best - 1e-12])


def solo_each(policy, games, seed):
    """Turns per game of a game-level policy on boards seeded seed + i (the boards per_game_turns plays)."""
    out = []
    for g in range(games):
        game = Game(GRID_SIZE, BOMB_COUNT, n_players=1, rng=random.Random(seed + g))
        picks = 0
        while not game.done:
            game.pick(*policy(game))
            picks += 1
        out.append(picks - BOMB_COUNT)
    return np.array(out)


def benchmark_arrays():
    path = f"{RESULTS_DIR}/benchmarks.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    random.seed(0)
    np.random.seed(0)
    out = {k: solo_each(p, EVAL_GAMES, EVAL_SEED) for k, p in [("random", random_policy), ("heuristic", heuristic_policy), ("exact", exact_policy)]}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(path, **out)
    return out


# ---- ablation report --------------------------------------------------------------------------------------------
def ci95(v):
    v = np.asarray(v, float)
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def seed_means(name, key):
    return [float(np.load(f)[key].mean()) for f in sorted(glob.glob(f"{RESULTS_DIR}/{name}_s*.npz"))]


def mean_ci(values):
    """Mean and the half-width of its 95% Student-t interval over training seeds."""
    from scipy import stats
    v = np.asarray(values, float)
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def welch(reference, other):
    """other minus reference, and the half-width of its 95% Welch interval."""
    from scipy import stats
    a, b = np.asarray(reference, float), np.asarray(other, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(b.mean() - a.mean()), float(stats.t.ppf(0.975, df) * np.sqrt(va + vb))


def table(title, rows, reference="ab_final", differences=True):
    ref = seed_means(reference, "plain")
    head = "| Version | Weights | Seeds | Turns, plain network | Turns, symmetry-averaged |" + (" Difference to final (plain) |" if differences else "")
    lines = [f"**{title}**", "", head, "|---" * (6 if differences else 5) + "|"]
    for label, name in rows:
        plain, sym = seed_means(name, "plain"), seed_means(name, "symmetric")
        if len(plain) < 2:
            lines.append(f"| {label} | | {len(plain)} | not enough runs | | |")
            continue
        (m, h), (ms, hs) = mean_ci(plain), mean_ci(sym)
        weights = sum(p.numel() for p in VARIANTS[name]["net"]().parameters())
        row = f"| {label} | {weights:,} | {len(plain)} | {m:.2f} ± {h:.2f} | {ms:.2f} ± {hs:.2f} |"
        if differences:
            d, dh = welch(ref, plain) if name != reference and len(ref) > 1 else (None, None)
            row += " reference |" if name == reference else f" {d:+.2f} ({d - dh:+.2f} to {d + dh:+.2f}) |"
        lines.append(row)
    return "\n".join(lines)


def report():
    bench = benchmark_arrays()
    out = [f"**Reference players** (same {len(bench['random'])} boards, ±1.96 standard errors over boards)\n", "| Player | Turns |", "|---|---|"]
    out += [f"| {label} | {bench[k].mean():.2f} ± {ci95(bench[k]):.2f} |" for label, k in
            [("Random legal cell", "random"), ("Hand-written heuristic", "heuristic"), ("Exact probability solver", "exact")]]
    out += ["", table("Ingredients added one at a time", [
        ("1. Raw channels, fully connected head, DQN", "ab_raw"), ("2. + engineered feature channels", "ab_features"),
        ("3. + fully convolutional head", "ab_fcn"), ("4. + Double DQN", "ab_double"),
        ("5. + symmetry augmentation (final)", "ab_final")], differences=False),
        "", table("Leaving one ingredient out of the final recipe", [
            ("Final recipe", "ab_final"), ("without symmetry augmentation", "ab_double"), ("without Double DQN", "ab_no_double"),
            ("without engineered features (3 raw channels)", "ab_no_features"), ("without the fully convolutional head", "ab_no_fcn")])]
    print("\n".join(out))


# ---- the deployed model and the difficulty levels ---------------------------------------------------------------
def deployed(checkpoint="weight/best_ddqn.pt"):
    random.seed(0)                       # the exact solver breaks ties at random
    model = load_model(checkpoint).to(DEVICE)
    plain = per_game_turns(model, environment, DEVICE, seed=FRESH_SEED)
    averaged = per_game_turns(Wrapper(model).to(DEVICE).eval(), environment, DEVICE, seed=FRESH_SEED)
    exact = solo_each(exact_policy, EVAL_GAMES, FRESH_SEED)
    d = averaged - exact
    print(f"{EVAL_GAMES} fresh boards (±1.96 standard errors over boards)\n"
          f"  deployed, plain network       {plain.mean():.2f} ± {ci95(plain):.2f}\n"
          f"  deployed, symmetry-averaged   {averaged.mean():.2f} ± {ci95(averaged):.2f}\n"
          f"  exact solver                  {exact.mean():.2f} ± {ci95(exact):.2f}\n"
          f"  deployed minus exact (paired) {d.mean():+.2f} (95% CI {d.mean() - ci95(d):+.2f} to {d.mean() + ci95(d):+.2f})")


def web_levels(path="docs/js/config.js"):
    """Temperatures of the three levels, read from the web page's own settings so they cannot drift apart."""
    m = re.search(r"LEVELS = \{ easy: ([\d.]+), medium: ([\d.]+), hard: ([\d.]+) \}", open(path).read())
    return dict(zip(("easy", "medium", "hard"), map(float, m.groups())))


def onnx_policy(path, temperature, rng):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])

    def policy(game):
        obs, mask = observe(game)
        q = session.run(None, {"observation": obs[None]})[0][0]
        legal = np.flatnonzero(mask)
        if temperature <= 0:
            cell = rng.choice(legal[q[legal] >= q[legal].max() - 1e-5])      # best cell, ties broken at random
        else:
            w = np.exp((q[legal] - q[legal].max()) / temperature)
            cell = rng.choice(legal, p=w / w.sum())
        return divmod(int(cell), GRID_SIZE)
    return policy


def two_player(policy_a, policy_b, games, seed):
    """Wins of A and its mean bomb margin. A's seat and who moves first both alternate."""
    wins, margins = 0, []
    for g in range(games):
        seat_a = g % 2
        game = Game(GRID_SIZE, BOMB_COUNT, n_players=2, rng=random.Random(seed + g))
        game.reset(first_player=(g // 2) % 2)
        while not game.done:
            game.pick(*(policy_a if game.current == seat_a else policy_b)(game))
        wins += game.scores[seat_a] > game.scores[1 - seat_a]
        margins.append(game.scores[seat_a] - game.scores[1 - seat_a])
    return wins, float(np.mean(margins))


def wilson(wins, n, z=1.96):
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def levels(path="docs/ddqn.onnx", match_games=1000, match_seed=31000):
    rng = np.random.default_rng(0)
    random.seed(0)
    np.random.seed(0)
    temps = web_levels()
    pol = {k: onnx_policy(path, t, rng) for k, t in temps.items()}
    print(f"**The three levels, solo** (same {EVAL_GAMES} boards as the ablation, ±1.96 standard errors over boards)\n\n| Level | Temperature | Turns |\n|---|---|---|")
    for k, t in temps.items():
        v = solo_each(pol[k], EVAL_GAMES, EVAL_SEED)
        print(f"| {k.capitalize()} | {t} | {v.mean():.2f} ± {ci95(v):.2f} |")
    print(f"\n**Two-player matches** ({match_games} games each, seat and first move alternate, 95% Wilson interval for the first player)\n\n"
          "| Match | Win rate | 95% interval | Mean bomb margin |\n|---|---|---|---|")
    for na, a, nb, b in [("Hard", pol["hard"], "hand-written heuristic", heuristic_policy), ("Medium", pol["medium"], "hand-written heuristic", heuristic_policy),
                         ("Easy", pol["easy"], "hand-written heuristic", heuristic_policy), ("Hard", pol["hard"], "Medium", pol["medium"]),
                         ("Medium", pol["medium"], "Easy", pol["easy"]), ("Hard", pol["hard"], "Easy", pol["easy"]),
                         ("Exact solver", exact_policy, "Hard", pol["hard"])]:
        wins, margin = two_player(a, b, match_games, match_seed)
        p, lo, hi = wilson(wins, match_games)
        print(f"| {na} vs {nb} | {p:.1%} | {lo:.1%} to {hi:.1%} | {margin:+.2f} |")


def calibrate(checkpoint="weight/best_ddqn.pt", targets=(("easy", 17.5), ("medium", 12.5)), games=600, seed=4242):
    """Bisect each level's temperature until its solo average matches the target turns."""
    model = Wrapper(load_model(checkpoint)).to(DEVICE).eval()

    def turns(T):
        torch.manual_seed(seed)                                  # same noise for every T, so the search stays monotone

        def policy(obs, mask):
            with torch.no_grad():
                q = model(torch.from_numpy(obs).to(DEVICE))
                if T > 0:                                         # Gumbel-max: argmax(q / T + noise) samples softmax(q / T)
                    q = q / T - torch.log(-torch.log(torch.rand_like(q).clamp_min(1e-20)))
                return torch.where(torch.from_numpy(mask).to(DEVICE), q, torch.full_like(q, -1e30)).argmax(1).tolist()
        envs = [environment() for _ in range(games)]
        for i, env in enumerate(envs):
            env.game.rng = random.Random(seed + i)
        play_lockstep(envs, policy)
        return float(np.mean([turns_used(e) for e in envs]))

    for name, target in targets:
        lo, hi = 0.0, 3.0
        for _ in range(14):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if turns(mid) < target else (lo, mid)
        print(f"{name}: T = {(lo + hi) / 2:.3f} gives {turns((lo + hi) / 2):.2f} turns (target {target})")


def vectors(model="docs/ddqn.onnx", out="docs/tests/test_vectors.json", count=40):
    """Boards with the observation and values Python produces, for docs/tests/parity.mjs to check the JavaScript port against."""
    import onnxruntime as ort
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    rng, states = random.Random(99), []
    for g in range(40):
        game = Game(GRID_SIZE, BOMB_COUNT, n_players=1, rng=random.Random(99 + g))
        while not game.done:
            states.append((game.board_view(), *observe(game)))
            game.pick(*rng.choice(game.valid_moves()))
    chosen = np.random.default_rng(0).choice(len(states), count, replace=False)
    obs = np.stack([states[i][1] for i in chosen])
    q = session.run(None, {"observation": obs})[0]
    records = [{"board": states[i][0], "observation": np.round(obs[k], 7).tolist(), "legal_mask": states[i][2].tolist(),
                "q_values": np.round(q[k], 6).tolist(), "greedy_move": int(np.where(states[i][2], q[k], -1e30).argmax())}
               for k, i in enumerate(chosen)]
    with open(out, "w") as f:
        json.dump({"note": "board: null = covered, \"bomb\" = found mine, 0-8 = clue", "vectors": records}, f)
    print(f"wrote {out} ({count} boards)")


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "help"), sys.argv[2:]
    opt = lambda flag, default: args[args.index(flag) + 1] if flag in args else default
    positional = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    if cmd == "train" and positional:
        seeds, workers = [int(s) for s in opt("--seeds", "0").split(",")], int(opt("--workers", 1))
        jobs = [(name, seed) for name in positional[0].split(",") for seed in seeds]
        if workers > 1:   # one process per (variant, seed)
            with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                list(pool.map(run_job, jobs))
        else:
            [run_job(job) for job in jobs]
    elif cmd == "report":
        report()
    elif cmd in ("deployed", "levels", "calibrate"):
        {"deployed": deployed, "levels": levels, "calibrate": calibrate}[cmd](*positional[:1])
    elif cmd == "vectors":
        vectors()
    else:
        print(__doc__)
