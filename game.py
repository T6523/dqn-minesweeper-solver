"""Find My Mines rules. Pure game logic: no pygame, no RL code, no clock.

Bomb: scorer keeps the turn. Empty: shows its 8-neighbour count, turn passes.
Match ends when all bombs are found. n_players=1 is solo (for training).
"""
import random

BOMB = "bomb"  # a revealed bomb, in a board view


class Game:
    def __init__(self, grid_size=6, bomb_count=11, n_players=2, rng=None):
        self.grid_size = grid_size
        self.bomb_count = bomb_count
        self.n_players = n_players
        self.rng = rng or random.Random()
        self.reset()

    def reset(self, first_player=None):
        """Deal a fresh board. first_player=None picks a seat at random."""
        n = self.grid_size
        cells = [(r, c) for r in range(n) for c in range(n)]
        self.bombs = set(self.rng.sample(cells, self.bomb_count))
        self.revealed = {}  # (row, col) -> BOMB or int
        self.owner = {}     # (row, col) of a found bomb -> seat that found it
        self.scores = [0] * self.n_players
        self.current = self.rng.randrange(self.n_players) if first_player is None else first_player
        self.adjacent = [
            [sum(nb in self.bombs for nb in self._neighbours(r, c)) for c in range(n)]
            for r in range(n)
        ]

    def _neighbours(self, row, col):
        """The eight surrounding cells, clipped to the grid."""
        n = self.grid_size
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if (dr or dc) and 0 <= r < n and 0 <= c < n:
                    yield r, c

    def pick(self, row, col):
        """Current seat reveals one cell.

        Returns {"ok", "player", "is_bomb", "value", "turn_changed", "done"}.
        ok=False (off board / already revealed / game over) changes nothing;
        the env decides what penalty that earns.
        """
        n = self.grid_size
        player = self.current
        if self.done or not (0 <= row < n and 0 <= col < n) or (row, col) in self.revealed:
            return {"ok": False, "player": player, "is_bomb": False, "value": None,
                    "turn_changed": False, "done": self.done}
        is_bomb = (row, col) in self.bombs
        self.revealed[(row, col)] = BOMB if is_bomb else self.adjacent[row][col]
        if is_bomb:
            self.scores[player] += 1
            self.owner[(row, col)] = player
        else:
            self.pass_turn()
        return {"ok": True, "player": player, "is_bomb": is_bomb,
                "value": self.revealed[(row, col)],
                "turn_changed": self.current != player, "done": self.done}

    def pass_turn(self):
        """Hand the turn to the next seat. The app calls this on a timeout."""
        if not self.done:
            self.current = (self.current + 1) % self.n_players

    def valid_moves(self):
        """Unrevealed (row, col) cells, handy for action masking."""
        n = self.grid_size
        return [(r, c) for r in range(n) for c in range(n) if (r, c) not in self.revealed]

    def board_view(self):
        """Nested lists: None hidden, BOMB found bomb, 0-8 neighbour count.

        This is all a real client sees. Train on this, never on self.bombs.
        """
        n = self.grid_size
        return [[self.revealed.get((r, c)) for c in range(n)] for r in range(n)]

    @property
    def done(self):
        return sum(self.scores) >= self.bomb_count

    @property
    def winner(self):
        """Seat with the most bombs once done, else None (odd bomb_count: no ties for 2)."""
        return max(range(self.n_players), key=self.scores.__getitem__) if self.done else None

    @property
    def bombs_left(self):
        return self.bomb_count - sum(self.scores)
