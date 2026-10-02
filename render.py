"""pygame viewer for game.Game. Call Viewer.draw(game) after each pick.

Demo (random agents):  python render.py
"""
import pygame

from game import BOMB, Game

CELL = 72
HEADER = 56
COLORS = {
    "bg": (30, 30, 36), "hidden": (70, 74, 90), "safe": (200, 204, 214),
    "unfound": (120, 50, 50), "text": (20, 20, 20), "hud": (235, 235, 235),
}
PLAYER_COLORS = [(220, 60, 60), (60, 120, 220), (60, 180, 90), (230, 180, 40)]


class Viewer:
    def __init__(self, game, fps=5):
        pygame.init()
        n = game.grid_size
        self.screen = pygame.display.set_mode((n * CELL, n * CELL + HEADER))
        pygame.display.set_caption("Find My Mines")
        self.font = pygame.font.SysFont(None, 30)
        self.big = pygame.font.SysFont(None, 64)
        self.clock = pygame.time.Clock()
        self.fps = fps

    def draw(self, game, reveal_all=False):
        """Render one frame. Returns False once the window is closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        self.screen.fill(COLORS["bg"])
        self._draw_scores(game)
        self._draw_board(game, reveal_all)
        if game.done:
            self._draw_banner(f"Player {game.winner + 1} wins")
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _draw_scores(self, game):
        x = 8
        for p, score in enumerate(game.scores):
            label = f"P{p + 1}: {score}"
            if p == game.current and not game.done:
                label = "> " + label
            txt = self.font.render(label, True, PLAYER_COLORS[p % len(PLAYER_COLORS)])
            self.screen.blit(txt, (x, 8))
            x += txt.get_width() + 24
        left = self.font.render(f"bombs left: {game.bombs_left}", True, COLORS["hud"])
        self.screen.blit(left, (8, 32))

    def _draw_board(self, game, reveal_all):
        for r, row in enumerate(game.board_view()):
            for c, cell in enumerate(row):
                rect = pygame.Rect(c * CELL, HEADER + r * CELL, CELL - 2, CELL - 2)
                if cell is None:
                    unfound = reveal_all and (r, c) in game.bombs
                    pygame.draw.rect(self.screen, COLORS["unfound" if unfound else "hidden"], rect)
                elif cell == BOMB:
                    pygame.draw.rect(self.screen, PLAYER_COLORS[game.owner[(r, c)] % len(PLAYER_COLORS)], rect)
                else:
                    pygame.draw.rect(self.screen, COLORS["safe"], rect)
                    txt = self.font.render(str(cell), True, COLORS["text"])
                    self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_banner(self, text):
        txt = self.big.render(text, True, COLORS["hud"])
        rect = txt.get_rect(center=self.screen.get_rect().center)
        pygame.draw.rect(self.screen, COLORS["bg"], rect.inflate(32, 24))
        self.screen.blit(txt, rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    import random

    game = Game()
    view = Viewer(game)
    while view.draw(game):
        if game.done:
            pygame.time.wait(1500)
            game.reset(first_player=game.winner)  # winner starts the rematch
        game.pick(*random.choice(game.valid_moves()))
