"""Records assets/demo.gif: Hard bot, a few moves, then the dark-mode button.

Needs: pip install playwright, Chrome or Chromium (set CHROME=/path if not at /usr/bin/google-chrome), ffmpeg.
Run from the repo root: python scripts/record_demo.py
"""
import http.server
import os
import random
import subprocess
import tempfile
import threading
import time
from functools import partial
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
CHROME = os.environ.get('CHROME', '/usr/bin/google-chrome')
SIZE = {'width': 760, 'height': 800}
MOVES = 3                  # human picks before the theme toggle
random.seed(3)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve():
    handler = partial(Quiet, directory=str(ROOT / 'docs'))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def record(tmp):
    server = serve()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME)
        ctx = browser.new_context(viewport=SIZE, record_video_dir=tmp, record_video_size=SIZE, color_scheme='light')
        t0 = time.time()
        page = ctx.new_page()
        page.goto(f'http://127.0.0.1:{server.server_port}/')
        page.wait_for_selector('body.ready', timeout=30000)
        page.click('label.level:has(input[value=hard])')       # the model is loaded: this is the readiness signal
        ready = time.time() - t0
        page.click('#new-game')
        page.wait_for_timeout(600)
        picked = 0
        deadline = time.time() + 40
        while picked < MOVES and time.time() < deadline:
            cells = page.locator('.cell.open')
            if cells.count():
                cells.nth(random.randrange(cells.count())).click()
                picked += 1
                page.wait_for_timeout(700)
            else:
                page.wait_for_timeout(200)                     # the bot is moving
        page.wait_for_timeout(2500)                            # let the bot's reply land
        page.click('#theme-toggle')
        page.wait_for_timeout(2200)
        ctx.close()
        video = Path(page.video.path())
        browser.close()
    server.shutdown()
    return video, ready


def to_gif(webm, start, duration, out):
    palette = Path(tempfile.gettempdir()) / 'demo_palette.png'
    vf = 'fps=12,scale=640:-1:flags=lanczos'      # the viewport is the app, so no crop
    subprocess.run(['ffmpeg', '-y', '-ss', str(start), '-t', str(duration), '-i', str(webm),
                    '-vf', vf + ',palettegen=stats_mode=diff', str(palette)], check=True, capture_output=True)
    subprocess.run(['ffmpeg', '-y', '-ss', str(start), '-t', str(duration), '-i', str(webm), '-i', str(palette),
                    '-lavfi', vf + '[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle', str(out)],
                   check=True, capture_output=True)


if __name__ == '__main__':
    (ROOT / 'assets').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        webm, ready = record(tmp)
        total = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                               '-of', 'csv=p=0', str(webm)]).strip())
        start = max(0.0, ready - 1.0)
        to_gif(webm, start, min(14.0, total - start), ROOT / 'assets' / 'demo.gif')
    print('wrote assets/demo.gif')
