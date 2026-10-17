"""Dithered "cereza.boot" terminal log for the About section.

Lines appear one by one, a dithered progress bar charges, then the prompt
cursor blinks before the loop restarts. Same palette and Bayer 8x8 ordered
dithering as hero.py.

Fonts: Silkscreen-Regular.ttf and Silkscreen-Bold.ttf in assets/src/fonts/.
Run: python assets/src/boot.py assets/boot.gif
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONTS = HERE / "fonts"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "boot.gif"

W, H = 450, 158
SCALE = 2

# void, dried blood, crimson, bright red, bone
PALETTE = np.array(
    [(13, 17, 23), (69, 10, 10), (153, 27, 27), (239, 68, 68), (240, 232, 226)],
    dtype=np.uint8,
)
VOID, BLOOD, CRIMSON, BRIGHT, BONE = range(5)
LEVELS = len(PALETTE)

# (tag, label, value); the last entry is the charging bar
LINES = [
    ("OK", "WAKING THE WITCH", "DONE"),
    ("OK", "BINDING AGENTS", "3/3"),
    ("OK", "LOADING STACK", "GO / PY / TS"),
    ("OK", "ROLE MODEL", "GUTS"),
    ("OK", "OFF-DUTY", "ANIME / FILM / STORY GAMES"),
    ("!!", "WISHLIST", "UNLIMITED CLAUDE CODE"),
    ("..", "UMBRAN CLIMAX", None),
]


def bayer(n):
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (n * n)


B8 = bayer(8)


def dither(field):
    h, w = field.shape
    t = np.tile(B8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    v = np.clip(field, 0, 1) * (LEVELS - 1)
    base = np.floor(v)
    return np.clip(base + (v - base > t), 0, LEVELS - 1).astype(np.uint8)


_fonts = {}


def text(idx, s, x, y, color, bold=False):
    """Draw pixel text into the index buffer; returns the x after the text."""
    name = "Silkscreen-Bold.ttf" if bold else "Silkscreen-Regular.ttf"
    f = _fonts.setdefault(name, ImageFont.truetype(str(FONTS / name), 8))
    l, t, r, b = f.getbbox(s)
    im = Image.new("L", (r - l, b - t), 0)
    ImageDraw.Draw(im).text((-l, -t), s, font=f, fill=255)
    m = np.asarray(im) > 127
    h, w = m.shape
    idx[y : y + h, x : x + w][m] = color
    return x + w


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# static panel: faint glow from the top-right corner, frame, title bar
glow = np.clip(1.0 - np.sqrt(((xx - W) / 260.0) ** 2 + (yy / 150.0) ** 2), 0, 1) ** 2 * 0.22
panel = dither(glow)
panel[0, :] = panel[-1, :] = CRIMSON
panel[:, 0] = panel[:, -1] = CRIMSON
bar = dither(np.tile(np.clip(1.0 - np.arange(W - 2) / (W * 0.9), 0, 1) * 0.5, (13, 1)))
panel[1:14, 1:-1] = bar
text(panel, "CEREZA.BOOT", 8, 4, BONE, bold=True)
text(panel, "TTY0  //  OPERATOR BYNGUTS", W - 150, 4, BRIGHT)
text(panel, "// JUST SOMEONE TRYING TO MAKE THINGS HAPPEN.", 10, 22, CRIMSON)

LX, LY, STEP = 10, 38, 13
VALUE_X = W - 10  # values are right-aligned to this edge
TAG_COLOR = {"OK": BRIGHT, "!!": BONE, "..": CRIMSON}


def draw_line(idx, i, chars=None):
    tag, label, value = LINES[i]
    y = LY + i * STEP
    text(idx, "[", LX, y, CRIMSON)
    probe = np.zeros_like(idx)
    tw = text(probe, tag, 0, 0, 1, bold=True)
    text(idx, tag, LX + 4 + (13 - tw) // 2, y, TAG_COLOR[tag], bold=True)
    text(idx, "]", LX + 19, y, CRIMSON)
    shown = label if chars is None else label[:chars]
    x = text(idx, shown, LX + 31, y, BONE)
    return x, y


def draw_value(idx, i, end_x):
    _, _, value = LINES[i]
    y = LY + i * STEP
    probe = np.zeros_like(idx)
    w = text(probe, value, 0, 0, 1) if value else 0
    vx = VALUE_X - w
    for dx in range(end_x + 6, vx - 6, 4):  # dotted leader
        idx[y + 4, dx] = BLOOD
    text(idx, value, vx, y, BRIGHT)


def draw_bar(idx, end_x, fill):
    y = LY + (len(LINES) - 1) * STEP
    x0, x1 = end_x + 10, VALUE_X - 34
    idx[y - 1, x0 - 1 : x1 + 1] = BLOOD
    idx[y + 7, x0 - 1 : x1 + 1] = BLOOD
    idx[y - 1 : y + 8, x0 - 1] = BLOOD
    idx[y - 1 : y + 8, x1] = BLOOD
    n = int((x1 - x0) * fill)
    if n > 0:
        ramp = np.linspace(0.35, 1.0, x1 - x0)[:n]
        idx[y : y + 7, x0 : x0 + n] = dither(np.tile(ramp, (7, 1)))
    text(idx, f"{int(fill * 100):>3d}%", VALUE_X - 26, y, BRIGHT)


PROMPT_Y = LY + len(LINES) * STEP + 2

frames, durations = [], []


def emit(idx, ms):
    frames.append(idx.copy())
    durations.append(ms)


# 1) type each line, then drop its value in
for i in range(len(LINES)):
    label = LINES[i][1]
    for c in range(0, len(label) + 1, 4):
        idx = panel.copy()
        for j in range(i):
            ex, _ = draw_line(idx, j)
            if LINES[j][2] is not None:
                draw_value(idx, j, ex)
        draw_line(idx, i, chars=c)
        emit(idx, 45)
    if LINES[i][2] is not None:
        idx = panel.copy()
        for j in range(i + 1):
            ex, _ = draw_line(idx, j)
            draw_value(idx, j, ex)
        emit(idx, 160)

# 2) charge the umbran climax bar
base = panel.copy()
last_end = 0
for j in range(len(LINES)):
    ex, _ = draw_line(base, j)
    if LINES[j][2] is not None:
        draw_value(base, j, ex)
    else:
        last_end = ex
for k in range(13):
    idx = base.copy()
    draw_bar(idx, last_end, min(0.72, k * 0.06))
    emit(idx, 70)

# 3) prompt with a blinking cursor, then hold
done = frames[-1].copy()
px = text(done, "> OPERATOR: BYNGUTS", LX, PROMPT_Y, BONE)
for k in range(8):
    idx = done.copy()
    if k % 2 == 0:
        idx[PROMPT_Y : PROMPT_Y + 7, px + 3 : px + 8] = BRIGHT
    emit(idx, 420)

pal = Image.new("P", (1, 1))
pal.putpalette(PALETTE.flatten().tolist() + [0] * (768 - PALETTE.size))
imgs = []
for idx in frames:
    im = Image.fromarray(PALETTE[idx], "RGB").resize((W * SCALE, H * SCALE), Image.NEAREST)
    imgs.append(im.quantize(palette=pal, dither=Image.Dither.NONE))
imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=True, disposal=1)
print(OUT, OUT.stat().st_size // 1024, "KB", len(imgs), "frames")
