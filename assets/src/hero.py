"""Dithered animated hero banner for the bynguts profile README.

Renders at a low internal resolution, applies multi-level ordered (Bayer 8x8)
dithering into a 5-tone blood palette, then upscales with nearest neighbour so
every dither dot stays crisp.

Fonts (OFL, from github.com/google/fonts) go in assets/src/fonts/:
UnifrakturMaguntia-Book.ttf, Silkscreen-Regular.ttf.
Run: python assets/src/hero.py assets/hero.gif
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONTS = HERE / "fonts"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "hero.gif"

W, H = 450, 150          # internal resolution
SCALE = 2                # output is 900x300
FRAMES = 40
FRAME_MS = 80

# Palette: void, dried blood, crimson, bright red, bone
PALETTE = np.array(
    [
        (13, 17, 23),
        (69, 10, 10),
        (153, 27, 27),
        (239, 68, 68),
        (240, 232, 226),
    ],
    dtype=np.uint8,
)
LEVELS = len(PALETTE)


def bayer(n):
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (n * n)


B8 = bayer(8)


def dither(field, ox=0, oy=0):
    """Map a 0..1 intensity field to palette indices with ordered dithering."""
    h, w = field.shape
    t = np.roll(np.roll(np.tile(B8, (h // 8 + 1, w // 8 + 1))[:h, :w], oy, 0), ox, 1)
    v = np.clip(field, 0, 1) * (LEVELS - 1)
    base = np.floor(v)
    idx = base + (v - base > t)
    return np.clip(idx, 0, LEVELS - 1).astype(np.uint8)


def text_mask(text, font, size, pad=4):
    f = ImageFont.truetype(str(FONTS / font), size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + pad * 2, b - t + pad * 2), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=255)
    return np.asarray(im, dtype=np.float32) / 255.0


def paste(dst, src, x, y):
    h, w = src.shape
    dst[y : y + h, x : x + w] = np.maximum(dst[y : y + h, x : x + w], src)


# --- static layers ---------------------------------------------------------

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# Guts portrait: crop the upper body, map luminance onto the blood ramp.
guts = Image.open(HERE.parents[1] / "guts-pixel.png").convert("L")
gw, gh = guts.size
guts = guts.crop((0, 0, gw, int(gw * 1.05)))
PW = 150
guts = guts.resize((PW, int(PW * guts.height / guts.width)), Image.LANCZOS)
g = np.asarray(guts, dtype=np.float32) / 255.0
g = g[: H, :]
# lift contrast, keep blacks deep
g = np.clip((g - 0.12) / 0.8, 0, 1) ** 1.15
PX = W - PW - 6
portrait = np.zeros((H, W), np.float32)
portrait[: g.shape[0], PX : PX + PW] = g * 0.98
# feather the portrait's left edge and bottom into the void
fx = np.clip((xx - PX) / 46.0, 0, 1)
fy = np.clip((H - 2 - yy) / 26.0, 0, 1)
portrait *= fx * fy
# vignette around the face so the bright sky doesn't compete with it
FACE_X, FACE_Y = PX + 78, 62
dv = np.sqrt(((xx - FACE_X) / 95.0) ** 2 + ((yy - FACE_Y) / 80.0) ** 2)
portrait *= np.clip(1.15 - 0.75 * dv ** 2, 0.25, 1.0)

# Title and lines of copy
title = text_mask("bynguts", "UnifrakturMaguntia-Book.ttf", 64)
tag = text_mask("FULL-STACK DEV  /  AI AGENTS", "Silkscreen-Regular.ttf", 8, pad=0)
prompt = text_mask("> shipping agents that do real work", "Silkscreen-Regular.ttf", 8, pad=0)
cursor = np.ones((7, 5), np.float32)

TX, TY = 22, 16
title_layer = np.zeros((H, W), np.float32)
paste(title_layer, title, TX, TY)
# vertical bone -> crimson gradient through the letters
title_grad = np.clip(1.0 - (yy - TY) / (title.shape[0] * 1.05), 0.42, 1.0)

copy_layer = np.zeros((H, W), np.float32)
tag_y = TY + title.shape[0] + 4
paste(copy_layer, tag, TX + 4, tag_y)
prompt_y = tag_y + 18
paste(copy_layer, prompt, TX + 4, prompt_y)
cursor_x = TX + 4 + prompt.shape[1] + 3

# thin rule under the title, spanning the wordmark's ink from "b" to "s"
ink = np.nonzero(title.max(axis=0) > 0.5)[0]
rule = np.zeros((H, W), np.float32)
rule[tag_y - 6, TX + ink[0] : TX + ink[-1] + 1] = 1.0

# --- frames ----------------------------------------------------------------

frames = []
for f in range(FRAMES):
    p = f / FRAMES
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * p)

    # background: dark void with a breathing crimson glow behind the portrait
    cx, cy = PX + PW * 0.55, H * 0.42
    d = np.sqrt(((xx - cx) / 150.0) ** 2 + ((yy - cy) / 95.0) ** 2)
    glow = np.clip(1.0 - d, 0, 1) ** 1.6 * (0.30 + 0.10 * pulse)
    field = glow.copy()

    # portrait with a glitch scan band travelling downward
    band_y = int((p * (H + 40)) - 20)
    por = portrait.copy()
    band = (yy >= band_y) & (yy < band_y + 7) & (xx >= PX + 20)
    if band.any():
        rows = np.unique(yy[band].astype(int))
        for r in rows:
            if 0 <= r < H:
                shift = int(5 * math.sin(r * 1.7 + f))
                seg = por[r, PX:]
                por[r, PX:] = np.roll(seg, shift) * 1.25 + 0.08
    # breathing makes the dither pattern crawl across the portrait
    por *= 0.92 + 0.10 * pulse
    field = np.maximum(field, por)

    # title, rule, copy
    # diagonal light sweep across the blackletter
    sweep_x = -60 + p * (title.shape[1] + 260)
    sweep = np.exp(-(((xx - TX) + (yy - TY) * 0.6 - sweep_x) / 14.0) ** 2) * 0.55
    field = np.maximum(field, title_layer * np.clip(title_grad + sweep, 0, 1))
    field = np.maximum(field, rule * 0.45)
    field = np.where(copy_layer > 0.5, 0.80, field)
    if (f // 5) % 2 == 0:
        field[prompt_y : prompt_y + 7, cursor_x : cursor_x + 5] = 0.62

    idx = dither(field)
    # shimmer: re-dither only the scan band with a shifted threshold matrix
    if band.any():
        alt = dither(field, ox=f % 8, oy=(f * 3) % 8)
        idx = np.where(band, alt, idx)

    rgb = PALETTE[idx]
    im = Image.fromarray(rgb, "RGB").resize((W * SCALE, H * SCALE), Image.NEAREST)
    frames.append(im)

pal_img = Image.new("P", (1, 1))
pal_img.putpalette(PALETTE.flatten().tolist() + [0] * (768 - PALETTE.size))
q = [fr.quantize(palette=pal_img, dither=Image.Dither.NONE) for fr in frames]
q[0].save(
    OUT,
    save_all=True,
    append_images=q[1:],
    duration=FRAME_MS,
    loop=0,
    optimize=True,
    disposal=1,
)
print(OUT, OUT.stat().st_size // 1024, "KB")
