"""Dithered section headers and animated ember footer for the profile README.

Both sit on the same solid void background as the hero and boot log, so the
page reads as one dark system on GitHub's light and dark themes. Same Bayer
8x8 ordered dithering as hero.py.

Fonts: Silkscreen-Regular.ttf and Silkscreen-Bold.ttf in assets/src/fonts/.
Run: python assets/src/extras.py assets
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONTS = HERE / "fonts"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 2

# index 0 is the void background
PALETTE = [(13, 17, 23), (127, 29, 29), (185, 28, 28), (239, 68, 68)]
LEVELS = len(PALETTE)


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


def text_mask(text, font, size):
    f = ImageFont.truetype(str(FONTS / font), size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l, b - t), 0)
    ImageDraw.Draw(im).text((-l, -t), text, font=f, fill=255)
    return np.asarray(im, dtype=np.float32) / 255.0 > 0.5


def to_image(idx):
    im = Image.fromarray(idx, "P")
    im.putpalette([c for rgb in PALETTE for c in rgb])
    w, h = im.size
    return im.resize((w * SCALE, h * SCALE), Image.NEAREST)


def header(num, label, name):
    W, H = 450, 22
    num_m = text_mask(num, "Silkscreen-Regular.ttf", 8)
    lab_m = text_mask(label, "Silkscreen-Bold.ttf", 8)
    idx = np.zeros((H, W), np.uint8)
    y = (H - lab_m.shape[0]) // 2
    x = 0
    idx[y : y + num_m.shape[0], x : x + num_m.shape[1]][num_m] = 2
    x += num_m.shape[1] + 6
    slash = text_mask("/", "Silkscreen-Regular.ttf", 8)
    idx[y : y + slash.shape[0], x : x + slash.shape[1]][slash] = 2
    x += slash.shape[1] + 6
    idx[y : y + lab_m.shape[0], x : x + lab_m.shape[1]][lab_m] = 3
    x += lab_m.shape[1] + 10
    # dithered rule fading out to the right
    ry = y + lab_m.shape[0] // 2 - 1
    xs = np.arange(W - x, dtype=np.float32)
    fade = np.clip(1.0 - xs / (W - x), 0, 1) ** 0.8 * 0.95
    rule = dither(np.tile(fade, (3, 1)))
    idx[ry : ry + 3, x:] = rule
    im = to_image(idx)
    im.save(OUT / name, optimize=True)
    print(name, (OUT / name).stat().st_size, "B")


def periodic_noise(h, w, seed, scale):
    rng = np.random.default_rng(seed)
    spec = np.fft.fft2(rng.standard_normal((h, w)))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    spec *= np.exp(-((fx * w / scale) ** 2 + (fy * h / scale) ** 2))
    n = np.real(np.fft.ifft2(spec))
    return (n - n.min()) / (n.max() - n.min())


def typing_script(quotes, cps=3, hold=14, erase=8):
    """Per-frame (quote index, visible chars, cursor on) for type/hold/erase."""
    steps = []
    for qi, q in enumerate(quotes):
        n = len(q)
        for c in range(0, n + cps, cps):
            steps.append((qi, min(c, n), True))
        for k in range(hold):
            steps.append((qi, n, (k // 3) % 2 == 0))
        for k in range(erase):
            steps.append((qi, int(n * (1 - (k + 1) / erase)), True))
    return steps


def footer(name, quotes):
    W, H = 450, 64
    script = typing_script(quotes)
    FRAMES = len(script)
    masks = [text_mask(q, "Silkscreen-Regular.ttf", 8) for q in quotes]
    # per-prefix masks so each frame shows a clean partial string
    qy = 6
    # tileable noise in y so the upward scroll loops seamlessly
    n1 = periodic_noise(H, W, 7, 9.0)
    n2 = periodic_noise(H, W, 11, 22.0)
    yy = np.arange(H, dtype=np.float32)[:, None]
    ramp = np.clip((yy - 16) / (H - 16), 0, 1) ** 1.4  # embers rise from the bottom
    frames = []
    for f in range(FRAMES):
        s1 = int(round(f * H / FRAMES))       # one full noise period per loop
        s2 = int(round(f * H / FRAMES)) * 2 % H
        a = np.roll(n1, -s1, 0)
        b = np.roll(n2, -s2, 0)
        field = np.clip((a * 0.65 + b * 0.55 - 0.42) * 2.2, 0, 1) * ramp
        idx = dither(field)
        qi, chars, cursor = script[f]
        full = masks[qi]
        qx = (W - full.shape[1]) // 2  # anchor at the full quote's centred start
        x = qx
        if chars:
            part = text_mask(quotes[qi][:chars], "Silkscreen-Regular.ttf", 8)
            idx[qy : qy + part.shape[0], qx : qx + part.shape[1]][part] = 3
            x = qx + part.shape[1]
        if cursor:
            idx[qy : qy + full.shape[0], x + 2 : x + 6] = 3
        frames.append(to_image(idx))
    frames[0].save(
        OUT / name,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        disposal=1,
        optimize=False,
    )
    print(name, (OUT / name).stat().st_size // 1024, "KB")


header("01", "ABOUT", "h-about.png")
header("02", "ACTIVITY", "h-activity.png")
footer(
    "footer.gif",
    [
        "FATE MAY BE WRITTEN, BUT THE STRUGGLER KEEPS FIGHTING AGAINST IT.",
        "JUST SOMEONE TRYING TO MAKE THINGS HAPPEN.",
    ],
)
