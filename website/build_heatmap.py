"""
Build a per-example answer-evidence HEATMAP: for STI and SIT, overlay a smooth
heat field on the raw image showing which patches decode an answer-evidence token.
STI concentrates heat on the answer object; SIT stays mostly cold. Offline -- uses
the per-patch decoded-word grids already saved in the manifest (no model/GPU).

    python build_heatmap.py     # reads assets/manifest.json + assets/ex*_image.png
                                #  writes assets/ex*_heat.png, updates manifest
"""
import json, os, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom, gaussian_filter
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
FDIR = "/usr/share/fonts/truetype/dejavu"
COLW = 468            # width of each heatmap column
PAD, GAP = 20, 16
GREEN = (28, 150, 88); RED = (206, 58, 48); INK = (26, 28, 31)
CMAP = cm.get_cmap("inferno")


def font(name, size):
    try:
        return ImageFont.truetype(os.path.join(FDIR, name), size)
    except Exception:
        return ImageFont.load_default()


F_EYE = font("DejaVuSans-Bold.ttf", 14)
F_COL = font("DejaVuSans-Bold.ttf", 17)
F_CAP = font("DejaVuSans.ttf", 15)


def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def matches(pw, targets):
    pw = pw.strip().lower()
    if not re.fullmatch(r"[a-z-]{3,}", pw):
        return False
    a = stem(pw)
    for t in targets:
        b = stem(t.lower())
        if len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)
                                            or a in b or b in a):
            return True
    return False


def heat_overlay(raw, words, gh, gw, targets, tint):
    """Grayscale raw image with a warm heat field on evidence patches."""
    W, H = raw.size
    mask = np.array([[1.0 if matches(words[r * gw + c], targets) else 0.0
                      for c in range(gw)] for r in range(gh)])
    # smooth upscale to image size + blur -> continuous heat field in [0,1]
    zy, zx = H / gh, W / gw
    heat = zoom(mask, (zy, zx), order=1)[:H, :W]
    if heat.shape != (H, W):                      # pad/crop to exact size
        hh = np.zeros((H, W)); h2 = heat[:H, :W]
        hh[:h2.shape[0], :h2.shape[1]] = h2; heat = hh
    heat = gaussian_filter(heat, sigma=max(zy, zx) * 0.5)
    if heat.max() > 0:
        heat = heat / heat.max()
    gray = np.asarray(raw.convert("L").convert("RGB")).astype(float) * 0.55
    color = CMAP(heat)[:, :, :3] * 255.0
    a = (heat ** 0.75)[:, :, None]                # alpha ramps with intensity
    out = gray * (1 - a) + color * a
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    # thin coloured frame to key it to STI (green) / SIT (red)
    ImageDraw.Draw(img).rectangle([0, 0, W - 1, H - 1], outline=tint, width=3)
    return img


def panel(e):
    raw = Image.open(os.path.join(ASSETS, e["image"])).convert("RGB")
    scale = COLW / raw.width
    raw = raw.resize((COLW, int(raw.height * scale)), Image.LANCZOS)
    gh, gw = e["grid_h"], e["grid_w"]
    targets = e.get("evidence") or [e.get("key", "")]
    sti = heat_overlay(raw, e["sti_words"], gh, gw, targets, GREEN)
    sit = heat_overlay(raw, e["sit_words"], gh, gw, targets, RED)
    ih = sti.height
    W = PAD + COLW + GAP + COLW + PAD
    eyeh, caph = 30, 30
    out = Image.new("RGB", (W, eyeh + caph + ih + 34), (255, 255, 255))
    d = ImageDraw.Draw(out)
    d.text((PAD, 8), "ANSWER-EVIDENCE HEATMAP  ·  where each ordering decodes "
           "the answer tokens", font=F_EYE, fill=(90, 96, 104))
    lx = PAD + COLW / 2
    rx = PAD + COLW + GAP + COLW / 2
    d.text((lx, eyeh + 15), "STI (question-first)", font=F_COL, fill=GREEN, anchor="mm")
    d.text((rx, eyeh + 15), "SIT (question-last)", font=F_COL, fill=RED, anchor="mm")
    out.paste(sti, (PAD, eyeh + caph))
    out.paste(sit, (PAD + COLW + GAP, eyeh + caph))
    d.text((PAD, eyeh + caph + ih + 8),
           "Warm = patches whose logit lens decodes an answer-evidence token; "
           "STI concentrates on the object, SIT stays mostly cold.",
           font=F_CAP, fill=(70, 76, 84))
    ImageDraw.Draw(out).rectangle([0, 0, W - 1, out.height - 1],
                                  outline=(210, 216, 222), width=1)
    return out


def main():
    man = json.load(open(os.path.join(ASSETS, "manifest.json")))
    for e in man["examples"]:
        if "sti_words" not in e or "grid_w" not in e:
            continue
        fig = panel(e)
        fn = "ex%d_heat.png" % e["idx"]
        fig.save(os.path.join(ASSETS, fn))
        e["heatmap"] = fn
        print("  %d: %s (%dx%d)" % (e["idx"], fn, fig.width, fig.height))
    json.dump(man, open(os.path.join(ASSETS, "manifest.json"), "w"), indent=2)
    print("[done]")


if __name__ == "__main__":
    main()
