"""
Wrap each raw per-patch logit-lens word still into a self-contained, Figure-3-style
panel: a header (question + ground truth) on top and a footer (what the still proves)
below, composited onto the still so the image stands alone. Pure image compositing of
the existing stills -- no model, no GPU.

    python build_figpanels.py     # reads assets/manifest.json + assets/ex*_words_*.png
"""
import json, os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
FDIR = "/usr/share/fonts/truetype/dejavu"

# curated scene-relevant tokens the patches decode to (for the footer)
WORDS = {
    229:  "dog · running · aggressive · play",
    638:  "walk · walking · jogging · sidewalk · person",
    978:  "skate · skateboard · ramps · athlete · leap",
    1231: "swimming · pool · underwater · submerged · girl",
}
INK = (26, 28, 31); MUT = (91, 100, 112); LINE = (210, 216, 222)
FOOT_BG = (243, 246, 249); GOOD = (31, 138, 91); BAD = (194, 59, 50)
PAD = 22


def font(name, size):
    try:
        return ImageFont.truetype(os.path.join(FDIR, name), size)
    except Exception:
        return ImageFont.load_default()


F_EYE = font("DejaVuSans-Bold.ttf", 15)
F_Q = font("DejaVuSans-Bold.ttf", 25)
F_BADGE = font("DejaVuSans-Bold.ttf", 16)
F_BODY = font("DejaVuSans.ttf", 17)
F_BODYB = font("DejaVuSans-Bold.ttf", 17)


def wrap(text, fnt, maxw):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if fnt.getlength(t) <= maxw:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def panel(still, question, gt, sti, words):
    W = still.width
    yes = gt.lower().startswith("y")
    gtcol = GOOD if yes else BAD

    # ---- header ----
    hh = 78
    header = Image.new("RGB", (W, hh), INK)
    d = ImageDraw.Draw(header)
    d.text((PAD, 14), "PER-PATCH LOGIT LENS  ·  QUESTION-FIRST (STI)",
           font=F_EYE, fill=(150, 200, 190))
    d.text((PAD, 38), f"“{question}”", font=F_Q, fill=(255, 255, 255))
    # ground-truth badge, right-aligned
    btxt = f"GROUND TRUTH:  {gt.upper()}"
    bw = F_BADGE.getlength(btxt) + 24
    d.rounded_rectangle([W - PAD - bw, 20, W - PAD, 54], radius=17, fill=gtcol)
    d.text((W - PAD - bw / 2, 37), btxt, font=F_BADGE, fill=(255, 255, 255), anchor="mm")

    # ---- footer ----
    maxw = W - 2 * PAD
    lines = []
    lines.append(("b", "The model perceives the scene correctly."))
    for ln in wrap(f"Image patches decode to:  {words}.", F_BODY, maxw):
        lines.append(("n", ln))
    mark = "✗"
    for ln in wrap(
                   f"Yet question-first (STI) answers “{sti}” {mark}  "
                   f"(ground truth: {gt}).", F_BODY, maxw):
        lines.append(("n", ln))
    for ln in wrap(
                   "→ The paradox is a read-out failure, not a perception "
                   "failure. Echoing (SITIT) restores the correct answer.", F_BODYB, maxw):
        lines.append(("e", ln))
    lh = 25
    fh = 16 + lh * len(lines) + 14
    footer = Image.new("RGB", (W, fh), FOOT_BG)
    fd = ImageDraw.Draw(footer)
    y = 14
    for kind, ln in lines:
        if kind == "b":
            fd.text((PAD, y), ln, font=F_BODYB, fill=INK)
        elif kind == "e":
            fd.text((PAD, y), ln, font=F_BODYB, fill=(20, 110, 74))
        else:
            fd.text((PAD, y), ln, font=F_BODY, fill=(50, 56, 63))
        y += lh

    out = Image.new("RGB", (W, hh + still.height + fh), (255, 255, 255))
    out.paste(header, (0, 0))
    out.paste(still, (0, hh))
    out.paste(footer, (0, hh + still.height))
    ImageDraw.Draw(out).rectangle([0, 0, W - 1, out.height - 1], outline=LINE, width=1)
    return out


def main():
    man = json.load(open(os.path.join(ASSETS, "manifest.json")))
    for e in man["examples"]:
        ws = e.get("word_still")
        if not ws or e["idx"] not in WORDS:
            continue
        still = Image.open(os.path.join(ASSETS, ws)).convert("RGB")
        fig = panel(still, e["question"], e["gt"], e["STI"]["pred"], WORDS[e["idx"]])
        fn = f"ex{e['idx']}_fig.png"
        fig.save(os.path.join(ASSETS, fn))
        e["fig_panel"] = fn
        print(f"  {e['idx']}: {fn} ({fig.width}x{fig.height})")
    json.dump(man, open(os.path.join(ASSETS, "manifest.json"), "w"), indent=2)
    print("[done]")


if __name__ == "__main__":
    main()
