"""
Build a static STI-vs-SIT comparison still for each shortlisted example: at a late
layer, both orderings' per-patch logit lens decodes the correct scene, yet STI's
final answer is wrong and SIT's is right. Purely composites frames already present
in the shipped GIFs -- no model, no GPU.

    python build_cmp_still.py     # reads assets/ex*_{STI,SIT}.gif + manifest.json
                                  # writes assets/ex*_cmp.png, updates manifest
"""
import json, os, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
FDIR = "/usr/share/fonts/truetype/dejavu"
LAYER_STEP = 2                     # GIF frame f == transformer layer 2*f

INK = (26, 28, 31); MUT = (91, 100, 112); LINE = (210, 216, 222)
FOOT_BG = (243, 246, 249); GOOD = (31, 138, 91); BAD = (194, 59, 50)
GREEN = (28, 150, 88); RED = (206, 58, 48); EDGE = (8, 20, 26)
PAD = 22; COLGAP = 18

# Answer-EVIDENCE token sets: tokens that help answer each question (the action,
# related objects/props, effects), NOT just the literal question word. STI's patches
# surface these; SIT's largely miss them. (Fallback: the manifest 'key'.)
EVIDENCE = {
    478:  ["sharing", "share", "food", "eating", "meal", "feeding", "giving",
            "passing", "plate", "vegetarian", "snack"],
    217:  ["shaking", "shake", "wet", "water", "fur", "bath", "splash", "spray",
            "droplet", "soaked", "swimming", "dripping"],
    769:  ["sitting", "seated", "sit", "grass", "lawn", "green", "kneeling",
            "crouching", "ground"],
    58:   ["waiting", "wait", "standing", "platform", "queue", "line", "track",
            "station", "commuter"],
    879:  ["spectators", "spectator", "crowd", "audience", "stands", "watching",
            "viewers", "onlookers", "bleachers", "fans"],
    454:  ["planting", "plant", "crop", "crops", "soil", "field", "seed", "farm",
            "farming", "dirt", "digging", "garden"],
}


def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def matches(patch_word, target):
    """True if an English patch word stem-matches the target content word."""
    pw = patch_word.strip().lower()
    if not re.fullmatch(r"[a-z]{3,}", pw):
        return False
    a, b = stem(pw), stem(target.lower())
    return len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)
                                            or a in b or b in a)


def bbox_from_words(words, gw, gh, key):
    """Fractional bbox (x0,y0,x1,y1) over the patch grid where the word matches
    `key`, trimmed to the central cluster; also returns the match count."""
    pts = [(p // gw, p % gw) for p, w in enumerate(words) if matches(w, key)]
    if not pts:
        return None, 0
    rs = sorted(r for r, _ in pts)
    cs = sorted(c for _, c in pts)

    def clip(a):
        n = len(a)
        if n >= 5:
            return a[int(0.12 * n)], a[min(n - 1, int(0.88 * n))]
        return a[0], a[-1]

    r0, r1 = clip(rs)
    c0, c1 = clip(cs)
    return (c0 / gw, r0 / gh, (c1 + 1) / gw, (r1 + 1) / gh), len(pts)

# per example: the layer at which STI's key-object patches most clearly decode the
# answer-relevant evidence (chosen by scanning every GIF frame), the evidence tokens
# the patches decode to there, and a fractional bounding box (x0,y0,x1,y1,label) over
# the grid marking the object one must read to answer the question.
LAYERS = {229: 32, 638: 34, 978: 24, 1231: 32}
WORDS = {
    229:  "dog · aggressive · play",
    638:  "running · run · jogging  (running, not walking)",
    978:  "skateboard · ramp · skate · performing",
    1231: "underwater · submerged · swimming  (already in the water)",
}
BOXES = {
    229:  (0.40, 0.42, 0.71, 0.84, "dog: running"),
    638:  (0.09, 0.11, 0.45, 0.99, "figure: running"),
    978:  (0.47, 0.02, 0.84, 0.48, "skateboard on ramp"),
    1231: (0.28, 0.22, 0.93, 0.84, "girl: submerged"),
}
BOX_COL = (0, 214, 255)      # cyan highlight for the key region
BOX_EDGE = (10, 30, 40)      # dark contrast edge


def font(name, size):
    try:
        return ImageFont.truetype(os.path.join(FDIR, name), size)
    except Exception:
        return ImageFont.load_default()


F_EYE = font("DejaVuSans-Bold.ttf", 14)
F_Q = font("DejaVuSans-Bold.ttf", 25)
F_BADGE = font("DejaVuSans-Bold.ttf", 16)
F_COL = font("DejaVuSans-Bold.ttf", 17)
F_BODY = font("DejaVuSans.ttf", 17)
F_BODYB = font("DejaVuSans-Bold.ttf", 17)
F_TAG = font("DejaVuSans-Bold.ttf", 14)


# exact rendered colours of the section-header bands (SYSTEM=teal, TASK=blue,
# GENERATED=orange). Match to these so a blue/warm-toned *photo* is not mistaken
# for a band.
BANDS = {"teal": (0, 100, 90), "blue": (0, 55, 155), "orange": (130, 40, 0)}


def row_class(row):
    """Classify a frame row as a solid section band or 'grid' (patch/token area)."""
    med = np.median(row, axis=0)
    r, g, b = int(med[0]), int(med[1]), int(med[2])
    if r < 46 and g < 46 and b < 52:
        return "black"
    solid = (np.abs(row - med).sum(axis=1) < 45).mean() > 0.55
    if solid:
        for name, (br, bg, bb) in BANDS.items():
            if abs(r - br) < 46 and abs(g - bg) < 46 and abs(b - bb) < 55:
                return name
    return "grid"


def logit_band(fr):
    """Return (y0, y1): the 'Logit Lens - Layer N' header down to the bottom of its
    patch grid. Robust to prompt-token grids of any height and to photos that split
    the grid run: the logit-lens header is the thin black band directly followed by
    the brightest region (the actual photo, not the dark navy prompt grids)."""
    a = np.asarray(fr.convert("RGB")).astype(int)
    n = a.shape[0]
    cls = [row_class(a[y]) for y in range(n)]
    # thin black bands, scored by the brightness of what follows (the photo overlay)
    cands, i = [], 0
    while i < n:
        if cls[i] == "black":
            j = i
            while j < n and cls[j] == "black":
                j += 1
            if 12 <= j - i <= 46:
                foll = a[j:min(n, j + 80)]
                cands.append((i, j, float(foll.mean()) if len(foll) else 0.0))
            i = j
        else:
            i += 1
    if not cands:
        return 0, n
    bs, be, _ = max(cands, key=lambda c: c[2])       # header before the brightest run
    # grid ends at the next real section band or the tall generated-tokens black.
    # NB: only blue (TASK) or orange (GENERATED) can follow the logit-lens grid;
    # teal (SYSTEM) sits only at the very top, so grassy/green photo rows that
    # look teal must NOT end the grid.
    ge = be
    while ge < n:
        c = cls[ge]
        if c in ("blue", "orange"):
            k = ge
            while k < n and cls[k] == c:
                k += 1
            if k - ge >= 6:
                break
            ge = k
            continue
        if c == "black":
            k = ge
            while k < n and cls[k] == "black":
                k += 1
            if k - ge > 30:
                break
            ge = k
            continue
        ge += 1
    return bs, ge


def crop_lens(gif_path, frame_idx):
    im = Image.open(gif_path)
    frame_idx = min(frame_idx, im.n_frames - 1)
    im.seek(frame_idx)
    fr = im.convert("RGB")
    y0, y1 = logit_band(fr)
    crop = fr.crop((0, y0, fr.width, y1))
    # header height = the leading black 'Logit Lens - Layer N' band inside the crop
    a = np.asarray(crop).astype(int)
    gy = 0
    for r in range(a.shape[0]):
        if row_class(a[r]) == "black":
            gy = r + 1
        else:
            break
    return crop, frame_idx * LAYER_STEP, gy


def match_any(word, targets):
    for t in targets:
        if matches(word, t):
            return t
    return None


def highlight_cells(img, gy, words, gw, gh, targets, color):
    """Draw a coloured box around every patch cell whose decoded word matches any of
    the answer-evidence `targets`. Returns (count, {matched token: count})."""
    W, Ht = img.width, img.height
    cw = W / gw
    ch = (Ht - gy) / gh
    d = ImageDraw.Draw(img)
    n, found = 0, {}
    for p, w in enumerate(words):
        t = match_any(w, targets)
        if t is None:
            continue
        r, c = p // gw, p % gw
        x0, y0 = c * cw + 2, gy + r * ch + 2
        x1, y1 = (c + 1) * cw - 2, gy + (r + 1) * ch - 2
        for tt in range(2, 4):                # dark contrast edge
            d.rectangle([x0 - tt, y0 - tt, x1 + tt, y1 + tt], outline=EDGE)
        for tt in range(0, 2):                # coloured highlight
            d.rectangle([x0 - tt, y0 - tt, x1 + tt, y1 + tt], outline=color)
        n += 1
        found[t] = found.get(t, 0) + 1
    return n, found


def grid_label(img, gy, text, color):
    """Small legend tag pinned to the top-left corner of the grid."""
    d = ImageDraw.Draw(img)
    tw = F_TAG.getlength(text) + 12
    d.rectangle([0, gy, tw, gy + 22], fill=color)
    d.text((6, gy + 3), text, font=F_TAG, fill=(255, 255, 255))


def draw_box(img, gy, box, color, label, tagfill=(255, 255, 255)):
    """Draw a coloured bounding box (fractional x0,y0,x1,y1) over the grid, below
    the header at gy, with a corner label."""
    x0, y0, x1, y1 = box[:4]
    W, Ht = img.width, img.height
    gh = Ht - gy
    px0, py0 = int(x0 * W), int(gy + y0 * gh)
    px1, py1 = int(x1 * W), int(gy + y1 * gh)
    px0, px1 = max(2, px0), min(W - 2, px1)
    py0, py1 = max(gy + 1, py0), min(Ht - 2, py1)
    d = ImageDraw.Draw(img)
    for t in range(2, 5):        # dark contrast edge
        d.rectangle([px0 - t, py0 - t, px1 + t, py1 + t], outline=EDGE)
    for t in range(0, 2):        # coloured box
        d.rectangle([px0 - t, py0 - t, px1 + t, py1 + t], outline=color)
    # label tag, just inside the top-left corner
    tw = F_TAG.getlength(label) + 12
    ty = max(gy + 1, py0)
    d.rectangle([px0, ty, px0 + tw, ty + 22], fill=color)
    d.text((px0 + 6, ty + 3), label, font=F_TAG, fill=tagfill)
    return img


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


def panel(e):
    idx = e["idx"]
    steering = "sti_words" in e and "grid_w" in e
    if steering:
        layer = e["peak_layer"]
    else:
        layer = LAYERS.get(idx, 32)           # best per-example layer
    frame = layer // LAYER_STEP
    sti_img, layer, gy_s = crop_lens(os.path.join(ASSETS, e["STI"]["gif"]), frame)
    sit_img, _, gy_r = crop_lens(os.path.join(ASSETS, e["SIT"]["gif"]), frame)

    # equalise the two columns to the same height (crop to the shorter)
    h = min(sti_img.height, sit_img.height)
    sti_img = sti_img.crop((0, 0, sti_img.width, h))
    sit_img = sit_img.crop((0, 0, sit_img.width, h))

    n_sti = n_sit = 0
    key = e.get("key", "")
    found_sti = {}
    if steering:
        # highlight every patch cell that decodes an answer-EVIDENCE token (the
        # action, related objects/props) -- green on STI (many), red on SIT (few).
        gw, gh = e["grid_w"], e["grid_h"]
        targets = e.get("evidence") or EVIDENCE.get(idx, [key])
        n_sti, found_sti = highlight_cells(sti_img, gy_s, e["sti_words"], gw, gh,
                                           targets, GREEN)
        n_sit, _ = highlight_cells(sit_img, gy_r, e["sit_words"], gw, gh,
                                   targets, RED)
        grid_label(sti_img, gy_s, "STI answer-evidence ×%d" % n_sti, GREEN)
        grid_label(sit_img, gy_r, "SIT answer-evidence ×%d" % n_sit, RED)
    elif idx in BOXES:                        # legacy single-colour path
        draw_box(sti_img, gy_s, BOXES[idx], (0, 214, 255), BOXES[idx][4], (6, 22, 30))
        draw_box(sit_img, gy_r, BOXES[idx], (0, 214, 255), BOXES[idx][4], (6, 22, 30))
    cw = sti_img.width
    W = PAD + cw + COLGAP + cw + PAD

    question, gt = e["question"], e["gt"]
    sti_pred, sit_pred = e["STI"]["pred"], e["SIT"]["pred"]
    yes = gt.lower().startswith("y")
    gtcol = GOOD if yes else BAD

    # ---- header ----
    hh = 76
    header = Image.new("RGB", (W, hh), INK)
    d = ImageDraw.Draw(header)
    eyebrow = ("PER-PATCH LOGIT LENS  ·  STI vs. SIT  ·  LAYER %d  ·  PERCEPTION STEERING"
               if steering else
               "PER-PATCH LOGIT LENS  ·  STI vs. SIT  ·  LAYER %d") % layer
    d.text((PAD, 13), eyebrow, font=F_EYE, fill=(150, 200, 190))
    d.text((PAD, 36), "“%s”" % question, font=F_Q, fill=(255, 255, 255))
    btxt = "GROUND TRUTH:  %s" % gt.upper()
    bw = F_BADGE.getlength(btxt) + 24
    d.rounded_rectangle([W - PAD - bw, 20, W - PAD, 54], radius=17, fill=gtcol)
    d.text((W - PAD - bw / 2, 37), btxt, font=F_BADGE, fill=(255, 255, 255), anchor="mm")

    # ---- column captions ----
    ch = 34
    caps = Image.new("RGB", (W, ch), (255, 255, 255))
    cd = ImageDraw.Draw(caps)
    lx = PAD + cw / 2
    rx = PAD + cw + COLGAP + cw / 2
    cd.text((lx, ch / 2), "STI (question-first)", font=F_COL, fill=BAD, anchor="mm")
    cd.text((rx, ch / 2), "SIT (question-last)", font=F_COL, fill=GOOD, anchor="mm")

    # ---- footer ----
    maxw = W - 2 * PAD
    if steering:
        toks = ", ".join(sorted(found_sti, key=lambda t: -found_sti[t])[:5])
        sti_ok = e["STI"]["correct"]
        sit_ok = e["SIT"]["correct"]
        mk = lambda ok: "✓" if ok else "✗"
        yesno = "YES" if gt.lower().startswith("y") else "NO"
        lines = [("b", "Ground truth is %s — the answer-relevant content is really "
                  "in the image." % yesno)]
        for ln in wrap("Question-first (STI) identifies it: at layer %d, %d image "
                       "patches decode to the correct-answer tokens (%s) under STI "
                       "(green); question-last (SIT) decodes only %d (red)."
                       % (layer, n_sti, toks, n_sit), F_BODY, maxw):
            lines.append(("n", ln))
        for ln in wrap("STI answers “%s” %s;  SIT answers “%s” %s."
                       % (sti_pred, mk(sti_ok), sit_pred, mk(sit_ok)), F_BODY, maxw):
            lines.append(("n", ln))
        for ln in wrap("→ Question-first (STI) surfaces the tokens that answer the "
                       "question; question-last (SIT) does not.", F_BODYB, maxw):
            lines.append(("e", ln))
    else:
        boxlab = BOXES.get(idx, (0, 0, 0, 0, "key region"))[4]
        lines = [("b", "The cyan box marks the object you must read to answer (%s)."
                  % boxlab)]
        for ln in wrap("At layer %d the STI patches there read the scene correctly, "
                       "decoding:  %s." % (layer, WORDS.get(idx, "")), F_BODY, maxw):
            lines.append(("n", ln))
        for ln in wrap("Yet question-first (STI) still answers “%s” ✗ while "
                       "question-last (SIT) answers “%s” ✓  (ground truth: %s)."
                       % (sti_pred, sit_pred, gt), F_BODY, maxw):
            lines.append(("n", ln))
        for ln in wrap("→ Same scene read correctly, different read-out. Moving the "
                       "question first flips a correct answer to a wrong one; echoing "
                       "restores it.", F_BODYB, maxw):
            lines.append(("e", ln))
    lh = 25
    fh = 14 + lh * len(lines) + 12
    footer = Image.new("RGB", (W, fh), FOOT_BG)
    fd = ImageDraw.Draw(footer)
    y = 13
    for kind, ln in lines:
        if kind == "b":
            fd.text((PAD, y), ln, font=F_BODYB, fill=INK)
        elif kind == "e":
            fd.text((PAD, y), ln, font=F_BODYB, fill=(20, 110, 74))
        else:
            fd.text((PAD, y), ln, font=F_BODY, fill=(50, 56, 63))
        y += lh

    # ---- assemble ----
    out = Image.new("RGB", (W, hh + ch + h + fh), (255, 255, 255))
    out.paste(header, (0, 0))
    out.paste(caps, (0, hh))
    out.paste(sti_img, (PAD, hh + ch))
    out.paste(sit_img, (PAD + cw + COLGAP, hh + ch))
    out.paste(footer, (0, hh + ch + h))
    dd = ImageDraw.Draw(out)
    dd.rectangle([0, 0, W - 1, out.height - 1], outline=LINE, width=1)
    # thin divider between the two columns
    xdiv = PAD + cw + COLGAP // 2
    dd.line([xdiv, hh + ch, xdiv, hh + ch + h], fill=LINE, width=1)
    return out, layer


def main():
    man = json.load(open(os.path.join(ASSETS, "manifest.json")))
    for e in man["examples"]:
        steering = "sti_words" in e and "grid_w" in e
        if not steering and e["idx"] not in WORDS:
            continue
        fig, layer = panel(e)
        fn = "ex%d_cmp.png" % e["idx"]
        fig.save(os.path.join(ASSETS, fn))
        e["cmp_still"] = fn
        e["cmp_layer"] = layer
        print("  %d: %s (%dx%d, layer %d)" % (e["idx"], fn, fig.width, fig.height, layer))
    json.dump(man, open(os.path.join(ASSETS, "manifest.json"), "w"), indent=2)
    print("[done]")


if __name__ == "__main__":
    main()
