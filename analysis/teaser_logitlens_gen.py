"""Teaser: logit-lens zoomed insets on the TV and Cat regions showing that
question-first (STI) steers the visual readout by the question while image-first
(SIT) does not (the image tokens cannot attend to the later question, so the
readout is question-invariant).

For orderings {STI, SIT} and questions {q1=sport, q2=cat} we compute the per-patch
logit-lens concept probability (cat-words, sport/TV-words) at layer 28, crop the two
object ROIs, and report the concept mass on each ROI. SIT q1 and q2 are identical by
construction; STI differ. Builds teaser_logitlens.png.
"""
import os
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core.constants import SYSTEM_MESSAGE
from core.model_manager import ModelManager
from core.utils import setup_seeds, disable_torch_init
from transformers.utils import logging as hl; hl.set_verbosity_error()

OUT = "patchcos_contrast"; DISP_W, ALPHA, CMAP, LAYER = 620, 0.6, "turbo", 28
IMG = "examples/17226816-32fb-4dbc-8798-3c90d4f1be1b.png"
CONCEPTS = {
    "cat": ["cat","cats","kitten","kitty","pet","pets","feline","猫","猫咪","宠物","小猫"],
    "sport/TV": ["football","soccer","player","players","match","game","stadium","crowd",
                 "audience","sport","sports","television","球员","比赛","球场","人群","观众","直播","记者","足球"],
}
ROI = {"cat": (19, 25, 30, 45), "sport/TV": (3, 21, 3, 27)}   # grid rows/cols
QS = [("q1", "What sport is on TV?"), ("q2", "What is the cat doing?")]
ORDERS = ["STI", "SIT"]


def find_norm(mm):
    for path in ("model.language_model.norm","model.model.norm","model.norm"):
        obj = mm.llm_model
        try:
            for p in path.split("."): obj = getattr(obj, p)
            return obj
        except AttributeError: continue
    return torch.nn.Identity()


def cids(tok, words):
    ids = set()
    for w in words:
        for surf in (w, " "+w):
            t = tok(surf, add_special_tokens=False).input_ids
            if len(t) == 1: ids.add(t[0])
    return sorted(ids)


def concept_maps(mm, pil, q, order, norm, lm_head, CID):
    sys = "" if order == "I" else SYSTEM_MESSAGE
    _, ids, kw = mm.prepare_inputs_from_pil([q], pil, system_prompt=sys, order=order)
    gh, gw, s, e = mm.grid_h, mm.grid_w, mm.img_start_idx, mm.img_end_idx
    with torch.inference_mode():
        out = mm.llm_model(ids, output_hidden_states=True, use_cache=False, **kw)
    h = out.hidden_states[LAYER][0, s:e]
    probs = torch.softmax(lm_head(norm(h.to(lm_head.weight.dtype))).float(), -1)
    P = gh*gw
    return {c: probs[:P, CID[c]].sum(-1).reshape(gh, gw).detach().cpu().numpy()
            for c in CONCEPTS}, gh, gw


def smooth(g, k=1):
    out = g.copy()
    for _ in range(k):
        p = np.pad(out, 1, mode="edge")
        out = (p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]+4*out)/8.0
    return out


def main():
    setup_seeds(); disable_torch_init()
    pil = Image.open(IMG).convert("RGB")
    disp = pil.resize((DISP_W, int(pil.height*DISP_W/pil.width)), Image.LANCZOS)
    Wd, Hd = disp.size
    mm = ModelManager("qwen3-vl-8b", attn_implementation="sdpa")
    norm, lm_head, tok = find_norm(mm), mm.llm_model.lm_head, mm.tokenizer
    CID = {c: cids(tok, w) for c, w in CONCEPTS.items()}

    M = {}                          # M[order][qid] = {concept: grid}
    gh = gw = None
    for order in ORDERS:
        M[order] = {}
        for qid, qt in QS:
            M[order][qid], gh, gw = concept_maps(mm, pil, qt, order, norm, lm_head, CID)

    # ROI concept mass table + SIT-invariance check
    def roimass(order, qid, concept):
        g = M[order][qid][concept]; r0,r1,c0,c1 = ROI[concept]
        return float(g[r0:r1, c0:c1].mean())
    print("\n[ROI mass]  region:concept   STI-q1  STI-q2   SIT-q1  SIT-q2")
    for concept in CONCEPTS:
        row = [roimass("STI","q1",concept), roimass("STI","q2",concept),
               roimass("SIT","q1",concept), roimass("SIT","q2",concept)]
        print(f"  {concept:9s}  " + "  ".join(f"{v:.4f}" for v in row)
              + f"   | SIT q1==q2: {np.isclose(row[2],row[3])}")

    # px ROI boxes
    def pxbox(concept):
        r0,r1,c0,c1 = ROI[concept]
        return (c0/gw*Wd, r0/gh*Hd, c1/gw*Wd, r1/gh*Hd)

    def overlay_crop(order, qid, concept):
        g = np.clip(smooth(M[order][qid][concept],1)/VMAX[concept], 0, 1)
        hm = plt.get_cmap(CMAP)(g)[:,:,:3]
        hm = Image.fromarray((hm*255).astype(np.uint8)).resize(disp.size, Image.BICUBIC)
        blend = Image.blend(disp, hm, ALPHA)
        x0,y0,x1,y1 = pxbox(concept)
        return blend.crop((int(x0),int(y0),int(x1),int(y1)))

    VMAX = {c: (np.percentile(np.concatenate([smooth(M[o][q][c],1).ravel()
            for o in ORDERS for q,_ in QS]), 99) or 1e-6) for c in CONCEPTS}

    # ---- figure ----
    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.75, 1.1], hspace=0.34, wspace=0.16)
    axi = fig.add_subplot(gs[0, 0]); axi.imshow(disp)
    for concept,(col,lab) in (("cat",("red","Cat")),("sport/TV",("deepskyblue","TV"))):
        x0,y0,x1,y1 = pxbox(concept)
        axi.add_patch(mpatches.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=col,lw=2.5))
        axi.text(x0, y0-5, lab, color=col, fontsize=12, fontweight="bold")
    axi.set_title("q1: “What sport is on TV?”   q2: “What is the cat doing?”",
                  fontsize=9); axi.axis("off")
    axt = fig.add_subplot(gs[1, 0]); axt.axis("off")
    axt.text(0.0, 0.98,
             "Per-patch logit-lens (layer 28):\n"
             "probability a patch puts on cat- /\n"
             "sport-words. Insets zoom the ROIs.", fontsize=8.5, va="top")
    axt.text(0.0, 0.52,
             "SIT (image-first): the image cannot\n"
             "see the later question, so the read-\n"
             "out is identical for any question\n"
             "(ΔP = 0, frozen).",
             fontsize=8.7, va="top", color="#2a8a4a")
    axt.text(0.0, 0.10,
             "STI (question-first): the question\n"
             "precedes the image and moves the\n"
             "read-out (ΔP > 0, question-sensitive).",
             fontsize=8.7, va="top", color="#c0392b")

    regions = [("cat", "Cat region", "red"), ("sport/TV", "TV region", "deepskyblue")]
    for r,(concept,rlab,rc) in enumerate(regions):
        # one 4-wide strip: STI q1|q2  <gap>  SIT q1|q2
        axs = fig.add_subplot(gs[r, 1])
        crops = {o:{q: overlay_crop(o,q,concept) for q in ("q1","q2")} for o in ORDERS}
        w,h = crops["STI"]["q1"].size; gap, mid = 5, 26
        strip = Image.new("RGB",(w*4+gap*2+mid, h),(255,255,255))
        strip.paste(crops["STI"]["q1"],(0,0)); strip.paste(crops["STI"]["q2"],(w+gap,0))
        strip.paste(crops["SIT"]["q1"],(2*w+gap+mid,0)); strip.paste(crops["SIT"]["q2"],(3*w+2*gap+mid,0))
        axs.imshow(strip); axs.axis("off")
        axs.text(0.25,1.07,"STI:  q1 ≠ q2  (steered)", transform=axs.transAxes,
                 ha="center", fontsize=9.5, color="#c0392b", fontweight="bold")
        axs.text(0.76,1.07,"SIT:  q1 = q2  (frozen)", transform=axs.transAxes,
                 ha="center", fontsize=9.5, color="#2a8a4a", fontweight="bold")
        axs.text(-0.03,0.5,rlab,transform=axs.transAxes,rotation=90,va="center",ha="right",
                 fontsize=10,color=rc,fontweight="bold")
        # bars: question-effect magnitude |P(q1)-P(q2)| — STI>0 (question-sensitive),
        # SIT==0 exactly (frozen). We plot the magnitude of the change, not its sign,
        # because the sign is not robust on a single image; the robust fact is that
        # STI's read-out moves with the question and SIT's cannot.
        axb = fig.add_subplot(gs[r, 2])
        dST = abs(roimass("STI","q1",concept) - roimass("STI","q2",concept))
        dSI = abs(roimass("SIT","q1",concept) - roimass("SIT","q2",concept))
        axb.bar(["STI","SIT"], [dST, dSI], color=["#c0392b", "#2a8a4a"], width=0.6)
        axb.annotate("question-\nsensitive", (0, dST), textcoords="offset points",
                     xytext=(0,5), ha="center", fontsize=8, color="#c0392b", fontweight="bold")
        axb.annotate("frozen\n(=0)", (1, dSI), textcoords="offset points",
                     xytext=(0,5), ha="center", fontsize=8, color="#2a8a4a", fontweight="bold")
        axb.set_title(f"|ΔP({concept})| between q1 and q2", fontsize=8.5)
        axb.tick_params(labelsize=8); axb.grid(axis="y", alpha=0.3); axb.margins(y=0.35)
    fig.suptitle("Question-first (STI) makes the visual read-out question-dependent; "
                 "image-first (SIT) freezes it ($\\Delta$P = 0)", fontsize=12.5, y=0.99)
    out = f"{OUT}/teaser_logitlens.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
    print("[saved]", out)


if __name__ == "__main__":
    main()
