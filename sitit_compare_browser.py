"""
Streamlit page: SITIT vs STIT — per-layer logit-lens GIF comparison on NaturalBench
(Qwen3-VL-8B). Read-only; shows the GIFs + generated tokens produced by
sitit_stit_gif_gen.py (sitit_compare/manifest.json + ex*.gif). Each GIF animates the
logit lens through every model layer (vision heatmap + token grid), so you can watch
where SITIT and STIT diverge across depth.
"""
import json
import os

import streamlit as st

DIR = "sitit_compare"
GEN_CMD = "CUDA_VISIBLE_DEVICES=0 python sitit_stit_gif_gen.py --num 40 --layer-step 1"


def _load():
    p = os.path.join(DIR, "manifest.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _frames(path, mtime, maxw=560):
    """Decode a (large) GIF once into a list of down-scaled RGB frames. Cached per
    file so the per-layer slider is snappy and never embeds the whole multi-MB GIF."""
    from PIL import Image, ImageSequence
    out = []
    im = Image.open(path)
    for f in ImageSequence.Iterator(im):
        fr = f.convert("RGB")
        if fr.width > maxw:
            fr = fr.resize((maxw, int(fr.height * maxw / fr.width)), Image.LANCZOS)
        out.append(fr)
    return out


def render_sitit_compare_page():
    st.subheader("🎞️ SITIT vs STIT — per-layer logit lens")
    st.caption(
        "For NaturalBench (image, question) examples, the same query is run under "
        "**SITIT** (S·I·T·I·T — image repeated, question after each) and **STIT** "
        "(S·T·I·T). Each GIF animates the logit lens through **every layer** — the "
        "vision heatmap (what each image patch decodes to) and the token grid "
        "(what every token, including the generated answer, decodes to) — so you can "
        "watch where the two orderings diverge with depth."
    )

    man = _load()
    if not man or not man.get("examples"):
        st.info(f"**No GIFs yet.** Generate with (one free GPU, ~all layers):\n\n`{GEN_CMD}`")
        return

    ex = man["examples"]
    layers = man.get("layers", [])

    # ── summary of agreement ───────────────────────────────────────────────────
    both = sum(1 for e in ex if e["orders"].get("SITIT", {}).get("correct")
               and e["orders"].get("STIT", {}).get("correct"))
    s_only = sum(1 for e in ex if e["orders"].get("SITIT", {}).get("correct")
                 and not e["orders"].get("STIT", {}).get("correct"))
    t_only = sum(1 for e in ex if not e["orders"].get("SITIT", {}).get("correct")
                 and e["orders"].get("STIT", {}).get("correct"))
    neither = len(ex) - both - s_only - t_only
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("both correct", both)
    c2.metric("SITIT-only correct", s_only)
    c3.metric("STIT-only correct", t_only)
    c4.metric("neither", neither)
    st.caption(f"{len(ex)} examples · {len(layers)} layers per GIF "
               f"(layers {layers[0]}–{layers[-1]} step {layers[1]-layers[0] if len(layers)>1 else 1})")

    # ── pick an example ────────────────────────────────────────────────────────
    def _lbl(e):
        s = "✓" if e["orders"].get("SITIT", {}).get("correct") else "✗"
        t = "✓" if e["orders"].get("STIT", {}).get("correct") else "✗"
        return f"ex{e['idx']:02d}  SITIT {s} · STIT {t}  — {e['question'][:44]}"

    # default to a divergent example if any
    order_idx = sorted(range(len(ex)), key=lambda i: 0 if (
        ex[i]["orders"].get("SITIT", {}).get("correct")
        != ex[i]["orders"].get("STIT", {}).get("correct")) else 1)
    pick = st.selectbox("Example", options=order_idx, format_func=lambda i: _lbl(ex[i]))
    e = ex[pick]

    st.markdown(f"**Q{e['group']}:** {e['question']}  ·  **gt = {e['gt']}**  ·  `{e['image']}`")

    # per-layer slider (the full GIFs are 20-40 MB — too heavy to embed; we extract
    # one frame per layer instead, which is also a cleaner "see each layer" view).
    layers = man.get("layers", list(range(36)))
    li = st.select_slider(
        "Layer  (drag to scrub through depth)", options=list(range(len(layers))),
        value=min(len(layers) - 1, len(layers) // 2 + 4),
        format_func=lambda i: f"layer {layers[i]}", key=f"layer_{e['idx']}")
    show_gif = st.toggle("▶ play full animation instead (large — may load slowly)",
                         value=False, key=f"anim_{e['idx']}")

    cols = st.columns(2)
    for col, order in zip(cols, ("SITIT", "STIT")):
        o = e["orders"].get(order, {})
        with col:
            ok = o.get("correct")
            st.markdown(f"#### {order} — answer: **{o.get('answer','?')}** "
                        f"{'✅' if ok else '❌'}")
            gif = os.path.join(DIR, o.get("gif", ""))
            if not os.path.exists(gif):
                st.caption("(gif missing)")
                continue
            if show_gif:
                st.image(gif, use_container_width=True)
            else:
                frames = _frames(gif, os.path.getmtime(gif))
                st.image(frames[min(li, len(frames) - 1)], use_container_width=True,
                         caption=f"{order} · layer {layers[min(li, len(layers)-1)]}")
    st.caption("Slider shows the logit lens at one layer for both orderings "
               "(SITIT left, STIT right); toggle above to play the full multi-layer "
               "animation. SITIT stacks IMAGE 1 · TASK · IMAGE 2 · TASK · GENERATED.")
    st.markdown("---")
    st.caption(f"Regenerate / extend:  `{GEN_CMD}`")
