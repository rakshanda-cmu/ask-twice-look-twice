"""
Streamlit page: per-layer logit-lens GIFs for ONE image, side by side, for the two
questions × two orderings (q1/q2 × IST/STI) produced by logitlens_demo_gen.py. Each
GIF animates the logit lens through every layer — the image-patch heatmap plus the
text / GENERATED token grid (the model's answer decoded at each layer). The raw GIFs
are ~14 MB each, so by default we scrub a per-layer frame slider (all four side by
side) and offer a toggle to play the full animations.
"""
import json
import os

import streamlit as st

DIR = "logitlens_demo"
GEN_CMD = ("CUDA_VISIBLE_DEVICES=0 python logitlens_demo_gen.py "
           "--image examples/<img>.png")


@st.cache_data(show_spinner=False)
def _frames(path, mtime, maxw=380):
    """Decode a (large) GIF once into down-scaled RGB frames, cached per file."""
    from PIL import Image, ImageSequence
    out = []
    im = Image.open(path)
    for f in ImageSequence.Iterator(im):
        fr = f.convert("RGB")
        if fr.width > maxw:
            fr = fr.resize((maxw, int(fr.height * maxw / fr.width)), Image.LANCZOS)
        out.append(fr)
    return out


def render_logitlens_demo_page():
    st.subheader("🔬 Logit lens (this image) — IST vs STI, 2 questions")
    st.caption(
        "Per-layer **logit lens** on the shared example image, Qwen3-VL-8B. Each panel "
        "animates through **every layer** — the image-patch heatmap (what each patch "
        "decodes to) and the token grid, **including the model's generated answer** "
        "decoded at each layer. Four panels side by side: the two questions "
        "(*What sport is on TV?* / *What is the cat doing?*) × the two orderings "
        "(**IST** = image first · **STI** = question first). Drag the layer slider to "
        "scrub depth across all four at once; toggle to play the full GIFs."
    )
    _render_ll_panels(DIR, "full resolution", "full")

    # ── same logit lens at HALF resolution (added below; full-res above unchanged) ─
    st.markdown("---")
    st.markdown("### Half resolution (image down-scaled 0.5× before the model)")
    st.caption(
        "The **same** four panels, but the input image is first **down-scaled to half "
        "resolution** (fewer image-patch tokens, coarser heatmap grid). Lets you see "
        "whether the answer still resolves at the same depth with less visual detail."
    )
    _render_ll_panels("logitlens_demo_half", "half resolution", "half")


def _render_ll_panels(dir_, tag, key):
    """Render the 4 (question × order) logit-lens panels from <dir_>/manifest.json."""
    mp = os.path.join(dir_, "manifest.json")
    if not os.path.exists(mp):
        cmd = GEN_CMD + (" --scale 0.5 --out-dir logitlens_demo_half" if key == "half" else "")
        st.info(f"**No {tag} logit-lens GIFs yet.** Generate with (one free GPU):\n\n`{cmd}`")
        return
    man = json.load(open(mp))
    layers = man.get("layers", list(range(36)))
    cells = man.get("cells", {})
    questions = man.get("questions", [])
    orders = man.get("orders", ["IST", "STI"])

    li = st.select_slider(
        "Layer  (drag to scrub through depth — applies to all four panels)",
        options=list(range(len(layers))),
        value=min(len(layers) - 1, len(layers) // 2 + 4),
        format_func=lambda i: f"layer {layers[i]}", key=f"ll_layer_{key}")
    show_gif = st.toggle("▶ play full animations instead (large — may load slowly)",
                         value=False, key=f"ll_anim_{key}")

    panels = [(q, o) for q in questions for o in orders]   # 4 panels side by side
    cols = st.columns(len(panels))
    for col, (q, o) in zip(cols, panels):
        cell = cells.get(q["id"], {}).get(o, {})
        with col:
            st.markdown(f"**{o}** · “{q['text']}”")
            ans = cell.get("answer", "")
            if ans:
                st.caption(f"answer: *{ans}*")
            gif = os.path.join(dir_, cell.get("gif", ""))
            if not os.path.exists(gif):
                st.caption("(gif missing)")
                continue
            if show_gif:
                st.image(gif, use_container_width=True)
            else:
                frames = _frames(gif, os.path.getmtime(gif))
                st.image(frames[min(li, len(frames) - 1)], use_container_width=True,
                         caption=f"layer {layers[min(li, len(layers)-1)]}")
    sz = man.get("img_size")
    st.caption((f"Input {sz[0]}×{sz[1]}px · " if sz else "")
               + f"{len(layers)} layers.  Regenerate:  `{GEN_CMD}"
               + (" --scale 0.5 --out-dir logitlens_demo_half" if key == "half" else "") + "`")
