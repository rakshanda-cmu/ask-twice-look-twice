"""
Build the project page (index.html) in the arXiv-HTML (ar5iv) paper
style: serif body, centered title/authors/abstract, numbered sections, booktabs
tables ("Table N:") and "Figure N:" captions.

Content is intentionally limited to the *primary* orderings and results:
    STI  (question-first, the paradox)
    SIT  (question-last, baseline)
    STIT (question echo, ours)
    SITIT     (image echo, ours -- best)
    SITIT_rev (image echo, reversed 2nd copy -- "image reversal")

    python build_html.py     # writes ../index.html, reads assets/manifest.json
                             #                       and ../<dataset>/results/*.json
"""
import json, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "index.html")
SELECT = [1264, 22, 33, 373, 994, 189, 232, 1007, 1077]
# Section 4 (the fix) needs cases where STI errs and image-echoing (SITIT) recovers
# the correct answer -- a different, fix-valid subset from the perception panels.
SELECT_FIX = [1007, 978, 922, 229]

TITLE = ("Ask Twice, Look Twice: Prompt Echoing Resolves the "
         "Question-First Paradox in Vision-Language Models")

# ---- orderings we report (primary set) --------------------------------------
ORDERS = ["STI", "SIT", "STIT", "SITIT", "SITIT_rev"]
ORDER_HEAD = {"STI": "STI", "SIT": "SIT", "STIT": "STIT",
              "SITIT": "SITIT", "SITIT_rev": "SITIT<sub>rev</sub>"}
LEGEND = [
    ("STI",   "System &middot; Task &middot; Image",
     "question-first &mdash; the paradox"),
    ("SIT",   "System &middot; Image &middot; Task",
     "question-last &mdash; the baseline"),
    ("STIT",  "System &middot; Task &middot; Image &middot; Task",
     "question echo (ours)"),
    ("SITIT", "System &middot; Image &middot; Task &middot; Image &middot; Task",
     "image echo (ours) &mdash; best"),
    ("SITIT<sub>rev</sub>", "System &middot; Image &middot; Task &middot; &#298; &middot; Task",
     "image echo, 2nd copy reversed"),
]

MODELS = [("qwen3-vl-8b", "Qwen3-VL-8B"), ("gemma-3-27b", "Gemma-3-27B")]
BENCHES = [
    ("naturalbench", "NaturalBench", "group acc."),
    ("winoground",   "Winoground",   "group acc."),
    ("pope",         "POPE",         "acc."),
    ("rf20",         "RF20",         "acc."),
]

# curated scene tokens each example's patches decode to (for the running text)
SCENE = {
    229:  "dog, aggressive, play",
    638:  "running, run, jogging",
    978:  "skateboard, ramp, skate, performing",
    1231: "underwater, submerged, swimming",
}


def esc(s):
    return html.escape(str(s))


def metric(ds, meta):
    if ds == "naturalbench":
        return meta.get("g_acc")
    o = meta.get("overall", {})
    if ds == "winoground":
        return o.get("group_acc")
    return o.get("acc")


def load_scores():
    """scores[model][ds][order] = float (or None)."""
    scores = {}
    for model, _ in MODELS:
        scores[model] = {}
        for ds, _, _ in BENCHES:
            scores[model][ds] = {}
            for o in ORDERS:
                p = os.path.join(ROOT, ds, "results",
                                 f"{model}__{o}__results.json")
                if os.path.exists(p):
                    scores[model][ds][o] = metric(ds, json.load(open(p))["meta"])
                else:
                    scores[model][ds][o] = None
    return scores


def result_table(scores, model):
    """One booktabs-style table per model: benchmarks x orderings."""
    head = "".join(f"<th>{ORDER_HEAD[o]}</th>" for o in ORDERS)
    rows = ""
    for ds, name, unit in BENCHES:
        d = scores[model][ds]
        vals = {o: d[o] for o in ORDERS}
        present = [v for v in vals.values() if v is not None]
        best = max(present) if present else None
        # the paradox: STI vs SIT
        cells = ""
        for o in ORDERS:
            v = vals[o]
            if v is None:
                cells += '<td class="na">&ndash;</td>'
                continue
            cls = []
            if best is not None and abs(v - best) < 1e-9:
                cls.append("best")
            if o == "STI":
                cls.append("worst")
            c = f' class="{" ".join(cls)}"' if cls else ""
            cells += f"<td{c}>{v:.3f}</td>"
        rows += (f"<tr><td class='rowname'>{esc(name)}</td>"
                 f"<td class='unit'>{esc(unit)}</td>{cells}</tr>")
    return (f'<div class="table-wrapper"><table class="booktabs">'
            f'<thead><tr><th class="rowname">Benchmark</th><th class="unit">metric</th>'
            f'{head}</tr></thead><tbody>{rows}</tbody></table></div>')


def mark(ok):
    return ('<span class="ok">&#10003;</span>' if ok
            else '<span class="bad">&#10007;</span>')


def lazy_gif(e, key, label, role):
    """One animation column; its <img> loads only when the block is revealed."""
    d = e[key]
    return f"""
        <figure class="giffig">
          <figcaption class="gcap-top">{esc(label)} answers
            &ldquo;{esc(d['pred'])}&rdquo; {mark(d['correct'])}</figcaption>
          <img class="lazygif" data-gif="assets/{esc(d['gif'])}"
               alt="{esc(label)} per-layer logit lens">
        </figure>"""


def qhead(e):
    """The question, shown as a heading, with a ground-truth badge."""
    gtc = "gt-yes" if e["gt"].lower().startswith("y") else "gt-no"
    return (f'<h3 class="qhead">&ldquo;{esc(e["question"])}&rdquo;'
            f'<span class="{gtc}">ground truth: {esc(e["gt"])}</span></h3>')


def qrow(e):
    """Raw image and the question side by side (question wraps if long)."""
    gtc = "gt-yes" if e["gt"].lower().startswith("y") else "gt-no"
    return f"""
      <div class="qrow">
        <button class="rawbtn" type="button" data-full="assets/{esc(e['image'])}"
                aria-label="Enlarge raw image" title="Click to enlarge">
          <img src="assets/{esc(e['image'])}" alt="raw image" loading="lazy">
        </button>
        <h3 class="qhead qside">&ldquo;{esc(e['question'])}&rdquo;
          <span class="{gtc}">ground truth: {esc(e['gt'])}</span></h3>
      </div>"""


def rawthumb(e):
    """Centered, click-to-enlarge raw image."""
    return f"""
      <div class="uhead">
        <button class="rawbtn" type="button" data-full="assets/{esc(e['image'])}"
                aria-label="Enlarge raw image" title="Click to enlarge">
          <img src="assets/{esc(e['image'])}" alt="raw image" loading="lazy">
        </button>
      </div>"""


def heatfig(e):
    """Answer-evidence heatmap figure (STI vs SIT), if present."""
    if not e.get("heatmap"):
        return ""
    return f"""
      <figure class="panelfig heatfig">
        <img src="assets/{esc(e['heatmap'])}" alt="answer-evidence heatmap" loading="lazy">
      </figure>"""


def reveal(e, lkey, llab, lrole, rkey, rlab, rrole, btn):
    """A Play button that expands to two side-by-side animations of one image,
    with a Minimize button to collapse them back."""
    return f"""
      <div class="reveal">
        <button class="revealbtn" type="button">
          <span class="tri">&#9654;</span>&nbsp;{esc(btn)}</button>
        <div class="playarea">
          <div class="gifs two-col">
            {lazy_gif(e, lkey, llab, lrole)}
            {lazy_gif(e, rkey, rlab, rrole)}
          </div>
          {VCBAR}
        </div>
        <button class="minbtn" type="button">
          <span class="tri">&#9650;</span>&nbsp;Minimize</button>
      </div>"""


# Vertical logit-lens confidence scale, shown to the right of the GIFs.
VCBAR = """
      <div class="vcbar" title="logit-lens confidence (per-patch top-token probability)">
        <div class="vcbar-ticks"><span>1</span><span>0.75</span><span>0.5</span>
          <span>0.25</span><span>0</span></div>
        <div class="vcbar-bar"></div>
      </div>"""


def main():
    man = json.load(open(os.path.join(ASSETS, "manifest.json")))
    by = {e["idx"]: e for e in man["examples"]}
    exs = [by[i] for i in SELECT if i in by]
    scores = load_scores()

    legend_rows = "".join(
        f'<tr><td class="ord"><strong>{k}</strong></td><td>{seq}</td>'
        f'<td>{role}</td></tr>' for k, seq, role in LEGEND)

    # Section 3: per image -> static STI screenshot, then Play -> STI vs SIT gifs.
    problem = ""
    fnum = 3
    for i in SELECT:
        e = by.get(i)
        if not e:
            continue
        fig = ""
        still = e.get("cmp_still") or e.get("fig_panel")
        if still:
            fig = f"""
        <figure class="panelfig">
          <img src="assets/{esc(still)}" alt="STI vs. SIT per-patch logit lens" loading="lazy">
        </figure>"""
        problem += f"""
      <div class="unit">
        {qrow(e)}
        {fig}
        {reveal(e, "STI", "STI (question-first)", "commits early, wrong",
                "SIT", "SIT (question-last)", "keeps question access, correct",
                "Play STI vs. SIT for this image")}
      </div>"""

    # Section 4: the fix -> Play -> STI vs SITIT gifs of the same image.
    fix = ""
    for i in SELECT_FIX:
        e = by.get(i)
        if not e:
            continue
        fix += f"""
      <div class="unit">
        {qhead(e)}
        {reveal(e, "STI", "STI (question-first)", "still wrong",
                "SITIT", "SITIT (image echo)", "second look fixes it",
                "Play STI vs. SITIT for this image")}
      </div>"""

    q_tab = result_table(scores, "qwen3-vl-8b")
    g_tab = result_table(scores, "gemma-3-27b")

    # code -> paper map (primary results only)
    codemap = [
        ("Position ladder / paradox (NaturalBench)",
         "naturalbench_eval.py, gemma_eval.py"),
        ("POPE / Winoground / RF20",
         "pope_eval.py, winoground_eval.py, rf20_eval.py"),
        ("Image-echo, reversed 2nd copy (SITIT_rev)",
         "*_sitit_reverse.py, reverse_image_hooks.py"),
        ("Order-aware prompt builder (all orderings)",
         "model_manager.py, constants.py, utils.py"),
        ("Per-layer / per-patch logit lens (the animations)",
         "logit_lens_overlay.py, logit_lens_runner.py, sitit_stit_gif_gen.py"),
        ("Result and figure viewer for every analysis",
         "logit_lens_app.py + *_browser.py"),
    ]
    coderows = "".join(
        f'<tr><td class="rowname">{esc(a)}</td><td><code>{esc(b)}</code></td></tr>'
        for a, b in codemap)

    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(TITLE)}</title>
<style>
  :root {{ --ink:#111; --body:#222; --muted:#555; --rule:#000;
          --link:#b31b1b; --linkalt:#1a0dab; --good:#1a7f37; --bad:#c0392b;
          --page:#fff; --soft:#fafafa; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:"Latin Modern Roman","Times New Roman",Times,serif;
         background:#f6f7f9; color:var(--body); line-height:1.58;
         margin:0; font-size:17px; -webkit-font-smoothing:antialiased; }}
  .paper {{ width:min(1240px, 95vw); margin:2rem auto; padding:3rem 3rem 4rem;
           background:var(--page); border:1px solid #eceef1; border-radius:16px;
           box-shadow:0 4px 24px rgba(20,24,40,.06); }}
  h1.title {{ font-size:1.85rem; line-height:1.25; text-align:center;
             font-weight:700; color:var(--ink); margin:0 0 1.1rem; }}
  .authors {{ text-align:center; font-size:1.05rem; color:var(--ink);
             margin-bottom:.15rem; line-height:1.7; }}
  .authors a {{ color:var(--linkalt); border-bottom:1px solid transparent; }}
  .authors a:hover {{ border-bottom-color:var(--linkalt); text-decoration:none; }}
  .affil {{ text-align:center; font-size:.95rem; color:var(--muted);
           font-style:italic; margin-bottom:1.6rem; }}
  .abstract {{ max-width:46rem; margin:0 auto 1.8rem; }}
  .abstract h2 {{ text-align:center; font-size:1rem; font-variant:small-caps;
                 letter-spacing:.08em; font-weight:700; margin:.4rem 0 .6rem; }}
  .abstract p {{ font-size:.97rem; text-align:justify; margin:.5rem 0;
                max-width:none; }}
  .abstract .abridge {{ font-size:.9rem; color:var(--muted); }}
  .cyan {{ color:#0782a0; font-weight:700; }}
  .boxg {{ color:#1c9658; font-weight:700; }}
  .boxr {{ color:#ce3a30; font-weight:700; }}
  h2.sec {{ font-size:1.35rem; font-weight:700; color:var(--ink);
           margin:2.8rem 0 .9rem; padding-bottom:.35rem; text-align:center;
           border-bottom:2px solid #eef0f3; }}
  h3.sub {{ font-size:1.08rem; font-weight:700; color:var(--ink);
           margin:1.6rem 0 .5rem; text-align:center; }}
  p {{ margin:.7rem auto; text-align:justify; max-width:53rem; }}
  a {{ color:var(--linkalt); text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  code {{ font-family:"Latin Modern Mono",ui-monospace,Menlo,Consolas,monospace;
         font-size:.85em; background:var(--soft); padding:.05em .3em;
         border-radius:3px; }}
  strong {{ color:var(--ink); }}
  .lead {{ border-left:3px solid var(--rule); padding-left:1rem; font-size:1rem; }}

  /* tables */
  .table-wrapper {{ overflow-x:auto; margin:1rem 0; }}
  table {{ border-collapse:collapse; margin:0 auto; font-size:.9rem;
          width:100%; }}
  table caption {{ caption-side:top; text-align:left; font-size:.9rem;
                  color:var(--body); margin-bottom:.5rem; }}
  .booktabs {{ border-top:1.5px solid var(--rule);
              border-bottom:1.5px solid var(--rule); }}
  .booktabs thead th {{ border-bottom:1px solid var(--rule);
                       padding:.4rem .6rem; font-weight:700; text-align:center;
                       color:var(--ink); }}
  .booktabs td {{ padding:.35rem .6rem; text-align:center; color:var(--body); }}
  .booktabs .rowname {{ text-align:left; white-space:nowrap; }}
  .booktabs .unit {{ text-align:left; color:var(--muted); font-style:italic;
                    font-size:.82rem; white-space:nowrap; }}
  .booktabs tbody tr + tr td {{ border-top:1px solid #e7e9ec; }}
  .booktabs tbody tr:hover td {{ background:#f7f8fb; }}
  .best {{ font-weight:700; color:var(--good); }}
  .worst {{ color:var(--bad); }}
  .na {{ color:#aaa; }}
  .ord {{ white-space:nowrap; }}
  .legend td {{ padding:.3rem .6rem; text-align:left; font-size:.9rem; }}
  .legend {{ border-top:1.5px solid var(--rule); border-bottom:1.5px solid var(--rule); }}
  .caption {{ font-size:.85rem; color:var(--muted); text-align:left;
             margin:.4rem auto 0; max-width:53rem; }}

  /* figures */
  figure {{ margin:1.4rem 0; text-align:center; }}
  figcaption {{ font-size:.85rem; color:var(--body); text-align:justify;
               margin:.5rem auto 0; line-height:1.45; max-width:53rem; }}
  .fnum {{ font-weight:700; color:var(--ink); }}
  .panelfig {{ margin:1.2rem auto; }}
  .panelfig img {{ max-width:min(100%, 1002px); border:1px solid #e4e6ea;
                  border-radius:10px; box-shadow:0 2px 12px rgba(20,24,40,.07); }}
  .heatfig {{ margin:1rem auto 1.4rem; }}
  .qhead {{ text-align:center; font-size:1.35rem; font-weight:700; color:var(--ink);
           margin:1.4rem 0 1rem; }}
  .qhead span {{ display:inline-block; font-weight:400; font-size:.85rem;
                margin-left:.6rem; padding:.1em .55em; border-radius:12px;
                vertical-align:middle; font-family:Helvetica,Arial,sans-serif; }}
  /* raw image + question side by side */
  .qrow {{ display:flex; align-items:center; justify-content:center; gap:1.6rem;
          flex-wrap:wrap; margin:1.4rem 0 1.1rem; }}
  .qrow .qside {{ margin:0; text-align:left; max-width:32rem; line-height:1.3; }}
  .qrow .qside span {{ display:block; width:fit-content; margin-left:0;
                      margin-top:.6rem; }}
  .cmp {{ margin:1.6rem 0 1.2rem; }}
  .cmp-q {{ text-align:center; font-size:1.05rem; font-weight:700;
           color:var(--ink); margin:.2rem 0 .3rem; }}
  .cmp-q span {{ display:inline-block; font-weight:400; font-size:.85rem;
                margin-left:.5rem; padding:.05em .5em; border-radius:10px;
                vertical-align:middle; }}
  .gt-yes {{ background:#e6f4ea; color:var(--good); }}
  .gt-no  {{ background:#fdecea; color:var(--bad); }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:1.2rem;
             align-items:start; }}
  .unit {{ margin:1.8rem 0; padding:1.5rem 1.5rem 1.2rem; background:#fcfcfd;
          border:1px solid #e9ebef; border-radius:14px;
          box-shadow:0 1px 4px rgba(20,24,40,.04); }}
  .rawlab {{ margin:0; font-size:.9rem; font-style:italic; color:var(--muted);
            font-family:Helvetica,Arial,sans-serif; }}
  .rawlab span {{ display:inline-block; font-style:normal; font-size:.85rem;
                 margin-left:.5rem; padding:.05em .5em; border-radius:10px;
                 vertical-align:middle; }}
  .reveal {{ text-align:center; margin:.6rem 0 0; }}
  .revealbtn {{ display:inline-flex; align-items:center; gap:.15rem;
               font-family:inherit; font-size:.95rem; font-weight:700;
               color:#fff; background:var(--ink); border:0; cursor:pointer;
               padding:.55rem 1.2rem; border-radius:6px; transition:background .15s; }}
  .revealbtn:hover {{ background:#333; }}
  .reveal.playing .revealbtn {{ display:none; }}
  .minbtn {{ display:none; margin:1rem auto 0; align-items:center; gap:.15rem;
            font-family:inherit; font-size:.9rem; font-weight:700; color:var(--ink);
            background:#eee; border:1px solid #ccc; cursor:pointer;
            padding:.45rem 1.1rem; border-radius:6px; }}
  .minbtn:hover {{ background:#e2e2e2; }}
  .reveal.playing .minbtn {{ display:inline-flex; }}
  .tri {{ font-size:.85rem; line-height:1; }}
  .uhead {{ display:flex; align-items:center; justify-content:center; gap:.9rem;
           flex-wrap:wrap; margin:.4rem 0 .4rem; }}
  .uhead .cmp-q {{ margin:0; }}
  .rawbtn {{ padding:0; border:0; background:none; cursor:zoom-in; line-height:0; }}
  .rawbtn img {{ height:150px; width:auto; border-radius:8px; border:1px solid #bbb;
                box-shadow:0 2px 6px rgba(0,0,0,.18); transition:transform .1s; }}
  .rawbtn:hover img {{ transform:scale(1.06); }}
  .lightbox {{ position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:100;
              display:flex; align-items:center; justify-content:center;
              padding:2rem; cursor:zoom-out; }}
  .lightbox[hidden] {{ display:none; }}
  .lightbox img {{ max-width:92vw; max-height:92vh; border-radius:4px;
                  box-shadow:0 8px 40px rgba(0,0,0,.6); }}
  .giffig {{ margin:0; }}
  .gcap-top {{ font-size:1.05rem; font-weight:700; color:var(--ink);
              text-align:center; line-height:1.3; margin:0 0 .5rem; }}
  .giffig img {{ display:block; width:100%; height:auto; border:1px solid #ccc;
                border-radius:8px; background:#111; }}
  /* play area: the two GIFs with a vertical confidence colorbar on the right */
  .playarea {{ display:none; margin-top:1rem; }}
  .reveal.playing .playarea {{ display:flex; align-items:stretch; gap:1rem;
    justify-content:center; }}
  .gifs {{ flex:1 1 auto; display:grid; grid-template-columns:1fr 1fr; gap:1.2rem;
    min-width:0; }}
  .vcbar {{ flex:0 0 auto; display:flex; flex-direction:row-reverse; gap:.4rem;
    padding-top:.2rem; padding-bottom:.2rem; }}
  .vcbar-bar {{ width:16px; border-radius:8px; border:1px solid #ccd2da;
    background:linear-gradient(to top, #f7fbff, #deebf7, #c6dbef, #9ecae1,
      #6baed6, #4292c6, #2171b5, #08519c, #08306b); }}
  .vcbar-ticks {{ display:flex; flex-direction:column; justify-content:space-between;
    font-size:.75rem; font-family:Helvetica,Arial,sans-serif; color:var(--muted);
    text-align:right; }}

  hr {{ border:0; border-top:1px solid #ddd; margin:2.2rem 0; }}
  .footer {{ font-size:.82rem; color:var(--muted); text-align:justify;
            margin-top:2rem; }}
  @media (max-width:640px) {{
    h1.title {{ font-size:1.45rem; }}
    .two-col {{ grid-template-columns:1fr; }}
    body {{ font-size:16px; }}
  }}
</style></head>
<body><main class="paper">

  <h1 class="title">{esc(TITLE)}</h1>
  <div class="authors">
    <a href="https://rakshanda-cmu.github.io/" target="_blank" rel="noopener">Rakshanda Hassan Abhinandan</a> &emsp;
    <a href="https://www.ri.cmu.edu/ri-faculty/john-galeotti/" target="_blank" rel="noopener">John Galeotti</a> &emsp;
    <a href="https://www.cs.cmu.edu/~deva/" target="_blank" rel="noopener">Deva Ramanan</a> &emsp;
    <a href="https://ggare-cmu.github.io/" target="_blank" rel="noopener">Gautam Rajendrakumar Gare</a></div>
  <div class="affil">Carnegie Mellon University &middot; ECCV&nbsp;2026 Submission</div>

  <section class="abstract">
    <h2>Abstract</h2>
    <p>Where should the question go in a vision-language model (VLM) prompt: before
    the image or after it? Intuition says before: knowing what is asked should tell
    the model where to look. Yet across visual question answering benchmarks,
    question-first prompting consistently underperforms the image-first ordering
    recommended for frontier VLMs, a phenomenon we term the
    <em>question-first paradox</em>.</p>
    <p>A vision-language model reads system, image, and question tokens as one
    sequence and answers from the final position. We show that <em>where the
    question is placed changes the answer</em>. Placing the question <em>before</em>
    the image (question-first, <strong>STI</strong>) makes the model commit early
    to an image-anchored, often wrong answer, even though its intermediate
    representations perceive the scene correctly. Placing the question <em>after</em>
    the image (question-last, <strong>SIT</strong>) recovers the correct answer.
    Re-presenting the image and question after the image (image echoing,
    <strong>SITIT</strong>) resolves the paradox with no training and gives the
    best accuracy across four benchmarks and two model families. We summarize the
    primary results and trace the answer forming <strong>layer by layer</strong>
    with a per-layer logit lens.</p>
  </section>

  <h2 class="sec">1&ensp;The Question-First Paradox</h2>
  <p>A prompt is three sections in token order: the <strong>S</strong>ystem
  message, the <strong>I</strong>mage (many visual tokens), and the
  <strong>T</strong>ask/question. Reordering these sections, with content held
  fixed, changes what the decoder answers. We report five orderings.</p>
  <div class="table-wrapper">
    <table class="legend"><tbody>{legend_rows}</tbody></table>
  </div>
  <p class="caption"><strong>Table 1:</strong> The five prompt orderings reported
  here. <code>&#298;</code> denotes the reversed second image copy.</p>

  <h2 class="sec">2&ensp;Primary Results</h2>
  <p>Group accuracy on NaturalBench and Winoground; answer accuracy on POPE and
  RF20. Question-first (<strong>STI</strong>, in <span class="worst">red</span>)
  is the weakest ordering on the compositional benchmarks; image echoing
  (<strong>SITIT</strong> / <strong>SITIT<sub>rev</sub></strong>) is the strongest
  (best per row in <span class="best">green</span>). Reversing the second image
  copy (<strong>SITIT<sub>rev</sub></strong>) retains almost all of the gain,
  showing the effect comes from a second <em>look</em>, not from copying tokens.</p>

  <h3 class="sub">2.1&ensp;Qwen3-VL-8B</h3>
  {q_tab}
  <p class="caption"><strong>Table 2:</strong> Qwen3-VL-8B. STI trails SIT by
  8&ndash;10 points on NaturalBench and Winoground with identical content; echoing
  recovers and exceeds the baseline.</p>

  <h3 class="sub">2.2&ensp;Gemma-3-27B</h3>
  {g_tab}
  <p class="caption"><strong>Table 3:</strong> Gemma-3-27B (4-bit, single GPU).
  The ordering effect and the echoing fix reproduce in a second model family.</p>

  <h2 class="sec">3&ensp;Perception Steering: STI Identifies the Answer, SIT Misses It</h2>
  <p>Projecting each image patch's hidden state through the output embedding
  decodes it to a vocabulary token; overlaying that token on the patch shows
  <strong>what the model sees</strong>. We choose clear, low-token NaturalBench
  cases <strong>whose answer is Yes</strong> &mdash; the queried thing is genuinely
  in the image &mdash; so a patch that decodes it is <em>correct</em>-answer
  evidence, not a hallucination. In every case, question-first
  (<strong>STI</strong>) steers perception to the question: every patch that decodes
  a <em>correct-answer</em> token &mdash; the action and related objects that
  confirm the answer (racket, glove, climbing, guitar, &hellip;) &mdash; is boxed in
  <span class="boxg">green</span>, while question-last (<strong>SIT</strong>), where
  the question comes after the image and cannot steer, decodes far fewer, boxed in
  <span class="boxr">red</span>.
  <strong>Question-first (STI) identifies the tokens that answer the question;
  question-last (SIT) does not</strong> &mdash; the answer verdict of each ordering
  is printed under its column. Press <em>Play</em> under a screenshot to expand the
  two per-layer animations for that same image side by side: each frame decodes one
  transformer layer.</p>
  {problem}

  <h2 class="sec">4&ensp;The Fix: Image Echoing</h2>
  <p>Better perception is wasted if the answer still reads out wrong. Re-presenting
  the (image, question) after the image (<strong>SITIT</strong>) gives the decoder a
  second, adjacent look and recovers the correct answer, with no training. In each
  case below question-first (<strong>STI</strong>) answers wrong and image-echoing
  (<strong>SITIT</strong>) fixes it. Press <em>Play</em> to compare STI
  (wrong&nbsp;&#10007;) vs.&nbsp;SITIT (correct&nbsp;&#10003;) layer by layer.</p>
  {fix}

  <h2 class="sec">5&ensp;Repository Guide</h2>
  <p>The project's code ships the analysis scripts and the per-run result JSONs for
  the primary orderings; model weights and raw image datasets are external. Each
  benchmark runner takes an <code>--order</code> flag
  and writes <code>&lt;dataset&gt;/results/&lt;model&gt;__&lt;order&gt;__results.json</code>,
  the exact files that populate the tables above. The order-aware prompt is built
  once in <code>model_manager.py</code> and shared across models, so a single code
  path produces STI, SIT, STIT, SITIT, and SITIT_rev. The per-layer animations on
  this page are rendered by the logit-lens overlay code from the same runs.</p>
  <div class="table-wrapper">
    <table class="booktabs"><thead>
      <tr><th class="rowname">Primary result / artifact</th><th>Script(s)</th></tr>
    </thead><tbody>{coderows}</tbody></table>
  </div>
  <p class="caption"><strong>Table 4:</strong> Where each primary result comes from.
  Run e.g. <code>python naturalbench_eval.py --order STI</code>. See
  <code>README.md</code> for the complete code&rarr;paper map and setup.</p>

  <hr>
  <p class="footer">Each animation is a per-layer logit lens on Qwen3-VL-8B: the
  image is shown at low resolution (few visual tokens) and each frame decodes one
  transformer layer &mdash; the heatmap is what each image patch decodes to and the
  token grid is what every token, including the generated answer, decodes to.
  Examples are NaturalBench yes/no pairs chosen for a clear, unambiguous answer.
  The code and full result files are available in the repository.</p>

</main>
<div id="lightbox" class="lightbox" hidden><img alt="raw image, enlarged"></div>
<script>
  // Play -> load + reveal the two GIFs
  document.querySelectorAll('.revealbtn').forEach(function (btn) {{
    btn.addEventListener('click', function () {{
      var r = btn.closest('.reveal');
      r.querySelectorAll('img.lazygif').forEach(function (img) {{
        if (!img.getAttribute('src')) img.src = img.dataset.gif;
      }});
      r.classList.add('playing');
    }});
  }});
  // Minimize -> collapse back to the static still + Play button, and stop the GIFs
  document.querySelectorAll('.minbtn').forEach(function (btn) {{
    btn.addEventListener('click', function () {{
      var r = btn.closest('.reveal');
      r.classList.remove('playing');
      r.querySelectorAll('img.lazygif').forEach(function (img) {{
        img.removeAttribute('src');
      }});
      r.scrollIntoView({{block: 'nearest'}});
    }});
  }});
  // Raw-image thumbnail -> lightbox
  var lb = document.getElementById('lightbox'), lbimg = lb.querySelector('img');
  document.querySelectorAll('.rawbtn').forEach(function (b) {{
    b.addEventListener('click', function () {{
      lbimg.src = b.dataset.full; lb.hidden = false;
    }});
  }});
  function closeLB() {{ lb.hidden = true; lbimg.removeAttribute('src'); }}
  lb.addEventListener('click', closeLB);
  document.addEventListener('keydown', function (e) {{ if (e.key === 'Escape') closeLB(); }});
</script>
</body></html>"""

    open(OUT, "w").write(page)
    print(f"[html] wrote {OUT}: {len(exs)} examples, {fnum - 3} figures")


if __name__ == "__main__":
    main()
