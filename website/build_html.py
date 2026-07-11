"""
Build the static review-supplement site (index.html) from the generated assets.

Structure:
  1. Intro + ordering legend
  2. Reading the model: per-patch logit-lens *word* stills (Figure-3 style)
  3. THE PROBLEM: question position flips the answer -- SIT (right) vs STI (wrong),
     with the SIT-vs-STI paradox numbers.
  4. THE FIX: image echoing restores the answer -- STI (wrong) vs SITIT (right),
     with the ordering-ladder numbers.

    python build_html.py      # writes ../index.html, reads assets/manifest.json
"""
import json, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "index.html")
SELECT = [229, 638, 978, 1231]
WORDS = [(229, "Is the dog chasing a person?"), (638, "Is the person walking?"),
         (978, "Is the man performing on a ramp?"), (1231, "Is the girl entering the water?")]

PARADOX = {  # SIT (question-last) vs STI (question-first); Qwen3-VL-8B group acc.
    "NaturalBench": {"STI": 0.270, "SIT": 0.351},
    "Winoground":   {"STI": 0.223, "SIT": 0.318},
}
LADDER = {   # the fix: STI -> STIT -> SITIT
    "NaturalBench": {"STI": 0.270, "STIT": 0.350, "SITIT": 0.374},
    "Winoground":   {"STI": 0.223, "STIT": 0.375, "SITIT": 0.403},
}
LEGEND = [
    ("STI",  "System · Task · Image", "question-first (the paradox)"),
    ("SIT",  "System · Image · Task", "question-last (baseline)"),
    ("STIT", "System · Task · Image · Task", "question echo (ours)"),
    ("SITIT","System · Image · Task · Image · Task", "image echo (ours, best)"),
]


def esc(s): return html.escape(str(s))


def gtbadge(gt):
    cls = "yes" if str(gt).lower().startswith("y") else "no"
    return f'<span class="gt {cls}">ground truth: {esc(gt)}</span>'


def col(e, key, label, role):
    d = e[key]; ok = d["correct"]
    tag = "good" if ok else "bad"
    mark = '<span class="c">&#10003; correct</span>' if ok else '<span class="x">&#10007; wrong</span>'
    return f"""<figure class="col">
        <figcaption><span class="tag {tag}">{esc(label)}</span>
          <span class="role2">{esc(role)}</span><br>answer: <b>{esc(d['pred'])}</b> {mark}</figcaption>
        <img src="assets/{esc(d['gif'])}" alt="{esc(label)} per-layer logit lens" loading="lazy">
      </figure>"""


def card(e, lkey, llab, lrole, rkey, rlab, rrole):
    return f"""
      <section class="card">
        <div class="ex-head">
          <img class="thumb" src="assets/{esc(e['image'])}" alt="example image (low resolution)">
          <div class="qbox">
            <div class="q">{esc(e['question'])}</div>
            {gtbadge(e['gt'])}
            <div class="tok">low resolution &middot; {e['img_size'][0]}&times;{e['img_size'][1]} px (few image tokens, zoomed)</div>
          </div>
        </div>
        <div class="pair">{col(e, lkey, llab, lrole)}{col(e, rkey, rlab, rrole)}</div>
      </section>"""


def numtable(data, cols, hi_worst, hi_best):
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for ds, d in data.items():
        cells = ""
        for c in cols:
            cls = "worst" if c == hi_worst else ("best" if c == hi_best else "")
            cells += f'<td class="{cls}">{d[c]:.3f}</td>'
        rows += f"<tr><th>{esc(ds)}</th>{cells}</tr>"
    return (f'<table><thead><tr><th>Benchmark (group acc.)</th>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def main():
    man = json.load(open(os.path.join(ASSETS, "manifest.json")))
    by = {e["idx"]: e for e in man["examples"]}
    exs = [by[i] for i in SELECT if i in by]

    legend = "".join(
        f'<li><b>{esc(k)}</b> <span class="seq">{esc(seq)}</span> '
        f'<span class="role">{esc(role)}</span></li>' for k, seq, role in LEGEND)

    words = ""
    for idx, q in WORDS:
        e = by.get(idx)
        if e and e.get("fig_panel"):
            words += (f'<figure class="wfig"><img src="assets/{esc(e["fig_panel"])}" '
                      f'alt="per-patch logit-lens panel: {esc(q)}" loading="lazy"></figure>')

    problem_cards = "".join(
        card(e, "SIT", "SIT — question-last", "moves after the image",
                "STI", "STI — question-first", "moves before the image") for e in exs)
    fix_cards = "".join(
        card(e, "STI", "STI — question-first", "still broken",
                "SITIT", "SITIT — image echo", "the fix") for e in exs)

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ask Twice, Look Twice &mdash; Interactive Supplement</title>
<style>
  :root {{ --ink:#1a1c1f; --mut:#5b6470; --line:#e2e6ea; --bg:#f7f9fb; --card:#fff;
          --bad:#c23b32; --good:#1f8a5b; --best:#e6f6ee; --worst:#fdeceb; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:30px; line-height:1.2; margin:0 0 6px; letter-spacing:-.01em; }}
  .sub {{ color:var(--mut); font-size:16px; margin:0 0 26px; }}
  .lead {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:20px 22px; margin:0 0 26px; }}
  .lead p {{ margin:0 0 12px; }} .lead p:last-child {{ margin:0; }}
  ul.legend {{ list-style:none; padding:0; margin:14px 0 0; display:grid;
              grid-template-columns:1fr 1fr; gap:6px 24px; }}
  ul.legend li {{ font-size:14.5px; }}
  ul.legend .seq {{ color:var(--ink); }} ul.legend .role {{ color:var(--mut); }}
  h2 {{ font-size:22px; margin:40px 0 6px; }}
  .h2sub {{ color:var(--mut); margin:0 0 14px; }}
  .sec-problem h2 {{ color:var(--bad); }} .sec-fix h2 {{ color:var(--good); }}
  table {{ border-collapse:collapse; width:100%; background:var(--card);
          border:1px solid var(--line); border-radius:12px; overflow:hidden; margin:6px 0; }}
  th,td {{ padding:10px 14px; text-align:center; border-bottom:1px solid var(--line); }}
  thead th {{ background:#eef2f6; font-weight:600; }}
  tbody th {{ text-align:left; font-weight:600; }}
  td.worst {{ background:var(--worst); color:var(--bad); font-weight:700; }}
  td.best {{ background:var(--best); color:var(--good); font-weight:700; }}
  .note {{ color:var(--mut); font-size:13.5px; margin:8px 2px 20px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:20px; margin:18px 0; }}
  .ex-head {{ display:flex; gap:18px; align-items:center; margin-bottom:16px; }}
  .thumb {{ width:190px; height:auto; border-radius:10px; border:1px solid var(--line); }}
  .qbox .q {{ font-size:19px; font-weight:600; }}
  .gt {{ display:inline-block; margin-top:8px; padding:3px 10px; border-radius:20px;
        font-size:14px; font-weight:600; }}
  .gt.yes {{ background:var(--best); color:var(--good); }}
  .gt.no {{ background:var(--worst); color:var(--bad); }}
  .tok {{ color:var(--mut); font-size:13px; margin-top:8px; }}
  .pair {{ display:flex; gap:18px; align-items:flex-start; }}
  .col {{ flex:1; margin:0; min-width:0; }}
  .col img {{ width:100%; height:auto; border:1px solid var(--line); border-radius:8px; background:#111; }}
  figcaption {{ font-size:14px; margin-bottom:8px; }}
  .role2 {{ color:var(--mut); font-size:13px; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:6px; font-weight:600;
         font-size:13px; margin-right:6px; }}
  .tag.bad {{ background:var(--worst); color:var(--bad); }}
  .tag.good {{ background:var(--best); color:var(--good); }}
  .x {{ color:var(--bad); font-weight:700; }} .c {{ color:var(--good); font-weight:700; }}
  /* one still per row, centered -> mismatched aspect ratios never leave a gap */
  .words {{ display:block; }}
  .wfig {{ max-width:760px; margin:0 auto 22px; }}
  .wfig img {{ width:100%; height:auto; border:1px solid var(--line); border-radius:10px; display:block; }}
  .wfig figcaption {{ color:var(--mut); font-size:13.5px; margin-top:6px; text-align:center; }}
  footer {{ color:var(--mut); font-size:13.5px; margin-top:44px; border-top:1px solid var(--line); padding-top:18px; }}
  @media (max-width:720px) {{ .pair{{flex-direction:column;}} .ex-head{{flex-direction:column;align-items:flex-start;}}
    ul.legend{{grid-template-columns:1fr;}} .thumb{{width:100%;max-width:280px;}} }}
</style></head>
<body><div class="wrap">
  <h1>Ask Twice, Look Twice</h1>
  <p class="sub">Interactive supplement &mdash; the question-first paradox and its fix, layer by layer.</p>

  <div class="lead">
    <p>A vision-language model reads <b>system</b>, <b>image</b>, and <b>question</b>
    tokens as one stream and answers from the final position. <b>Where the question goes
    changes the answer.</b> This page shows, on clear low-token NaturalBench yes/no cases,
    <b>first the problem</b> (moving the question <i>before</i> the image flips a correct
    answer to a wrong one) and <b>then the fix</b> (re-presenting the image and question
    restores it), each with a <b>per-layer logit lens</b> so you can watch the answer form
    with depth.</p>
    <ul class="legend">{legend}</ul>
  </div>

  <h2>Reading the model: per-patch logit lens</h2>
  <p class="h2sub">Each image patch, projected through the output embedding, decodes to a
    vocabulary token. Overlaying that token on the patch shows <b>what the model sees</b>.
    The animations further down scrub this through every layer.</p>
  <p class="note">One panel per example (Qwen3-VL-8B, question-first, a late layer). Each
    reads on its own: the question and ground truth on top, the per-patch decodings in the
    middle, and what it proves at the bottom. Images are shown at low resolution (few,
    large, zoomed patches).</p>
  <div class="words">{words}</div>

  <div class="sec-problem">
  <h2>1. The problem: question position flips the answer</h2>
  <p class="h2sub">Same content, only the question moves. Placing it <b>after</b> the image
    (<b>SIT</b>) answers correctly; placing it <b>first</b> (<b>STI</b>) commits early to a
    wrong, image-anchored answer.</p>
  {numtable(PARADOX, ["SIT","STI"], "STI", "SIT")}
  <p class="note">Qwen3-VL-8B group accuracy. Question-first (<b>STI</b>) trails
    question-last (<b>SIT</b>) by 8&ndash;10 points with identical content.</p>
  {problem_cards}
  </div>

  <div class="sec-fix">
  <h2>2. The fix: echoing restores the answer</h2>
  <p class="h2sub">Re-presenting the (image, question) after the image (<b>SITIT</b>) gives
    the decoder a second, adjacent look and fixes the same cases &mdash; no training.</p>
  {numtable(LADDER, ["STI","STIT","SITIT"], "STI", "SITIT")}
  <p class="note">Qwen3-VL-8B group accuracy. From question-first (<b>STI</b>, worst) up the
    ladder to image echoing (<b>SITIT</b>, best).</p>
  {fix_cards}
  </div>

  <footer>
    Each animation is a per-layer logit lens on Qwen3-VL-8B: the image is presented at low
    resolution (few visual tokens) and each frame decodes one transformer layer &mdash; the
    heatmap is what each image patch decodes to and the token grid is what every token,
    including the generated answer, decodes to. Examples are NaturalBench yes/no pairs chosen
    for a clear, unambiguous answer. Static supplement; the code, interactive viewer, and
    full result files accompany it in the repository.
  </footer>
</div></body></html>"""

    open(OUT, "w").write(page)
    print(f"[html] wrote {OUT}: {len(exs)} examples, {len(WORDS)} word stills")


if __name__ == "__main__":
    main()
