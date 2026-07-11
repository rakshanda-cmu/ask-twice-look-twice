"""
Build the static review-supplement site (index.html) as an academic project page
(serif body, centered headings, styled tables/figures), from the generated assets.

    python build_html.py      # writes ../index.html, reads assets/manifest.json
"""
import json, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "index.html")
SELECT = [229, 638, 978, 1231]
PANELS = [229, 638, 978, 1231]

TITLE = ("Ask Twice, Look Twice: Prompt Echoing Resolves the "
         "Question-First Paradox in Vision-Language Models")
PARADOX = {"NaturalBench": {"STI": 0.270, "SIT": 0.351},
           "Winoground":   {"STI": 0.223, "SIT": 0.318}}
LADDER = {"NaturalBench": {"STI": 0.270, "SIT": 0.351, "STIT": 0.350, "SITIT": 0.374},
          "Winoground":   {"STI": 0.223, "SIT": 0.318, "STIT": 0.375, "SITIT": 0.403}}
LEGEND = [("STI",  "System &middot; Task &middot; Image", "question-first (the paradox)"),
          ("SIT",  "System &middot; Image &middot; Task", "question-last (baseline)"),
          ("STIT", "System &middot; Task &middot; Image &middot; Task", "question echo (ours)"),
          ("SITIT","System &middot; Image &middot; Task &middot; Image &middot; Task", "image echo (ours, best)")]


def esc(s): return html.escape(str(s))


def mark(ok): return ('<span style="color:#1a6b1a;font-weight:700;">&#10003; correct</span>'
                      if ok else '<span style="color:#c0392b;font-weight:700;">&#10007; wrong</span>')


def numtable(data, cols, best, worst):
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for ds, d in data.items():
        cells = ""
        for c in cols:
            cls = "best" if c == best else ""
            style = ' style="color:#c0392b;font-weight:700;"' if c == worst else ""
            cells += f'<td class="{cls}"{style}>{d[c]:.3f}</td>'
        rows += f"<tr><td>{esc(ds)}</td>{cells}</tr>"
    return (f'<div class="table-wrapper"><table><thead><tr>'
            f'<th>Benchmark (group accuracy)</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


def gif_cards(exs, lkey, llab, lrole, rkey, rlab, rrole):
    out = ""
    for e in exs:
        L, R = e[lkey], e[rkey]
        gtc = "#1a6b1a" if e["gt"].lower().startswith("y") else "#c0392b"
        out += f"""
        <h3 style="text-align:left;">&ldquo;{esc(e['question'])}&rdquo;
          <span style="font-weight:400;color:{gtc};font-size:1rem;">&nbsp;&mdash; ground truth: {esc(e['gt'])}</span></h3>
        <div class="two-col-grid">
          <div class="figure-container">
            <img src="assets/{esc(L['gif'])}" alt="{esc(llab)}" loading="lazy">
            <p class="figure-caption"><strong>{esc(llab)}</strong> &mdash; {esc(lrole)}.
              Answer &ldquo;{esc(L['pred'])}&rdquo; {mark(L['correct'])}.</p>
          </div>
          <div class="figure-container">
            <img src="assets/{esc(R['gif'])}" alt="{esc(rlab)}" loading="lazy">
            <p class="figure-caption"><strong>{esc(rlab)}</strong> &mdash; {esc(rrole)}.
              Answer &ldquo;{esc(R['pred'])}&rdquo; {mark(R['correct'])}.</p>
          </div>
        </div>"""
    return out


def main():
    man = json.load(open(os.path.join(ASSETS, "manifest.json")))
    by = {e["idx"]: e for e in man["examples"]}
    exs = [by[i] for i in SELECT if i in by]

    legend_rows = "".join(
        f'<tr><td><strong>{k}</strong></td><td>{seq}</td><td>{role}</td></tr>'
        for k, seq, role in LEGEND)

    panels = ""
    for i in PANELS:
        e = by.get(i)
        if e and e.get("fig_panel"):
            panels += (f'<div class="figure-container">'
                       f'<img src="assets/{esc(e["fig_panel"])}" '
                       f'alt="per-patch logit-lens panel" loading="lazy"></div>')

    problem = gif_cards(exs, "SIT", "SIT (question-last)", "the working baseline",
                        "STI", "STI (question-first)", "the paradox")
    fix = gif_cards(exs, "STI", "STI (question-first)", "still broken",
                    "SITIT", "SITIT (image echo)", "the fix")

    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(TITLE)}</title>
<style>
  :root {{ --primary-color:#363636; --link-color:#0000EE; --text-color:#4a4a4a; }}
  body {{ font-family:"Times New Roman",Times,serif; background:#fdfdfd; color:var(--text-color);
         line-height:1.6; margin:0; padding:0; font-size:1.1rem; }}
  h1,h2,h3 {{ color:var(--primary-color); font-weight:800; text-align:center; }}
  h1 {{ font-size:2.3rem; margin-bottom:.5rem; line-height:1.25; }}
  h2 {{ font-size:1.75rem; margin-top:2.5rem; margin-bottom:1.2rem; }}
  h3 {{ font-size:1.3rem; font-weight:600; margin-top:1.6rem; }}
  a {{ color:var(--link-color); text-decoration:none; }}
  .container {{ max-width:1000px; margin:0 auto; padding:2rem 1.5rem; }}
  .header-section {{ text-align:center; padding-bottom:1rem; }}
  .venue {{ font-size:1.1rem; color:#555; margin-bottom:.5rem; font-style:italic; }}
  .abstract-container {{ max-width:820px; margin:0 auto; text-align:justify; }}
  p {{ margin-bottom:1.2em; }}
  .figure-container {{ margin:1.6rem auto; text-align:center; }}
  .figure-container img {{ max-width:100%; border-radius:8px; box-shadow:0 5px 15px rgba(0,0,0,.08); }}
  .figure-caption {{ margin-top:.7rem; font-size:.95rem; color:#555;
                    font-family:Helvetica,Arial,sans-serif; text-align:justify;
                    max-width:820px; margin-left:auto; margin-right:auto; }}
  .figure-caption strong {{ color:#333; }}
  .two-col-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin:1.2rem auto; align-items:start; }}
  .two-col-grid .figure-container {{ margin:0; }}
  .table-wrapper {{ overflow-x:auto; margin:1.6rem 0; }}
  table {{ width:100%; border-collapse:collapse; box-shadow:0 4px 10px rgba(0,0,0,.05);
          background:#fff; border-radius:8px; overflow:hidden;
          font-family:Helvetica,Arial,sans-serif; font-size:.9rem; }}
  thead {{ background:#f4f4f4; border-bottom:2px solid #ddd; }}
  th,td {{ padding:9px 12px; text-align:center; }}
  th:first-child,td:first-child {{ text-align:left; }}
  th {{ font-weight:700; color:#333; }}
  td {{ color:#555; border-bottom:1px solid #eee; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover {{ background:#fafafa; }}
  .best {{ font-weight:700; color:#1a6b1a; }}
  .highlight-box {{ background:#f8f8f8; border-left:4px solid #888; border-radius:4px;
                   padding:.8rem 1.2rem; margin:1rem auto; max-width:820px;
                   font-family:Helvetica,Arial,sans-serif; font-size:.98rem; color:#444; }}
  .note {{ font-family:Helvetica,Arial,sans-serif; font-size:.9rem; color:#666;
          max-width:820px; margin:.4rem auto 0; text-align:center; }}
  hr {{ border:0; height:1px; background:#e0e0e0; margin:3rem auto; width:50%; }}
  @media (max-width:768px) {{ h1{{font-size:1.7rem;}} .two-col-grid{{grid-template-columns:1fr;}}
    table{{display:block;overflow-x:auto;}} }}
</style></head>
<body><div class="container">

  <div class="header-section">
    <h1>{esc(TITLE)}</h1>
    <div class="venue">Interactive Supplement &nbsp;&middot;&nbsp; Anonymous ECCV 2026 Submission</div>
  </div>

  <div class="abstract-container">
    <h2>Overview</h2>
    <p>A vision-language model reads <strong>system</strong>, <strong>image</strong>, and
    <strong>question</strong> tokens as one stream and answers from the final position.
    <strong>Where the question goes changes the answer.</strong> Placing it <em>before</em>
    the image (question-first, <strong>STI</strong>) makes the model commit early to an
    image-anchored, often wrong answer &mdash; even though it perceives the scene correctly.
    Re-presenting the image and question after the image (<strong>SITIT</strong>) restores the
    decoder's access to the question and fixes it, with no training.</p>
    <p>This page shows the effect on clear, low-token NaturalBench yes/no cases, with a
    <strong>per-layer logit lens</strong> so you can watch the answer form with depth: first
    the problem (moving the question first flips a correct answer to a wrong one), then the fix
    (echoing restores it). The four prompt orderings:</p>
  </div>
  <div class="table-wrapper" style="max-width:820px;margin:1rem auto;">
    <table><thead><tr><th>Ordering</th><th>Token sequence</th><th>Role</th></tr></thead>
    <tbody>{legend_rows}</tbody></table>
  </div>

  <hr>

  <h2>Reading the Model: Per-Patch Logit Lens</h2>
  <div class="abstract-container">
    <p>Each image patch, projected through the output embedding, decodes to a vocabulary
    token; overlaying that token on the patch shows <strong>what the model sees</strong>. Each
    panel below reads on its own &mdash; the question and ground truth on top, the per-patch
    decodings in the middle, and what it proves at the bottom. The patches decode to the correct
    scene even when question-first then answers wrong: <strong>the paradox is a read-out
    failure, not a perception failure.</strong></p>
  </div>
  {panels}
  <p class="note">Qwen3-VL-8B, question-first (STI), a late layer. Images shown at low
    resolution (few, large, zoomed patches).</p>

  <hr>

  <h2>The Problem: Question Position Flips the Answer</h2>
  <div class="abstract-container">
    <p>Same content, only the question moves. Placing it <strong>after</strong> the image
    (<strong>SIT</strong>) answers correctly; placing it <strong>first</strong>
    (<strong>STI</strong>) commits early to a wrong, image-anchored answer. Question-first
    trails question-last by 8&ndash;10 group-accuracy points with identical content.</p>
  </div>
  {numtable(PARADOX, ["SIT", "STI"], "SIT", "STI")}
  <p class="note">Qwen3-VL-8B group accuracy. The animations decode one transformer layer per
    frame; watch the correct answer emerge under SIT and the wrong answer lock in under STI.</p>
  {problem}

  <hr>

  <h2>The Fix: Echoing Restores the Answer</h2>
  <div class="abstract-container">
    <p>Re-presenting the (image, question) after the image (<strong>SITIT</strong>) gives the
    decoder a second, adjacent look and fixes the same cases &mdash; no training. Accuracy climbs
    the ordering ladder from question-first (STI, worst) to image echoing (SITIT, best).</p>
  </div>
  {numtable(LADDER, ["STI", "SIT", "STIT", "SITIT"], "SITIT", "STI")}
  <p class="note">Qwen3-VL-8B group accuracy. Best per benchmark in <span class="best">green</span>.</p>
  {fix}

  <hr>

  <div class="abstract-container">
    <p style="font-family:Helvetica,Arial,sans-serif;font-size:.9rem;color:#666;text-align:center;">
    Each animation is a per-layer logit lens on Qwen3-VL-8B: the image is shown at low resolution
    (few visual tokens) and each frame decodes one transformer layer &mdash; the heatmap is what
    each image patch decodes to and the token grid is what every token, including the generated
    answer, decodes to. Examples are NaturalBench yes/no pairs chosen for a clear, unambiguous
    answer. Static supplement; the code, interactive viewer, and full result files accompany it in
    the repository.</p>
  </div>

</div></body></html>"""

    open(OUT, "w").write(page)
    print(f"[html] wrote {OUT}: {len(exs)} examples, {len(PANELS)} panels")


if __name__ == "__main__":
    main()
