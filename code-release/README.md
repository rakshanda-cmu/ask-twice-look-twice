# code-release

Builds the anonymous, code-only supplementary package for the ICLR submission.
Nothing here is shipped to reviewers except `supplementary/`, which is what the
zip contains.

| Path | Role |
|------|------|
| `build_release.py` | stages `supplementary/` from the repository above, read-only |
| `README_release.md` | the anonymous README, copied into the release as `README.md` |
| `supplementary/` | the staged release tree, 112 files, regenerated on every build |
| `iclr_supplementary_code.zip` | the submission artifact, `supplementary/` under a top-level `code/` |

## Rebuild

```bash
python code-release/build_release.py

rm -rf /tmp/rel && mkdir -p /tmp/rel && cp -r code-release/supplementary /tmp/rel/code
(cd /tmp/rel && zip -qr9 iclr_supplementary_code.zip code -x '*.pyc')
mv /tmp/rel/iclr_supplementary_code.zip code-release/
```

The build wipes and recreates `supplementary/`, so never edit anything inside it.
Edit `README_release.md` or the source files above instead. This directory is
excluded from its own output, so the release never contains a copy of itself.

## Scope

Shipped: Python sources, `requirements.txt`, `detpo_map/PROMPTS.md`, the
anonymous README, a `.gitignore`.

Not shipped: every result file (the repository's tracked results run to about
2 GB, and the nine VQAv2-val dumps are 97 to 151 MB each), `website/` (author
names and institution links), `paper/` (LaTeX and PDF, carrying the prior
workshop title, which is public and searchable), and `fig3_pool.txt` (a dump of
absolute local image paths).

`refcoco_gaze/` ships although the paper never uses it, because
`logit_lens_app.py` imports `refcoco_gaze.gaze_browser` at module level and the
browser would not start without it. The release README labels it as unused.

## What the build rewrites

Absolute local paths become repository-relative placeholders: home directories
and personal conda interpreters, scratch directories under `/tmp`, and `/data2`
mounts become `datasets/`, `hf_cache/`, `third_party/`, `./scratch` and plain
`python`. The repository name, which is the prior workshop title, becomes
`vlm-prompt-ordering`. Author and institution strings are replaced.

Those last rules run on source files only, never on result data. Benchmark text
legitimately contains strings the rules would otherwise corrupt: "CMUs" for
concrete masonry units, "ncmu3uIYeY1" as a video id, "Claude Cahun" and
"Jean-Claude Van Damme" as answers.

## Verify before submitting

```bash
cd code-release/supplementary
for p in grg claude anthropic rakshanda ggare Gautam Carnegie CMU ask-twice \
         /data2 anaconda scratchpad Co-Authored gmail 2607.15565; do
  n=$(grep -rIl -- "$p" . | wc -l); [ "$n" -gt 0 ] && echo "HIT $p"
done
python -m compileall -q . && find . -name __pycache__ -type d -exec rm -rf {} +
```

A clean run prints nothing. The scan must also pass on the unpacked zip, not
only on the staged tree.
