# code-release

Builds the anonymous supplementary package for the ICLR submission. Two things
reach reviewers: `supplementary/`, which becomes `code/` in the zip, and
`prior_workshop_paper.pdf`, which sits at the zip root beside it.

| Path | Role |
|------|------|
| `build_release.py` | stages `supplementary/` from the repository above, read-only, and copies the prior paper |
| `README_release.md` | the anonymous README, copied into the release as `code/README.md` |
| `supplementary/` | the staged release tree, regenerated on every build |
| `prior_workshop_paper.pdf` | the anonymized prior workshop paper, copied verbatim, regenerated on every build |
| `iclr_supplementary_code.zip` | the submission artifact: `code/` plus the prior paper at the root |

## Rebuild

```bash
python code-release/build_release.py

rm -rf /tmp/rel && mkdir -p /tmp/rel && cp -r code-release/supplementary /tmp/rel/code
cp code-release/prior_workshop_paper.pdf /tmp/rel/
(cd /tmp/rel && zip -qr9 iclr_supplementary_code.zip code prior_workshop_paper.pdf -x '*.pyc')
mv /tmp/rel/iclr_supplementary_code.zip code-release/
```

The build wipes and recreates `supplementary/`, so never edit anything inside it.
Edit `README_release.md` or the source files above instead. This directory is
excluded from its own output, so the release never contains a copy of itself.

## Scope

Shipped: Python sources, `requirements.txt`, `detpo_map/PROMPTS.md`, the
anonymous README, a `.gitignore`, and the anonymized prior workshop paper. The
paper is copied byte for byte and never passes through the scrubbing rules,
which would corrupt a PDF; it is already anonymous, and the check below confirms
that on every build.

Not shipped: every result file (the repository's tracked results run to about
2 GB, and the nine VQAv2-val dumps are 97 to 151 MB each), `website/` (author
names and institution links), `paper/` except the prior workshop paper (the LaTeX and the
submission PDF carry the prior workshop title, which is public and searchable),
and `fig3_pool.txt` (a dump of absolute local image paths).

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

`grep -I` skips the prior paper because it is binary, so scan its text
separately. This also catches a PDF that carries an author name only in its
metadata, which `pdftotext` does not print:

```bash
pdfinfo code-release/prior_workshop_paper.pdf | grep -E '^(Author|Title|Subject|Keywords)'
pdftotext code-release/prior_workshop_paper.pdf - | grep -inE \
  'rakshanda|gautam|ggare|carnegie|cmu|acknowledg|funding|eccv|workshop|2607\.15565'
```

The `pdfinfo` fields must be empty and the `pdftotext` scan must print nothing.
A hit on `eccv` or `workshop` means the venue leaked; a hit on `2607.15565`
means the paper cites its own public arXiv id, either of which de-anonymizes the
submission through its own supplementary material.
