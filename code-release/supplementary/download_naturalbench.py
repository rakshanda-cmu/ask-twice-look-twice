"""
Download the NaturalBench benchmark (Zhiqiu Lin et al., NeurIPS 2024) from the
Hugging Face Hub and materialize it locally for evaluation + browsing.

NaturalBench structure (per group / row):
    - 2 images          : Image_0, Image_1
    - 2 questions        : Question_0, Question_1
    - 4 gold answers     : Image_i_Question_j  for i,j in {0,1}
      yielding 4 (image, question) pairs whose answers "alternate" so a blind
      LLM cannot win. Question_Type is "yes_no" (Yes/No) or "multiple_choice"
      (question embeds 'Option: A:…; B:…;', gold answer is the letter A/B).

Run:
    python download_naturalbench.py                 # all 1900 groups
    python download_naturalbench.py --num-groups 100   # quick subset

Produces:
    naturalbench/images/nb_<index>_0.jpg , nb_<index>_1.jpg
    naturalbench/groups.json   (metadata + image paths; no pixel data)
"""

import argparse
import json
import os

# Field layout of BaiqiL/NaturalBench
ANSWER_FIELDS = {
    (0, 0): "Image_0_Question_0",
    (1, 0): "Image_1_Question_0",
    (0, 1): "Image_0_Question_1",
    (1, 1): "Image_1_Question_1",
}


def main():
    ap = argparse.ArgumentParser(description="Download NaturalBench locally.")
    ap.add_argument("--out", default="./naturalbench", help="Output dir.")
    ap.add_argument("--num-groups", type=int, default=None, dest="num_groups",
                    help="Limit number of groups (default: all 1900).")
    ap.add_argument("--repo", default="BaiqiL/NaturalBench",
                    help="HF dataset repo id.")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = os.path.abspath(args.out)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"[load] {args.repo} (this downloads/caches the parquet shards) …")
    ds = load_dataset(args.repo, split="train")
    n = len(ds) if args.num_groups is None else min(args.num_groups, len(ds))
    print(f"[data] {len(ds)} groups available; materializing {n}.")

    groups = []
    for i in range(n):
        r = ds[i]
        idx = r["Index"]
        paths = {}
        for k in (0, 1):
            img = r[f"Image_{k}"].convert("RGB")
            fname = f"nb_{idx}_{k}.jpg"
            fpath = os.path.join(img_dir, fname)
            if not os.path.exists(fpath):
                img.save(fpath, format="JPEG", quality=95)
            paths[k] = os.path.join("images", fname)

        groups.append({
            "index": idx,
            "question_type": r["Question_Type"],
            "source": r["Source"],
            "image_0": paths[0],
            "image_1": paths[1],
            "question_0": r["Question_0"],
            "question_1": r["Question_1"],
            "image_0_question_0": r["Image_0_Question_0"],
            "image_1_question_0": r["Image_1_Question_0"],
            "image_0_question_1": r["Image_0_Question_1"],
            "image_1_question_1": r["Image_1_Question_1"],
        })

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] images written")

    groups_path = os.path.join(out_dir, "groups.json")
    with open(groups_path, "w") as f:
        json.dump({"num_groups": len(groups), "groups": groups}, f, indent=2)

    print(f"\n[done] {len(groups)} groups -> {groups_path}")
    print(f"[done] images under {img_dir}")


if __name__ == "__main__":
    main()
