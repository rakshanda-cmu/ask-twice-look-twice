"""
Download the VQA v2 *validation* split metadata from https://visualqa.org/.

VQA v2 reuses the MSCOCO val2014 images (already present under ./COCO/val2014/),
so we only need the Questions + Annotations JSONs — not the images.

Run:
    python download_vqa.py                 # downloads into ./vqa/
    python download_vqa.py --out ./vqa     # custom output dir

Produces (inside the output dir):
    v2_OpenEnded_mscoco_val2014_questions.json
    v2_mscoco_val2014_annotations.json
"""

import argparse
import os
import sys
import urllib.request
import zipfile

# Official VQA v2 download URLs (validation split). See https://visualqa.org/download.html
VQA_URLS = {
    "questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}

# Files expected inside the zips (used to skip re-downloading).
EXPECTED_FILES = {
    "questions": "v2_OpenEnded_mscoco_val2014_questions.json",
    "annotations": "v2_mscoco_val2014_annotations.json",
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / 1024 / 1024
        tot = total_size / 1024 / 1024
        sys.stdout.write(f"\r    {pct:5.1f}%  ({mb:7.1f} / {tot:7.1f} MB)")
    else:
        sys.stdout.write(f"\r    {downloaded / 1024 / 1024:7.1f} MB")
    sys.stdout.flush()


def download_and_extract(name, url, out_dir):
    target = os.path.join(out_dir, EXPECTED_FILES[name])
    if os.path.exists(target):
        print(f"[skip] {EXPECTED_FILES[name]} already present.")
        return target

    zip_path = os.path.join(out_dir, f"_{name}.zip")
    print(f"[download] {name}: {url}")
    urllib.request.urlretrieve(url, zip_path, _progress)
    print()  # newline after progress bar

    print(f"[extract] {os.path.basename(zip_path)} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    os.remove(zip_path)

    if not os.path.exists(target):
        raise FileNotFoundError(
            f"Expected {target} after extracting {name}, but it was not found."
        )
    print(f"[done] {target}")
    return target


def main():
    ap = argparse.ArgumentParser(description="Download VQA v2 val metadata.")
    ap.add_argument("--out", default="./vqa", help="Output directory (default: ./vqa)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    for name, url in VQA_URLS.items():
        download_and_extract(name, url, out_dir)

    print("\nAll VQA v2 val metadata ready in:", out_dir)
    print("Note: images come from MSCOCO val2014 (./COCO/val2014/), already present.")


if __name__ == "__main__":
    main()
