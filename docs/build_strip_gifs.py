"""
Strip the confidence colorbar (the light->dark-blue "0.0 .. 1.0" strip) off the
bottom of every logit-lens GIF and its poster. The strip is redundant on every
frame. Idempotent: skips a GIF whose bottom band is already dark (colorbar gone).

    python build_strip_gifs.py     # crops assets/ex*_{STI,SIT,SITIT}.gif in place
"""
import glob, os
import imageio, numpy as np
from PIL import Image, ImageSequence

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
CUT = 32          # px removed from the bottom of each 470-wide frame
FPS = 2.0


def has_colorbar(frame):
    """True if the bottom band still holds the bright white->blue gradient bar."""
    a = np.asarray(frame.convert("RGB"))
    H, W, _ = a.shape
    band = a[H - CUT:H, int(W * 0.25):int(W * 0.75)]
    return band.mean() > 60            # colorbar band is bright; generated area is dark


def main():
    gifs = sorted(glob.glob(os.path.join(ASSETS, "ex*_STI.gif")) +
                  glob.glob(os.path.join(ASSETS, "ex*_SIT.gif")) +
                  glob.glob(os.path.join(ASSETS, "ex*_SITIT.gif")))
    n = 0
    for path in gifs:
        try:
            im = Image.open(path)
            im.seek(0)
            if not has_colorbar(im.convert("RGB")):
                continue                # already stripped
            frames = []
            for f in ImageSequence.Iterator(im):
                fr = f.convert("RGB")
                frames.append(np.array(fr.crop((0, 0, fr.width, fr.height - CUT))))
        except Exception as ex:
            print("  skip %s (%s)" % (os.path.basename(path), ex))
            continue
        imageio.mimsave(path, frames, format="GIF",
                        duration=int(1000 / FPS), loop=0)
        poster = path.replace(".gif", "_poster.png")
        if os.path.exists(poster):
            Image.fromarray(frames[0]).save(poster)
        n += 1
        print("  stripped %s (%d frames)" % (os.path.basename(path), len(frames)))
    print("[done] stripped %d gifs" % n)


if __name__ == "__main__":
    main()
