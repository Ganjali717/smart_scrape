"""
Download SWDE verticals (raw HTML + ground truth) from Hugging Face
(dataset: abdo-Mansour/SWDE) into data/swde/.

    python scripts/download_swde.py camera book auto
    python scripts/download_swde.py all
"""

import os
import sys
import urllib.request

BASE = "https://huggingface.co/datasets/abdo-Mansour/SWDE/resolve/main"
VERTICALS = ["auto", "book", "camera", "job", "movie", "nbaplayer", "restaurant", "university"]


def fetch(url, dst):
    if os.path.exists(dst):
        print(f"  exists: {dst}")
        return
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dst)


def main(args):
    verticals = VERTICALS if (not args or args == ["all"]) else args
    os.makedirs("data/swde/html", exist_ok=True)
    import py7zr
    for v in verticals:
        print(f"[{v}]")
        fetch(f"{BASE}/data/{v}-00000-of-00001.parquet", f"data/swde/{v}-00000-of-00001.parquet")
        arc = f"data/swde/{v}.7z"
        fetch(f"{BASE}/files/{v}.7z", arc)
        if not os.path.isdir(f"data/swde/html/{v}"):
            print("  extracting ...")
            with py7zr.SevenZipFile(arc) as z:
                z.extractall("data/swde/html")


if __name__ == "__main__":
    main(sys.argv[1:])
