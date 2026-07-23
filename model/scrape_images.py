"""Scrape mineral specimen images from DuckDuckGo image search into model/data/train/<class>/.

Downloads are deduped against a URL-hash filename, so re-running with new queries is safe
and won't re-download images already saved from a prior run.

Usage:
    python3 model/scrape_images.py --class Gold --query "gold nugget specimen" --query "raw gold ore" --count 50
    python3 model/scrape_images.py --class Pyrite --query "pyrite mineral specimen" --count 50 --out-dir model/data/train
    python3 model/scrape_images.py --all --count 100   # scrape Gold, Pyrite, and Other in one go
"""
import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

DEFAULT_QUERIES = {
    "Gold": ["gold nugget specimen", "native gold ore mineral", "gold nugget rock"],
    "Pyrite": ["pyrite mineral specimen", "fools gold pyrite crystal", "pyrite cube crystal rock"],
    "Other": ["quartz mineral specimen", "granite rock specimen", "mica schist mineral", "calcite mineral specimen"],
}


def get_vqd(query):
    qenc = urllib.parse.quote(query)
    url = f"https://duckduckgo.com/?q={qenc}&iax=images&ia=images"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
    m = re.search(r"vqd=['\"]([\d-]+)['\"]", html)
    return m.group(1) if m else None


def search_images(query, vqd, max_results=80, max_pages=4):
    qenc = urllib.parse.quote(query)
    urls = []
    next_p = f"https://duckduckgo.com/i.js?q={qenc}&vqd={vqd}&o=json&f=,,,,,&p=1"
    for _ in range(max_pages):
        if not next_p or len(urls) >= max_results:
            break
        req = urllib.request.Request(next_p, headers={"User-Agent": UA})
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore"))
        except Exception as e:
            print(f"    search error: {e}", file=sys.stderr)
            break
        for r in data.get("results", []):
            if r.get("image"):
                urls.append(r["image"])
        nxt = data.get("next")
        next_p = ("https://duckduckgo.com" + nxt) if nxt else None
        time.sleep(0.5)
    return urls


def download(url, dest_path, min_bytes=8000):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://duckduckgo.com/"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
    if len(data) < min_bytes:
        raise ValueError("too small")
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.width < 200 or img.height < 200:
        raise ValueError("too low-res")
    img.convert("RGB").save(dest_path, "WEBP", quality=90)


def scrape_class(cls, queries, count, out_root, dedup_dirs=()):
    out_dir = Path(out_root) / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in out_dir.glob("scrape_*.webp")}
    for d in dedup_dirs:
        existing |= {p.name for p in (Path(d) / cls).glob("scrape_*.webp")}
    print(f"== {cls}: {len(existing)} existing scraped images (staging + already-committed) ==")

    all_urls = []
    for q in queries:
        try:
            vqd = get_vqd(q)
            if not vqd:
                print(f"[{cls}]   no vqd for query '{q}'")
                continue
            urls = search_images(q, vqd, max_results=80)
            print(f"[{cls}]   query '{q}': {len(urls)} candidate urls")
            all_urls.extend(urls)
        except Exception as e:
            print(f"[{cls}]   query '{q}' failed: {e}")
        time.sleep(1)

    saved = 0
    seen_hashes = set()
    for url in all_urls:
        if saved >= count:
            break
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        if url_hash in seen_hashes:
            continue
        seen_hashes.add(url_hash)
        fname = f"scrape_{url_hash}.webp"
        if fname in existing:
            continue
        dest = out_dir / fname
        try:
            download(url, dest)
            saved += 1
            print(f"[{cls}]   [{saved}/{count}] saved {fname} from {url[:70]}")
        except Exception as e:
            print(f"[{cls}]   skip ({e}): {url[:70]}")
        time.sleep(0.3)

    print(f"== {cls}: saved {saved} new images ==")
    return saved


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--class", dest="cls", help="Class name, e.g. Gold, Pyrite, Other")
    group.add_argument("--all", action="store_true", help="Scrape Gold, Pyrite, and Other using built-in default queries")
    parser.add_argument("--query", dest="queries", action="append", help="Search query (repeatable, ignored with --all)")
    parser.add_argument("--count", type=int, default=50, help="Number of new images to save per class")
    parser.add_argument(
        "--out-dir",
        default="model/data/_scrape_review",
        help="Staging directory for manual review before moving into model/data/train",
    )
    parser.add_argument(
        "--dedup-dir",
        dest="dedup_dirs",
        action="append",
        default=["model/data/train", "model/data/test"],
        help="Directory (with per-class subfolders) to check for already-downloaded images, "
             "in addition to --out-dir. Repeatable. Defaults to model/data/train and model/data/test.",
    )
    args = parser.parse_args()

    if args.all:
        with ThreadPoolExecutor(max_workers=len(DEFAULT_QUERIES)) as pool:
            futures = {
                pool.submit(scrape_class, cls, queries, args.count, args.out_dir, args.dedup_dirs): cls
                for cls, queries in DEFAULT_QUERIES.items()
            }
            for future in futures:
                future.result()
        return

    if not args.queries:
        parser.error("--query is required unless --all is set")
    scrape_class(args.cls, args.queries, args.count, args.out_dir, args.dedup_dirs)


if __name__ == "__main__":
    main()
