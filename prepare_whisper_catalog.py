"""Build whisper_catalog.json: the whisper.cpp ggml models from Hugging Face.

Same entry shape as model_catalog.json (one entry per Hugging Face repo, one `quants` item per
file), so a client can read both with the same code. Differences are only the fields a
download needs: every quant also carries size_bytes and sha256, and its path is pinned to the
repo commit (see catalog_helpers.build_quant).

A file's `model_id` is its name without the extension, e.g. "ggml-small", "ggml-small.en-q5_1".
"""

import json
import re

import requests

from catalog_helpers import build_quant, fetch_repo_detail

# --- Configuration ---
OUTPUT_FILE = "whisper_catalog.json"

PINNED_MODELS = [
    "ggerganov/whisper.cpp",
]

# Plain and quantized ggml weights only. CoreML encoder zips, tinydiarize (-tdrz),
# for-tests-* and silero VAD files intentionally do not match.
WHISPER_FILE = re.compile(
    r"^ggml-(?:(?:tiny|base|small|medium)(?:\.en)?|large-v1|large-v2|large-v3-turbo|large-v3)"
    r"(?:-q\d_\d)?\.bin$"
)

DESCRIPTION = (
    "OpenAI Whisper speech-to-text models converted to ggml for whisper.cpp, "
    "in plain and quantized variants."
)


def is_whisper_weight(filename: str) -> bool:
    return bool(WHISPER_FILE.match(filename.lower()))


def process_model_details(repo_id, detail):
    """Return a catalog entry for the repo, or None if it has no usable whisper files."""
    if not detail:
        return None
    revision = detail.get("sha")
    if not revision:
        print("  -> No commit sha in model detail, skipping")
        return None

    quants = []
    readme_url = None
    for sib in detail.get("siblings", []):
        raw = sib.get("rfilename")
        if not raw:
            continue
        if raw.lower() == "readme.md":
            readme_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{raw}"
        elif is_whisper_weight(raw):
            quant = build_quant(repo_id, revision, sib)
            if quant is None:
                print(f"  -> Skipping {raw}: no sha256 reported by the Hub")
                continue
            quants.append(quant)

    if not quants:
        print("  -> No whisper weight files found, skipping")
        return None
    quants.sort(key=lambda q: q["model_id"])
    print(f"  -> Found {len(quants)} whisper models")

    return {
        "model_name": repo_id.split("/")[-1],
        "developer": repo_id.split("/")[0],
        "downloads": detail.get("downloads", 0),
        "createdAt": detail.get("createdAt"),
        "pipeline_tag": detail.get("pipeline_tag"),
        "num_quants": len(quants),
        "quants": quants,
        "readme": readme_url,
        "description": DESCRIPTION,
        "revision": revision,
    }


def get_whisper_catalog():
    catalog = []
    for repo_id in PINNED_MODELS:
        print(f"Processing {repo_id}")
        try:
            detail = fetch_repo_detail(repo_id)
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"  -> Failed to fetch {repo_id}: {e}")
            continue
        entry = process_model_details(repo_id, detail)
        if entry is not None:
            catalog.append(entry)

    if not catalog:
        # Never replace a good catalog with an empty one.
        raise SystemExit("No whisper models could be fetched; leaving the existing catalog untouched")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated or Added {len(catalog)} models; catalog total now {len(catalog)}")


if __name__ == "__main__":
    get_whisper_catalog()
