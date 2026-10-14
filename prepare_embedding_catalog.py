"""Build embedding_catalog.json: GGUF embedding models served through llama.cpp.

Same entry shape as model_catalog.json (one entry per Hugging Face repo, one `quants` item per
file; quants also carry size_bytes and sha256, paths are commit-pinned). Unlike the chat
catalog, embedding models are not discovered from the Hub listing, because a client cannot
use one without facts the Hub does not expose reliably: the vector dimension, the input
limit, and any query/document prefix the model was trained with. Those live in
EMBEDDING_MODELS below and are written to each entry's `embedding` object.
"""

import json
import re

import requests

from catalog_helpers import (
    build_quant,
    fetch_repo_detail,
    is_mmproj_file,
    is_multipart_gguf,
)

# --- Configuration ---
OUTPUT_FILE = "embedding_catalog.json"

# `repos` are tried in order and the first that resolves with matching files wins.
# `required` makes the run fail instead of silently dropping the model.
EMBEDDING_MODELS = [
    {
        "repos": ["leliuga/all-MiniLM-L6-v2-GGUF", "second-state/All-MiniLM-L6-v2-Embedding-GGUF"],
        "include": r"^all-minilm-l6-v2",
        "dimension": 384,
        "max_sequence_tokens": 256,
        "query_prefix": "",
        "document_prefix": "",
        "description": "Very small English sentence embedder (384 dimensions); input beyond 256 tokens is truncated.",
        "required": True,
    },
    {
        "repos": ["CompendiumLabs/bge-small-en-v1.5-gguf"],
        "include": r"^bge-small-en-v1\.5",
        "dimension": 384,
        "max_sequence_tokens": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
        "description": "Small English retrieval embedder (384 dimensions) with a 512-token window.",
    },
    {
        "repos": ["nomic-ai/nomic-embed-text-v1.5-GGUF"],
        "include": r"^nomic-embed-text-v1\.5",
        "dimension": 768,
        "max_sequence_tokens": 2048,
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "description": "Larger English embedder (768 dimensions) for longer chunks; needs the search_query:/search_document: prefixes.",
    },
]


def process_model_details(repo_id, detail, meta):
    """Return a catalog entry for the repo, or None if it has no usable files."""
    if not detail:
        return None
    revision = detail.get("sha")
    if not revision:
        print("  -> No commit sha in model detail, skipping")
        return None

    include = re.compile(meta["include"], re.I)
    quants = []
    readme_url = None
    for sib in detail.get("siblings", []):
        raw = sib.get("rfilename")
        if not raw:
            continue
        name = raw.lower()
        if name == "readme.md":
            readme_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{raw}"
        elif name.endswith(".gguf") and include.search(raw):
            if is_multipart_gguf(raw):
                print(f"  -> Skipping multi-part file: {raw}")
                continue
            if is_mmproj_file(name):
                continue
            quant = build_quant(repo_id, revision, sib)
            if quant is None:
                print(f"  -> Skipping {raw}: no sha256 reported by the Hub")
                continue
            quants.append(quant)

    if not quants:
        print("  -> No matching GGUF files found")
        return None
    quants.sort(key=lambda q: q["size_bytes"])
    print(f"  -> Found {len(quants)} quants")

    return {
        "model_name": repo_id.split("/")[-1],
        "developer": repo_id.split("/")[0],
        "downloads": detail.get("downloads", 0),
        "createdAt": detail.get("createdAt"),
        "pipeline_tag": detail.get("pipeline_tag"),
        "num_quants": len(quants),
        "quants": quants,
        "readme": readme_url,
        "description": meta["description"],
        "revision": revision,
        "embedding": {
            "dimension": meta["dimension"],
            "max_sequence_tokens": meta["max_sequence_tokens"],
            "query_prefix": meta["query_prefix"],
            "document_prefix": meta["document_prefix"],
        },
    }


def get_embedding_catalog(models=None):
    catalog = []
    missing_required = []
    for meta in models if models is not None else EMBEDDING_MODELS:
        entry = None
        for repo_id in meta["repos"]:
            print(f"Processing {repo_id}")
            try:
                detail = fetch_repo_detail(repo_id)
            except (requests.exceptions.RequestException, ValueError) as e:
                print(f"  -> Failed to fetch {repo_id}: {e}")
                continue
            entry = process_model_details(repo_id, detail, meta)
            if entry is not None:
                break
        if entry is not None:
            catalog.append(entry)
        elif meta.get("required"):
            missing_required.append(meta["repos"][0])
        else:
            print(f"  -> {meta['repos'][0]} omitted from the catalog")

    if missing_required:
        raise SystemExit(f"Required embedding model(s) not found: {missing_required}; leaving the existing catalog untouched")
    if not catalog:
        raise SystemExit("No embedding models could be fetched; leaving the existing catalog untouched")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated or Added {len(catalog)} models; catalog total now {len(catalog)}")


if __name__ == "__main__":
    get_embedding_catalog()
