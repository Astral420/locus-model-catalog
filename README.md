# locus-model-catalog

A self-updating catalog of models sourced from the Hugging Face Hub, for the model manager of
[Locus](https://github.com/Astral420/Locus). It started as a fork of Jan's model catalog and keeps
its layout: Python scripts at the repo root, JSON files at the repo root, a daily GitHub Actions
cron that opens a PR, and releases that attach the JSON files.

| file | script | contents |
|---|---|---|
| `model_catalog.json` | `prepare_catalog.py` | chat/instruct GGUF models: discovered from the Hub, plus a pinned list |
| `whisper_catalog.json` | `prepare_whisper_catalog.py` | whisper.cpp `ggml-*.bin` models from `ggerganov/whisper.cpp` |
| `embedding_catalog.json` | `prepare_embedding_catalog.py` | pinned GGUF embedding models with their dimension and prefixes |

Locus fetches these files from this repo (raw GitHub URL or the latest release asset).

## Data structure

All three files are a list of repository entries in Jan's shape. One entry per Hugging Face repo,
one `quants` item per downloadable file:

```json
[
  {
    "model_name": "gemma-3-4b-it-GGUF",
    "developer": "unsloth",
    "downloads": 123456,
    "createdAt": "2025-03-12T10:00:00.000Z",
    "pipeline_tag": "image-text-to-text",
    "tools": false,
    "num_quants": 1,
    "quants": [
      {
        "model_id": "gemma-3-4b-it-Q4_K_M",
        "path": "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/<commit sha>/gemma-3-4b-it-Q4_K_M.gguf",
        "file_size": "2.3 GB",
        "size_bytes": 2489894016,
        "sha256": "<64 hex characters>"
      }
    ],
    "num_mmproj": 0,
    "mmproj_models": [],
    "readme": "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/<commit sha>/README.md",
    "description": "One-line summary.",
    "revision": "<commit sha>"
  }
]
```

What differs from Jan's files, and why:

- **`quants[].sha256` and `size_bytes`** (all three files): Locus verifies every download against a
  checksum; Jan's rounded `file_size` string (e.g. "2.3 GB") cannot be used for that. The Hub already returns both
  with the `?blobs=true` call the scripts make.
- **`path` is pinned to the commit** (`resolve/<sha>/...`, not `resolve/main/...`) and the entry has a
  `revision`. Otherwise a push to the Hub repo would invalidate the checksum until the next refresh.
- **`whisper_catalog.json` and `embedding_catalog.json` omit `tools`, `num_mmproj` and `mmproj_models`**
  (not applicable).
- **`embedding_catalog.json` entries add an `embedding` object**: `dimension`, `max_sequence_tokens`,
  `query_prefix`, `document_prefix`. The Hub does not expose these reliably, and Locus needs them to
  build and query its vector index.
- Whisper files are named `ggml-<size>[.en][-<quant>].bin`; the `model_id` is the file name without the extension.

## Pinned and excluded models

`prepare_catalog.py` keeps Jan's mechanisms: `PINNED_MODELS` are always included, `BLACKLISTED_DEVELOPERS`
are never included (Jan's own `janhq` and `Menlo` repos are blacklisted here), and `priority_devs` sort first.
Embedding models and their metadata are listed in `EMBEDDING_MODELS` in `prepare_embedding_catalog.py`.

## Running locally

```sh
pip install -r requirements.txt
python prepare_whisper_catalog.py
python prepare_embedding_catalog.py
python prepare_catalog.py        # long: walks the Hub listing; set HF_TOKEN to raise rate limits
python -m pytest
```

`API_KEY` / `BASE_URL` (an OpenAI-compatible endpoint) are only used to write one-line descriptions for
chat models; without them the descriptions are simply left as they are.
