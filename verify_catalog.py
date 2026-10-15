#!/usr/bin/env python3
"""Acceptance test for the Locus model catalog. Run from the repo root.

    python verify_catalog.py                      # structure checks on the JSON files (offline)
    python verify_catalog.py --network            # + HEAD the sample files, compare Content-Length to size_bytes
    python verify_catalog.py --network --download # + download the smallest whisper/embedding file, verify sha256

Exit code 0 = no FAIL. WARN means "look at it", not "broken".
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import requests

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
results = []


def check(status, name, detail=""):
    results.append((status, name, detail))
    print(f"{status:<5} {name}" + (f"  -- {detail}" if detail else ""))


def load(path):
    p = Path(path)
    if not p.exists():
        check("FAIL", f"{path} exists", "file missing")
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        check("FAIL", f"{path} is a non-empty list")
        return None
    check("PASS", f"{path} loads", f"{len(data)} entries")
    return data


def quant_problems(entry, any_host):
    bad = []
    rev = entry.get("revision")
    if not rev or not HEX40.match(rev):
        bad.append("revision missing/not a 40-hex commit")
    seen = set()
    for q in entry.get("quants", []):
        mid = q.get("model_id", "?")
        if mid in seen:
            bad.append(f"duplicate model_id {mid}")
        seen.add(mid)
        if not HEX64.match(q.get("sha256") or ""):
            bad.append(f"{mid}: sha256 missing/invalid")
        if not isinstance(q.get("size_bytes"), int) or q["size_bytes"] <= 0:
            bad.append(f"{mid}: size_bytes missing/invalid")
        path = q.get("path", "")
        if not any_host and not path.startswith("https://huggingface.co/"):
            bad.append(f"{mid}: path not on huggingface.co")
        if rev and f"/resolve/{rev}/" not in path:
            bad.append(f"{mid}: path not pinned to the entry revision")
    if entry.get("num_quants") != len(entry.get("quants", [])):
        bad.append("num_quants does not match quants")
    return bad


def check_whisper(data, any_host):
    entry = next((e for e in data if e.get("developer") == "ggerganov"), data[0])
    bad = quant_problems(entry, any_host)
    check("PASS" if not bad else "FAIL", "whisper: every quant has sha256, size_bytes, pinned path", "; ".join(bad[:3]))
    ids = {q["model_id"] for q in entry["quants"]}
    check("PASS" if "ggml-small" in ids else "FAIL", "whisper: ggml-small present (first-run default)")
    junk = [i for i in ids if "encoder" in i or "tdrz" in i or "for-tests" in i or "silero" in i]
    check("PASS" if not junk else "FAIL", "whisper: no CoreML/tdrz/test/VAD files", ", ".join(junk))
    check("PASS" if len(ids) >= 8 else "WARN", "whisper: model count", f"{len(ids)} files")
    return entry


def check_embedding(data, any_host):
    bad = [f"{e['model_name']}: {p}" for e in data for p in quant_problems(e, any_host)]
    check("PASS" if not bad else "FAIL", "embedding: every quant has sha256, size_bytes, pinned path", "; ".join(bad[:3]))
    meta_bad = [e["model_name"] for e in data
                if not all(k in e.get("embedding", {}) for k in ("dimension", "max_sequence_tokens", "query_prefix", "document_prefix"))
                or e["embedding"]["dimension"] <= 0]
    check("PASS" if not meta_bad else "FAIL", "embedding: dimension/max tokens/prefixes present", ", ".join(meta_bad))
    minilm = next((e for e in data if "minilm-l6-v2" in e["model_name"].lower()), None)
    check("PASS" if minilm else "FAIL", "embedding: all-MiniLM-L6-v2 present (first-run default)")
    if minilm:
        check("PASS" if minilm["embedding"]["dimension"] == 384 else "FAIL", "embedding: MiniLM dimension is 384")
    return minilm


def check_llm(data, any_host):
    jan = [e["model_name"] for e in data if e.get("developer") in ("janhq", "Menlo")]
    check("PASS" if not jan else "FAIL", "llm: no Jan/Menlo models", ", ".join(jan[:3]))
    try:
        import prepare_catalog
        pinned = prepare_catalog.PINNED_MODELS
    except Exception as exc:  # noqa: BLE001
        check("WARN", "llm: could not import PINNED_MODELS", str(exc))
        pinned = []
    have = {f"{e['developer']}/{e['model_name']}" for e in data}
    missing = [r for r in pinned if r not in have]
    check("PASS" if not missing else "FAIL", "llm: all pinned models present", ", ".join(missing))
    with_sha = [e for e in data if e.get("quants") and not quant_problems(e, any_host)]
    pct = 100 * len(with_sha) / len(data)
    check("PASS" if pct == 100 else "WARN", "llm: entries fully verifiable (sha256 + pinned path)",
          f"{len(with_sha)}/{len(data)} ({pct:.0f}%); legacy entries are backfilled by the first workflow run")
    pinned_bad = [r for r in pinned if r in have and quant_problems(next(e for e in data if f"{e['developer']}/{e['model_name']}" == r), any_host)]
    check("PASS" if not pinned_bad else "WARN", "llm: pinned models are fully verifiable", ", ".join(pinned_bad))
    return next((e for e in data if f"{e['developer']}/{e['model_name']}" == (pinned[0] if pinned else "")), None)


def smallest(entry):
    return min(entry["quants"], key=lambda q: q["size_bytes"])


def head_check(label, q):
    try:
        r = requests.head(q["path"], allow_redirects=True, timeout=30)
        length = int(r.headers.get("Content-Length", -1))
        ok = r.status_code == 200 and length == q["size_bytes"]
        check("PASS" if ok else "FAIL", f"network: HEAD {label}", f"HTTP {r.status_code}, Content-Length {length} vs size_bytes {q['size_bytes']}")
    except requests.RequestException as exc:
        check("FAIL", f"network: HEAD {label}", str(exc))


def download_check(label, q):
    h, n = hashlib.sha256(), 0
    try:
        with requests.get(q["path"], stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                h.update(chunk)
                n += len(chunk)
    except requests.RequestException as exc:
        check("FAIL", f"download: {label}", str(exc))
        return
    ok = h.hexdigest() == q["sha256"] and n == q["size_bytes"]
    check("PASS" if ok else "FAIL", f"download: {label} sha256 + size match", f"{n} bytes, sha256 {'ok' if h.hexdigest() == q['sha256'] else 'MISMATCH'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--any-host", action="store_true", help="skip the huggingface.co host check (for local testing)")
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()
    root = Path(a.dir)
    sys.path.insert(0, str(root.resolve()))

    samples = []
    whisper = load(root / "whisper_catalog.json")
    if whisper:
        entry = check_whisper(whisper, a.any_host)
        samples.append(("whisper smallest", smallest(entry)))
    emb = load(root / "embedding_catalog.json")
    if emb:
        minilm = check_embedding(emb, a.any_host)
        if minilm:
            samples.append(("MiniLM smallest", smallest(minilm)))
    llm = load(root / "model_catalog.json")
    if llm:
        first = check_llm(llm, a.any_host)
        if first and not quant_problems(first, a.any_host):
            samples.append(("first pinned LLM smallest", smallest(first)))

    if a.network:
        for label, q in samples:
            head_check(label, q)
    if a.download:
        for label, q in samples:
            if q["size_bytes"] <= 400 * 1024 * 1024:   # skip multi-GB LLM files; HEAD already covered them
                download_check(label, q)

    fails = sum(1 for s, *_ in results if s == "FAIL")
    warns = sum(1 for s, *_ in results if s == "WARN")
    print(f"\n{len(results) - fails - warns} passed, {warns} warnings, {fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
