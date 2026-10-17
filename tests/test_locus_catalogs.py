"""Tests for the Locus additions: sha256-carrying entries, the whisper and embedding catalogs.

No network: Hub responses are built by hand in the shape `GET /api/models/<repo>?blobs=true` returns.
"""

import hashlib
from unittest.mock import patch

import pytest
import requests

import catalog_helpers as helpers
import prepare_catalog as llm
import prepare_embedding_catalog as emb
import prepare_whisper_catalog as whisper

REV = "c0ffee00" * 5


def sib(name, size=1_000_000, sha=True):
    s = {"rfilename": name, "size": size}
    if sha:
        s["lfs"] = {"sha256": hashlib.sha256(name.encode()).hexdigest(), "size": size, "pointerSize": 134}
    return s


def detail(*siblings, sha=REV):
    d = {"id": "x/y", "downloads": 10, "createdAt": "2026-01-01T00:00:00.000Z",
         "pipeline_tag": "automatic-speech-recognition", "siblings": list(siblings)}
    if sha:
        d["sha"] = sha
    return d


# --- build_quant ------------------------------------------------------------

def test_build_quant_keeps_jans_fields_and_adds_size_bytes_and_sha256():
    q = helpers.build_quant("o/r", REV, sib("m-Q4_K_M.gguf", 2_500_000_000))
    assert set(q) == {"model_id", "path", "file_size", "size_bytes", "sha256"}
    assert q["model_id"] == "m-Q4_K_M"
    assert q["file_size"] == helpers.convert_bytes_to_human_readable(2_500_000_000)
    assert q["size_bytes"] == 2_500_000_000
    assert q["sha256"] == hashlib.sha256(b"m-Q4_K_M.gguf").hexdigest()


def test_build_quant_pins_the_path_to_the_commit_not_main():
    assert helpers.build_quant("o/r", REV, sib("m.gguf"))["path"] == f"https://huggingface.co/o/r/resolve/{REV}/m.gguf"


def test_build_quant_keeps_dots_and_subfolders_in_model_id():
    assert helpers.build_quant("o/r", REV, sib("ggml-small.en-q5_1.bin"))["model_id"] == "ggml-small.en-q5_1"
    assert helpers.build_quant("o/r", REV, sib("Q4_K_M/m.Q4_K_M.gguf"))["model_id"] == "Q4_K_M/m.Q4_K_M"


@pytest.mark.parametrize("bad", [sib("m.gguf", sha=False), {"rfilename": "m.gguf"}, {"size": 5}, sib("m.gguf", size=0)])
def test_build_quant_refuses_files_without_a_verifiable_checksum(bad):
    assert helpers.build_quant("o/r", REV, bad) is None


# --- LLM catalog (Jan's prepare_catalog.py) -----------------------------------

def test_llm_entries_carry_sha256_revision_and_pinned_paths():
    entry = llm.process_model_details(
        "unsloth/gemma-3-4b-it-GGUF",
        {**detail(sib("gemma-3-4b-it-Q4_K_M.gguf", 2_490_000_000), sib("mmproj-F16.gguf", 850_000_000)),
         "pipeline_tag": "image-text-to-text"},
    )
    assert entry["revision"] == REV
    quant, mmproj = entry["quants"][0], entry["mmproj_models"][0]
    for item in (quant, mmproj):
        assert len(item["sha256"]) == 64 and item["path"].startswith(f"https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/{REV}/")
    assert entry["num_quants"] == 1 and entry["num_mmproj"] == 1


def test_llm_entry_skips_files_without_sha_and_repos_without_a_commit():
    entry = llm.process_model_details(
        "o/m", {**detail(sib("m-Q4_K_M.gguf"), sib("m-Q8_0.gguf", sha=False)), "pipeline_tag": "text-generation"})
    assert [q["model_id"] for q in entry["quants"]] == ["m-Q4_K_M"]
    assert llm.process_model_details("o/m", {**detail(sib("m-Q4_K_M.gguf"), sha=None), "pipeline_tag": "text-generation"}) is None


def test_jans_own_models_are_excluded_and_not_pinned():
    assert {"janhq", "Menlo"} <= llm.BLACKLISTED_DEVELOPERS
    assert not any(r.split("/")[0] in {"janhq", "Menlo"} for r in llm.PINNED_MODELS)
    assert llm.process_model_details("janhq/Jan-v3-4B-gguf", {**detail(sib("j-Q4_K_M.gguf")), "pipeline_tag": "text-generation"}) is None


def test_pinned_models_are_the_locus_picks():
    assert llm.PINNED_MODELS[0] == "unsloth/Qwen3-4B-Instruct-2507-GGUF" and len(llm.PINNED_MODELS) == 6


def test_description_step_degrades_without_credentials(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(llm, "client", None)
    with pytest.raises(RuntimeError):  # process_model_details catches this and keeps the old description
        llm.summarize_readme_one_liner("some readme")


# --- whisper catalog ----------------------------------------------------------

@pytest.mark.parametrize("name", ["ggml-small.bin", "ggml-small.en.bin", "ggml-small-q5_1.bin", "ggml-small.en-q8_0.bin",
                                  "ggml-large-v3.bin", "ggml-large-v3-turbo-q5_0.bin", "ggml-tiny.en-q5_1.bin"])
def test_whisper_weights_accepted(name):
    assert whisper.is_whisper_weight(name)


@pytest.mark.parametrize("name", ["ggml-small-encoder.mlmodelc.zip", "ggml-small.en-tdrz.bin", "for-tests-ggml-tiny.bin",
                                  "ggml-silero-v5.1.2.bin", "ggml-large-v3.en.bin", "README.md", "ggml-small.bin.part"])
def test_whisper_non_weights_rejected(name):
    assert not whisper.is_whisper_weight(name)


def test_whisper_entry_uses_jans_shape():
    d = detail(sib("ggml-small.bin", 488_000_000), sib("ggml-small-q5_1.bin", 190_000_000), sib("ggml-tiny.bin", 77_000_000),
               sib("ggml-small-encoder.mlmodelc.zip"), sib("README.md", 100, sha=False))
    entry = whisper.process_model_details("ggerganov/whisper.cpp", d)
    assert (entry["model_name"], entry["developer"], entry["revision"]) == ("whisper.cpp", "ggerganov", REV)
    assert entry["num_quants"] == 3 == len(entry["quants"])
    assert [q["model_id"] for q in entry["quants"]] == ["ggml-small", "ggml-small-q5_1", "ggml-tiny"]
    assert entry["readme"] == f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{REV}/README.md"
    assert entry["quants"][0]["path"].endswith(f"/resolve/{REV}/ggml-small.bin")


def test_whisper_repo_without_weights_yields_nothing():
    assert whisper.process_model_details("o/r", detail(sib("README.md", sha=False))) is None


def test_whisper_run_writes_catalog_and_prints_summary_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(whisper, "OUTPUT_FILE", str(tmp_path / "whisper_catalog.json"))
    with patch.object(whisper, "fetch_repo_detail", return_value=detail(sib("ggml-small.bin"))):
        whisper.get_whisper_catalog()
    assert (tmp_path / "whisper_catalog.json").exists()
    assert "Updated or Added 1 models; catalog total now 1" in capsys.readouterr().out  # the workflow greps this


def test_whisper_run_never_overwrites_with_an_empty_catalog(tmp_path, monkeypatch):
    out = tmp_path / "whisper_catalog.json"
    out.write_text("[]")
    monkeypatch.setattr(whisper, "OUTPUT_FILE", str(out))
    with patch.object(whisper, "fetch_repo_detail", side_effect=requests.exceptions.ConnectionError("down")):
        with pytest.raises(SystemExit):
            whisper.get_whisper_catalog()
    assert out.read_text() == "[]"


# --- embedding catalog ---------------------------------------------------------

META = {"repos": ["bad/repo", "good/repo"], "include": r"^tiny", "dimension": 384, "max_sequence_tokens": 256,
        "query_prefix": "", "document_prefix": "", "description": "d", "required": True}


def fake_fetch(repos):
    def fetch(repo_id):
        if repo_id not in repos:
            raise requests.exceptions.HTTPError("404")
        return repos[repo_id]
    return fetch


def test_embedding_entry_has_jans_shape_plus_embedding_facts():
    entry = emb.process_model_details(
        "good/repo", detail(sib("tiny-Q8_0.gguf", 30_000_000), sib("tiny-f16.gguf", 45_000_000), sib("other-Q8_0.gguf"),
                            sib("mmproj-tiny.gguf")), META)
    assert [q["model_id"] for q in entry["quants"]] == ["tiny-Q8_0", "tiny-f16"]  # smallest first, others filtered
    assert entry["embedding"] == {"dimension": 384, "max_sequence_tokens": 256, "query_prefix": "", "document_prefix": ""}
    assert entry["revision"] == REV and entry["num_quants"] == 2


def test_embedding_falls_back_to_the_next_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "OUTPUT_FILE", str(tmp_path / "e.json"))
    with patch.object(emb, "fetch_repo_detail", fake_fetch({"good/repo": detail(sib("tiny-Q8_0.gguf"))})):
        emb.get_embedding_catalog([META])
    import json
    assert json.loads((tmp_path / "e.json").read_text())[0]["developer"] == "good"


def test_missing_required_embedding_model_fails_and_keeps_the_old_file(tmp_path, monkeypatch):
    out = tmp_path / "e.json"
    out.write_text("[]")
    monkeypatch.setattr(emb, "OUTPUT_FILE", str(out))
    with patch.object(emb, "fetch_repo_detail", fake_fetch({})):
        with pytest.raises(SystemExit, match="Required embedding"):
            emb.get_embedding_catalog([META])
    assert out.read_text() == "[]"


def test_optional_missing_embedding_model_is_omitted(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "OUTPUT_FILE", str(tmp_path / "e.json"))
    optional = {**META, "repos": ["nope/none"], "required": False}
    with patch.object(emb, "fetch_repo_detail", fake_fetch({"good/repo": detail(sib("tiny-Q8_0.gguf"))})):
        emb.get_embedding_catalog([META, optional])
    import json
    assert len(json.loads((tmp_path / "e.json").read_text())) == 1


def test_shipped_embedding_config_is_well_formed():
    assert any(m.get("required") for m in emb.EMBEDDING_MODELS)
    for m in emb.EMBEDDING_MODELS:
        assert m["repos"] and m["dimension"] > 0 and m["max_sequence_tokens"] > 0 and m["description"]
        assert "query_prefix" in m and "document_prefix" in m
    nomic = next(m for m in emb.EMBEDDING_MODELS if "nomic" in m["repos"][0])
    assert nomic["document_prefix"] == "search_document: "


# --- backfill of entries written before sha256 support --------------------------

def test_missing_locus_keys_flags_legacy_entries_for_refetch():
    legacy = {"model_name": "m", "quants": [{"model_id": "m-Q4_K_M", "path": "p", "file_size": "1 GB"}]}
    assert llm.missing_locus_keys(legacy) == ["revision", "quants.sha256"]
    done = {"revision": REV, "quants": [{"model_id": "m", "path": "p", "file_size": "1 GB", "size_bytes": 1, "sha256": "f" * 64}]}
    assert llm.missing_locus_keys(done) == []
    assert llm.missing_locus_keys({"revision": REV, "quants": []}) == []


def test_both_passes_use_the_locus_completeness_check():
    """Regression: Pass 2 once skipped every legacy entry because only Pass 1 had the check."""
    import inspect
    source = inspect.getsource(llm)
    assert source.count("missing_locus_keys(existing_entry)") == 2
