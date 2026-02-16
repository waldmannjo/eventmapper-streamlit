# HANDOVER - v0.3.0 Optimization Items

**Date:** 2026-02-16
**Branch:** `main`
**Overall Status:** v0.3.0 merged and pushed to origin

---

## What We Worked On

Implemented 6 short-term optimization items from the optimization audit, following the quick wins already committed in `7453914`. Version bumped from `0.2.0` to `0.3.0`.

### Tasks Completed (6/6)

| Item | Description | Commit |
|------|-------------|--------|
| 11 | Auto-detect CSV separator (`sep=None, engine='python'`) | `c44d0cb` |
| 10 | Migrate to OpenAI Responses API (`client.responses.create`) | `c44d0cb` |
| 9 | Wire `MAPPER_CONFIG` into `run_mapping_step4()` — all thresholds/weights from config | `c44d0cb` |
| 8 | Vectorize cosine similarity — single `cosine_similarity(q_vecs, code_vecs)` matrix op | `c44d0cb` |
| 7 | Batch cross-encoder predictions — two-phase loop, single `predict()` call | `c44d0cb` |
| 6 | Persist historical embeddings to disk (`.npy` + `.pkl` + hash metadata) | `c44d0cb` |

Also added: "What's new?" changelog popover in sidebar UI.

**22 tests passing** across 6 test files.

---

## Key Changes Explained

### Responses API Migration (Item 10)
All three LLM call sites migrated from `client.chat.completions.create()` to `client.responses.create()`:
- `backend/analyzer.py` — sync
- `backend/extractor.py` — sync
- `backend/mapper.py` — async (`await async_client.responses.create()`)

Pattern: `instructions=` for system prompt, `input=` for user text, `text={"format": {"type": "json_object"}}` for JSON mode, extract via `response.output_text`.

### Two-Phase Mapper Loop (Items 7+8)
The main matching loop in `run_mapping_step4()` was restructured:
- **Phase 1**: k-NN check + pre-filtering + collect all cross-encoder pairs into a flat list
- **Phase 2**: Single batched `ce_model.predict(all_ce_pairs)`, then unpack results per row

Cosine similarity is pre-computed as `all_sims = cosine_similarity(q_vecs, code_vecs)` — a `(N, 31)` matrix — before the loop starts.

### Disk Cache (Item 6)
Historical embeddings are cached to:
- `examples/history_embeddings.npy` — numpy array
- `examples/history_df.pkl` — DataFrame pickle
- `examples/history_cache_meta.json` — SHA-256 hash of `EMB_MODEL:EMB_DIMENSIONS:file_mtime:file_size`

On load: if cache files exist and hash matches, load from disk. Otherwise re-embed and save. `@st.cache_resource` still wraps this as a second-level in-memory cache.

### MAPPER_CONFIG (Item 9)
`run_mapping_step4()` now accepts `config=None`. Defaults match previous hardcoded values. No hardcoded thresholds remain in the matching logic — all from config dict.

---

## Known Issues

### OpenAI SDK Version
Using `openai==2.8.1`. The Responses API is supported but newer SDK versions may add type hints or helper methods.

---

## Clear Next Steps

### Short-term
1. **Run validation script** with API key to measure accuracy impact of the optimizations

### Medium-term (Phase 2)
2. Confidence calibration (isotonic regression)
3. Weighted k-NN voting (top-5 instead of top-1)
4. Fine-tune cross-encoder on domain data

See: `docs/plans/2026-02-12-semantic-mapping-improvements-design.md` (Phase 2 section)

---

## Files Modified in This Session

| File | What Changed |
|------|-------------|
| `app.py` | v0.3.0, expanded `MAPPER_CONFIG`, passes config to mapper, changelog popover |
| `backend/analyzer.py` | Responses API migration |
| `backend/extractor.py` | Responses API migration + auto-detect CSV separator |
| `backend/mapper.py` | Responses API (async), disk cache, vectorized cosine sim, batched CE, config parameter |
| `.gitignore` | Added embedding cache files |
| `CLAUDE.md` | Updated cache clearing instructions |
| `ARCHITECTURE.md` | Updated optimizations list, MAPPER_CONFIG docs |
| `HANDOVER.md` | Rewritten for v0.3.0 session |
