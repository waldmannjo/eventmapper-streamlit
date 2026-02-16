# Optimization Audit — Eventmapper

**Date:** 2026-02-16
**Branch:** `feature/phase1-quick-wins`
**Scope:** Performance, code quality, architecture, UX

---

## 1. Critical Issues

### 1.1 Dead code: duplicate `run_mapping_step4()`

**File:** `backend/mapper.py:162-277`

Python has two definitions of `run_mapping_step4()`. Only the second (line 353) is live — the first (line 162) is silently overwritten and never executed. This is 115 lines of dead code that:
- Lacks BM25 scoring, keyword boost, and LLM fallback (features only in the second definition)
- Confuses code review and maintenance
- Creates false search results when grepping for logic

**Fix:** Delete lines 162-277 entirely.

---

### 1.2 Inconsistent OpenAI API usage

**Files:** `backend/merger.py:129` vs `backend/analyzer.py:95`, `backend/extractor.py:64`, `backend/mapper.py:302`

Three modules use the Chat Completions API (`client.chat.completions.create()` with `messages=`), but `merger.py` uses the Responses API (`client.responses.create()` with `input=`, accessing `response.output_text`). Both APIs exist in OpenAI SDK >=1.66, but mixing them creates inconsistency and confusion.

```python
# merger.py:129 — Responses API
response = client.responses.create(
    model=model_name,
    input=[...],
)
code = response.output_text

# analyzer.py:95 — Chat Completions API
response = client.chat.completions.create(
    model=model_name,
    messages=[...],
)
return json.loads(response.choices[0].message.content)
```

**Fix:** Standardize on one API style across all modules. If the Responses API is preferred (newer, simpler), migrate all modules. Otherwise, convert `merger.py` to Chat Completions.

---

### 1.3 `asyncio.run()` inside Streamlit

**File:** `backend/mapper.py:522`

`asyncio.run()` creates a new event loop each call. If Streamlit or any library already has a running loop, this raises `RuntimeError: asyncio.run() cannot be called from a running event loop`. Currently works in some environments but is fragile.

**Fix:** Use `nest_asyncio` as a quick fix, or replace with `concurrent.futures.ThreadPoolExecutor` for more robust sync/async bridging.

---

## 2. Performance Bottlenecks

### 2.1 Historical embeddings recomputed on cold start

**File:** `backend/mapper.py:106-134` (`load_history_examples`)

On first run (or after code change), all 11k+ historical texts are re-embedded via OpenAI API. This costs ~$2-3 per cold start and adds significant latency. `@st.cache_resource` only persists within a single Streamlit server process — any code edit or restart triggers re-embedding.

**Fix:** Persist embeddings to disk (numpy `.npy` or pickle) alongside a version hash of the source data + `EMB_DIMENSIONS`. Load from disk if hash matches; regenerate only when source data changes.

---

### 2.2 Cross-encoder called per-row instead of batched

**File:** `backend/mapper.py:455`

Inside the per-row loop, `ce_model.predict(ce_pairs)` is called once per row with only 10 pairs. The cross-encoder model's predict method supports batching — calling it once with all pairs (N rows x 10 pairs) amortizes model overhead.

```python
# Current: N calls, 10 pairs each
for i, v in enumerate(q_vecs):
    ...
    ce_scores = ce_model.predict(ce_pairs)  # line 455, 10 pairs

# Better: 1 call, N*10 pairs, then reshape
all_ce_pairs = [...]  # collect all pairs
all_ce_scores = ce_model.predict(all_ce_pairs)
# reshape to (N, 10)
```

**Expected gain:** 3-10x speedup on the cross-encoder phase, depending on dataset size.

---

### 2.3 No embedding memoization for repeated texts

**File:** `backend/mapper.py:74-103` (`embed_texts`)

AEB code descriptions (31 texts) are re-embedded every time Step 4 runs, even though they never change. The function has no deduplication or caching beyond what `@st.cache_resource` provides at the `load_history_examples` level.

**Fix:** Cache AEB code embeddings separately (they're static). Consider a text-hash → embedding dict for input texts that may repeat across runs.

---

### 2.4 `sigmoid()` redefined inside loop

**File:** `backend/mapper.py:465`

```python
for i, v in enumerate(q_vecs):
    ...
    def sigmoid(x): return 1 / (1 + np.exp(-x))  # redefined every iteration
```

**Fix:** Move to module-level.

---

### 2.5 Per-row cosine similarity against all codes

**File:** `backend/mapper.py:429`

For each input row, `cosine_similarity(v.reshape(1,-1), code_vecs)` computes similarity against all 31 codes. This is fine for small datasets but could be vectorized into a single matrix operation:

```python
# Current: loop
for i, v in enumerate(q_vecs):
    sims = cosine_similarity(v.reshape(1,-1), code_vecs).ravel()

# Better: single call
all_sims = cosine_similarity(q_vecs, code_vecs)  # (N, 31) matrix
```

This also eliminates the per-row loop for the BM25 + keyword boost phase, which can be vectorized similarly.

---

## 3. Code Quality

### 3.1 Bare `except:` silently swallows all errors

**File:** `backend/extractor.py:80`

```python
def preview_csv_string(csv_str):
    ...
    try:
        return pd.read_csv(io.StringIO(csv_str), sep=";", on_bad_lines='skip')
    except:
        return pd.DataFrame()
```

This catches `KeyboardInterrupt`, `SystemExit`, and everything else. No logging, no indication of failure.

**Fix:** `except (ValueError, pd.errors.ParserError) as e:` with `print()` or `logging.warning()`.

---

### 3.2 Generic `except Exception` throughout backend

**Files and lines:**
- `backend/mapper.py:99` — embedding batch errors
- `backend/mapper.py:132` — history loading errors
- `backend/mapper.py:312` — LLM classification errors
- `backend/mapper.py:531` — async batch errors
- `backend/merger.py:152` — AI transformation errors
- `app.py:84` — debug file load errors

All catch `Exception` broadly and either `print()` or silently continue. This masks upstream bugs.

**Fix:** Catch specific exceptions (`requests.RequestException`, `json.JSONDecodeError`, `openai.APIError`, etc.). Add structured logging.

---

### 3.3 Imports not at top of file

**File:** `backend/mapper.py:278-279`

```python
import asyncio
from openai import AsyncOpenAI
```

These appear mid-file, after the first (dead) `run_mapping_step4()` definition. Standard Python convention (PEP 8) places all imports at module top.

**Fix:** Move to top of file with other imports.

---

### 3.4 `MAPPER_CONFIG` declared but never consumed

**File:** `app.py:40-47` (declaration) vs `backend/mapper.py` (ignores it)

```python
# app.py:40-47
MAPPER_CONFIG = {
    "use_multilingual_ce": True,    # mapper.py doesn't read this
    "use_bm25": True,               # mapper.py doesn't read this
    "use_keyword_boost": True,      # mapper.py doesn't read this
    "embedding_dimensions": 1024,   # mapper.py hardcodes EMB_DIMENSIONS = 1024
    "knn_threshold": 0.93,          # mapper.py hardcodes KNN_DIRECT_MATCH_THRESHOLD = 0.93
    "confidence_threshold": 0.60,   # passed via slider, not from config
}
```

The config exists but has zero effect. All values are hardcoded in `mapper.py` (lines 31, 205/400, 437, 446, etc.).

**Fix:** Either wire `MAPPER_CONFIG` into `run_mapping_step4()` as a parameter, or delete the config dict to avoid false confidence that it controls anything.

---

### 3.5 Hardcoded magic numbers

**File:** `backend/mapper.py`

| Value | Line(s) | Purpose |
|-------|---------|---------|
| `0.93` | 205, 400 | k-NN direct match threshold |
| `10` | 237, 446 | Top-K prefilter count |
| `0.7 / 0.3` | 437 | Embedding vs BM25 weight split |
| `0.5` | 67 | Max keyword boost cap |
| `0.1` | 67 | Per-keyword boost increment |
| `15` | 321 | Async semaphore concurrency limit |
| `500` | 74 | Embedding batch size |

**Fix:** Define as named constants at module top or accept from config.

---

### 3.6 No type hints on backend functions

**Files:** `backend/mapper.py`, `backend/merger.py`, `backend/analyzer.py`, `backend/extractor.py`

No function signatures have return type annotations. Parameter types are mostly unspecified (except `threshold: float` in `run_mapping_step4`). Makes IDE support and static analysis ineffective.

**Fix:** Add type hints to public functions (low priority, but improves maintainability).

---

### 3.7 DataFrame mutation without `.copy()`

**File:** `app.py:80`

```python
st.session_state.df_merged = df_d  # no .copy(), potential aliasing
```

Also at `app.py:239`:
```python
df_m = logic.merge_data_step3(st.session_state.extraction_res)
st.session_state.df_merged = df_m  # no .copy()
```

If `merge_data_step3` returns a view or shared reference, subsequent mutations could have unintended side effects.

**Fix:** Always `.copy()` when assigning to session state, or ensure backend functions return independent DataFrames.

---

### 3.8 `exec()` on LLM-generated code

**File:** `backend/merger.py:148`

```python
exec(code, {}, local_vars)
```

The AI transformation feature executes arbitrary Python code generated by the LLM. The comment at line 145 acknowledges this. For a local prototype this is acceptable, but for any shared deployment this is a code injection risk.

**Fix:** For production: use a sandbox (e.g., RestrictedPython, subprocess with resource limits). For now: document the risk prominently and restrict to local use.

---

## 4. Architecture Concerns

### 4.1 Streamlit-specific code in backend modules

**File:** `backend/mapper.py:24, 36, 70, 105`

Backend modules import `streamlit` and use `@st.cache_resource`. This couples business logic to the UI framework and makes the backend untestable without Streamlit.

```python
import streamlit as st  # line 24
@st.cache_resource      # lines 36, 70, 105
```

**Fix:** Move caching to the UI layer or use framework-agnostic caching (e.g., `functools.lru_cache`, disk-based cache).

---

### 4.2 No data validation between pipeline steps

Steps pass data (JSON dicts, CSV strings, DataFrames) without schema validation:
- Step 1 → Step 2: JSON with `status_candidates` / `reason_candidates` — no schema check (`app.py:136-137`)
- Step 2 → Step 3: JSON with `mode`, `status_csv`, etc. — no validation (`merger.py:9-12`)
- Step 3 → Step 4: DataFrame — column names assumed but not validated (`mapper.py:363-364`)

A malformed LLM response in Step 1 or 2 silently propagates until Step 4 fails, wasting time and API costs.

**Fix:** Add lightweight validation (assert required keys/columns exist) at each step boundary. Pydantic models are ideal but even simple assertions help.

---

### 4.3 No retry logic for API calls

**Files:** `backend/analyzer.py:95`, `backend/extractor.py:64`, `backend/mapper.py:95,302`

All OpenAI API calls are fire-once. Network timeouts, rate limits (`429`), or transient errors cause immediate failure with no recovery.

**Fix:** Add retry with exponential backoff. The `openai` SDK has built-in retry support (`max_retries` parameter on client initialization), or use `tenacity`.

---

### 4.4 No error recovery in multi-step pipeline

If Step 4 fails midway (e.g., API quota exhausted at row 500 of 1000), all progress is lost. The user must restart from Step 4, re-embedding everything.

**Fix:** Save partial results to `st.session_state` periodically during Step 4. On restart, offer to resume from last checkpoint.

---

### 4.5 CSV separator hardcoded

**Files:** `backend/extractor.py:79`, `backend/merger.py` (via `preview_csv_string`)

```python
pd.read_csv(io.StringIO(csv_str), sep=";", on_bad_lines='skip')
```

The LLM prompt requests semicolons, but LLMs sometimes produce commas or tabs instead. `on_bad_lines='skip'` silently drops rows that don't match.

**Fix:** Use `sep=None, engine='python'` for auto-detection, or try multiple separators with fallback.

---

### 4.6 Global SSL disable affects entire process

**Files:** `app.py:2-16`, `backend/mapper.py:8-20`

SSL verification is disabled globally via environment variables and monkey-patching `ssl.create_default_context`. This is documented as intentional for corporate proxy compatibility, but it disables SSL for ALL outbound connections (including OpenAI API calls carrying the API key).

**Current status:** Documented constraint in CLAUDE.md. No action needed unless deploying outside the corporate network.

---

## 5. UX Improvements

### 5.1 No file size validation on upload

**File:** `app.py:101`

```python
uploaded_file = st.file_uploader("Datei hochladen", type=["pdf", "xlsx", "csv", "txt"])
```

No size limit. A very large file will cause the LLM analysis step to fail (token limit) or cost excessively, with no warning.

**Fix:** Check `uploaded_file.size` after upload and warn if above a threshold (e.g., 10MB).

---

### 5.2 No API cost estimate before Step 4

Users have no indication of how many API calls (embeddings + LLM fallback) Step 4 will make. For large datasets, embedding costs alone can be significant.

**Fix:** Show an estimate before starting: row count, expected embedding calls, estimated LLM fallback rows (based on threshold), approximate cost.

---

### 5.3 DataFrame display without pagination

**Files:** `app.py:255, 358`

```python
st.dataframe(st.session_state.df_merged.head(), width='stretch')  # line 255 — only head()
st.dataframe(st.session_state.df_final, width="stretch")          # line 358 — ALL rows
```

Line 255 shows only `head()` (5 rows), which is fine. But line 358 renders the entire final DataFrame. For 10k+ rows, this freezes the browser.

**Fix:** Add pagination or limit display to first N rows with a "show all" option.

---

### 5.4 Confidence threshold slider description is ambiguous

**File:** `app.py:326-330`

```python
threshold = st.slider(
    "LLM-Schwelle (Confidence Threshold)",
    ...
    help="Werte unter dieser Schwelle werden vom LLM geprüft. Höher = mehr LLM-Aufrufe (teurer, genauer)."
)
```

The help text says "Höher = mehr LLM-Aufrufe" which is correct (higher threshold → more rows fall below it → more LLM calls), but the slider label "LLM-Schwelle" doesn't make the relationship obvious.

**Fix:** Add explicit info: "Bei {N} Zeilen und Schwelle {threshold}: geschätzt ~{X} LLM-Aufrufe".

---

### 5.5 Debug UI always visible

**File:** `app.py:68-85`

The debug file upload expander is always visible in the sidebar. Non-technical users see "Test: Mapping direkt" which is confusing.

**Fix:** Gate behind an environment variable (`EVENTMAPPER_DEBUG=1`) or a query parameter.

---

### 5.6 No extraction quality validation

After Step 2 (extraction), users see a raw preview but have no way to validate quality before proceeding. If the LLM extracted garbage, the user discovers it only at Step 4.

**Fix:** Show summary statistics after Step 2: row count, null percentage per column, sample rows. Optionally flag anomalies (e.g., "0 rows extracted" or "50% null values").

---

## 6. Prioritized Action Plan

### Quick Wins (minimal effort, immediate value)

| # | Action | File | Lines |
|---|--------|------|-------|
| 1 | Delete dead first `run_mapping_step4()` | `mapper.py` | 162-277 |
| 2 | Replace bare `except:` with specific exceptions | `extractor.py` | 80 |
| 3 | Move `sigmoid()` to module level | `mapper.py` | 465 |
| 4 | Move mid-file imports to top | `mapper.py` | 278-281 |
| 5 | Add `.copy()` on session state assignments | `app.py` | 80, 239 |

### Short-term (meaningful performance/quality gains)

| # | Action | Impact |
|---|--------|--------|
| 6 | Persist historical embeddings to disk | Eliminates cold-start re-embedding |
| 7 | Batch cross-encoder predictions | 3-10x speedup on re-ranking |
| 8 | Vectorize cosine similarity (matrix op) | Eliminates per-row loop overhead |
| 9 | Wire `MAPPER_CONFIG` or delete it | Removes misleading config |
| 10 | Standardize OpenAI API style | Consistency across modules |
| 11 | Auto-detect CSV separator | Prevents silent row drops |

### Medium-term (architecture improvements)

| # | Action | Impact |
|---|--------|--------|
| 12 | Add step boundary validation | Catches LLM errors early |
| 13 | Add retry logic for API calls | Resilience to transient failures |
| 14 | Decouple backend from Streamlit | Testability, reusability |
| 15 | Add partial result checkpointing in Step 4 | Prevents total loss on failure |
| 16 | Add cost estimator before Step 4 | User awareness |
| 17 | Fix `asyncio.run()` fragility | Prevents random crashes |
