# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Eventmapper** is a Streamlit application that automates the mapping of logistics event codes (Status and Reason codes) from carrier-specific documents to standardized AEB event codes. It uses OpenAI's GPT models for document analysis and extraction, combined with a hybrid ML approach for semantic mapping.

## Running the Application

```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Run the Streamlit app
streamlit run app.py
```

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## Architecture

### Architecture Diagrams

Visual diagrams are maintained in `.memory/diagrams/`:
- [Folder Structure](.memory/diagrams/structure.md) - Project directory layout and key files
- [Data Flow](.memory/diagrams/logic.md) - Complete 4-step workflow with API interactions
- [Component Relationships](.memory/diagrams/components.md) - Module dependencies and responsibilities
- [UI Workflow](.memory/diagrams/api.md) - Session state management and user interface flow

Review these diagrams to understand the codebase architecture before making changes.

### 4-Step Workflow
The application follows a sequential pipeline where each step produces data for the next:

1. **Step 0 (Upload)**: Extracts text from PDF/Excel/CSV/TXT files → produces `raw_text`
2. **Step 1 (Analysis)**: LLM analyzes document structure to identify status/reason code tables → produces `analysis_res` with table candidates
3. **Step 2 (Extraction)**: LLM extracts data from selected tables → produces `extraction_res` with CSV strings
4. **Step 3 (Merge)**: Combines and cleans extracted data → produces `df_merged` DataFrame
5. **Step 4 (Mapping)**: Maps codes to AEB standards → produces `df_final` DataFrame

### Backend Modules

The `backend/` package contains the core logic, organized by workflow step:

- **`loader.py`**: File ingestion (PDF/Excel/CSV parsing)
- **`analyzer.py`**: Step 1 - LLM-based document structure analysis
- **`extractor.py`**: Step 2 - LLM-based table extraction to CSV
- **`merger.py`**: Step 3 - Data cleaning, merging, and AI-powered transformation
- **`mapper.py`**: Step 4 - Hybrid semantic mapping engine
- **`synonyms.py`**: Domain vocabulary for identifying status/reason code columns

### Hybrid Mapping Strategy (Step 4)

The mapper uses a **multi-stage approach** for high accuracy:

1. **Historical k-NN Matching**: Direct lookup from 11k+ historical mappings (`examples/CES_Umschlüsselungseinträge_all.xlsx`)
2. **Semantic Search**: OpenAI `text-embedding-3-large` embeddings for similarity scoring
3. **Cross-Encoder Re-Ranking**: Sentence Transformers multilingual model (`ms-marco-MiniLM-L-6-v2`) for verification
4. **LLM Fallback**: GPT model with few-shot retrieval for low-confidence cases (below configurable threshold)

Key optimizations:
- **Batch embedding** (500 texts/batch) to reduce API calls
- **Streamlit caching** (`@st.cache_resource`) for Cross-Encoder model and historical embeddings
- Bilingual code definitions in `codes.py` with extensive keyword anchors for better semantic matching

## State Management

The app uses `st.session_state` to persist data across Streamlit reruns:

- `current_step`: Workflow position (0-4)
- `raw_text`: Extracted document text
- `analysis_res`: Structure analysis results (JSON with table candidates)
- `extraction_res`: Extracted CSV strings (mode-dependent: separate or combined)
- `df_merged`: Cleaned DataFrame ready for mapping
- `df_final`: Final mapped results

**Important**: Each step advances `current_step` only after successful completion. Use `st.rerun()` after state updates to refresh the UI.

## SSL/Proxy Workaround

The application **disables SSL verification** globally at startup to handle corporate proxies and self-signed certificates:

```python
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
```

This affects:
- OpenAI API calls (implicitly via urllib3)
- Hugging Face model downloads (Cross-Encoder in `mapper.py`)

**Do not remove this** unless testing in an environment with proper SSL certificates.

## Model Selection

The UI allows per-step model selection (configurable in `MODEL_CONFIG` dict in `app.py`):
- Balance cost vs. quality by choosing appropriate models per task
- Step 4 (mapping) is the most expensive due to batch embeddings
- Consider using cheaper models (e.g., `gpt-5-nano`) for steps 1-2 when processing large documents

## Testing & Debug Mode

**Debug Mode** (Sidebar → "Test: Mapping direkt"):
- Allows direct upload of pre-processed CSV/Excel to skip to Step 3
- Useful for iterating on mapping logic without re-running extraction
- Sets `df_merged` directly and bypasses steps 0-2

**Testing Directory**: Contains sample files for validating extraction and mapping logic.

## Data Files

- **`codes.py`**: Canonical list of AEB event codes with bilingual descriptions and keyword lists
  - Used by mapper for semantic matching
  - Format: `(code, short_desc, long_desc_with_keywords)`

- **`examples/CES_Umschlüsselungseinträge_all.xlsx`**: Historical mapping data
  - Required columns: `Description`, `AEB Event Code`
  - Used for k-NN matching in Step 4
  - Embeddings are cached after first load

- **`backend/synonyms.py`**: Domain vocabulary lists
  - Used by analyzer to identify relevant columns in carrier documents
  - Includes status synonyms (`STATUS_SYNONYMS`) and reason code variants (`REASON_SYNONYMS_*`)

## Common Patterns

### Adding a New Synonym
Edit `backend/synonyms.py` and add to appropriate list (`STATUS_SYNONYMS` or `REASON_SYNONYMS_*`). The analyzer will automatically use them in LLM prompts.

### Extending AEB Codes
Add entries to `codes.py` following the format:
```python
("CODE", "Short Name / DE Name",
 "English description with use cases. "
 "Deutsche Beschreibung. "
 "Keywords: en, de, keywords, for, matching.")
```
Keywords significantly improve semantic matching accuracy.

### Modifying Mapping Logic
The mapper's main entry point is `run_mapping_step4()` in `backend/mapper.py`. The function:
1. Loads historical data and embeddings (cached)
2. Embeds input codes in batches
3. Computes k-NN + semantic + cross-encoder scores
4. Calls LLM for low-confidence cases
5. Returns DataFrame with `AEB Event Code` and `Confidence` columns

Adjust `CROSS_ENCODER_MODEL_NAME` or `EMB_MODEL` constants at the top of `mapper.py` to test different models.

## Git Workflow

This is a git repository. The main branch is `main`.
- Create feature branches for all changes
- Commit frequently with descriptive messages
- Never push directly to main branch
- Add and commit automatically when tasks complete
