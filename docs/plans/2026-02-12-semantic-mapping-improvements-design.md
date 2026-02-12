# Semantic Mapping Improvements - Design Document

**Date:** 2026-02-12
**Author:** System Design
**Status:** Draft for Review

## Executive Summary

This document outlines a comprehensive improvement plan for the Eventmapper's semantic mapping engine (Step 4). The goal is to improve mapping accuracy from current baseline to 95%+ through a multi-phase approach combining better models, feature engineering, and active learning.

**Expected Outcomes:**
- 15-20% improvement in mapping accuracy
- 30-40% reduction in LLM API costs through better filtering
- Confidence scores that reflect actual prediction accuracy
- Continuous learning from user corrections

## Current State Analysis

### Existing Pipeline
1. **k-NN Historical Matching** (threshold 0.93): Direct lookup from 11k+ examples
2. **Bi-Encoder** (OpenAI text-embedding-3-large): Semantic similarity
3. **Cross-Encoder Re-Ranking** (ms-marco-MiniLM-L-6-v2): Top-10 candidates
4. **LLM Fallback** (async batch): Low confidence cases (<0.6)

### Key Limitations
- English-only cross-encoder on bilingual (DE/EN) data
- No lexical features (pure neural approach)
- Arbitrary confidence scoring (sigmoid without calibration)
- Single-pass LLM without structured reasoning
- Binary k-NN decision (match or skip)
- Generic embeddings without domain adaptation

## Implementation Plan

### Phase 1: Quick Wins (1-2 weeks)
**Goal:** 5-8% accuracy improvement with minimal risk

#### 1.1 Upgrade Cross-Encoder Model
**Priority:** HIGH | **Effort:** LOW | **Impact:** HIGH

- Replace `cross-encoder/ms-marco-MiniLM-L-6-v2` with multilingual variant
- Test candidates:
  - `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual)
  - `cross-encoder/msmarco-MiniLM-L6-en-de-v1` (EN-DE specific)
- Benchmark on validation set (20% of historical data)
- Select best performing model

**Files to modify:**
- `backend/mapper.py`: Update `CROSS_ENCODER_MODEL_NAME` constant

**Success metric:** 5%+ accuracy improvement on validation set

#### 1.2 Add BM25 Lexical Scoring
**Priority:** HIGH | **Effort:** MEDIUM | **Impact:** MEDIUM

- Install dependency: `rank-bm25`
- Build BM25 index on AEB code keyword-rich descriptions
- Combine BM25 scores with embedding scores (weighted ensemble)
- Initial weights: 70% embedding + 30% BM25

**Implementation:**
```python
from rank_bm25 import BM25Okapi

# Build index (cached)
@st.cache_resource
def build_bm25_index():
    corpus = [code[2].split() for code in CODES]
    return BM25Okapi(corpus)

# At inference
bm25_scores = bm25.get_scores(input_text.split())
combined_score = 0.7 * cosine_sim + 0.3 * normalize(bm25_scores)
```

**Files to modify:**
- `backend/mapper.py`: Add BM25 scoring before cross-encoder
- `requirements.txt`: Add `rank-bm25`

**Success metric:** 3%+ accuracy improvement, especially on keyword-heavy cases

#### 1.3 Keyword Boost Feature
**Priority:** MEDIUM | **Effort:** LOW | **Impact:** MEDIUM

- Extract keywords from AEB code definitions (already in `codes.py`)
- Check for exact keyword matches in input text
- Boost candidate scores by 10% per keyword match

**Implementation:**
```python
def get_keyword_boost(input_text, code_keywords):
    input_lower = input_text.lower()
    matches = sum(1 for kw in code_keywords if kw in input_lower)
    return min(matches * 0.1, 0.5)  # Cap at 50% boost
```

**Files to modify:**
- `backend/mapper.py`: Add keyword extraction and boosting
- `codes.py`: Ensure keywords are easily parseable

**Success metric:** 2%+ improvement on cases with explicit keywords

#### 1.4 OpenAI Embedding Dimension Reduction
**Priority:** MEDIUM | **Effort:** LOW | **Impact:** LOW (cost saving)

- Test dimension reduction: 3072 → 1024 dimensions
- Benchmark accuracy impact on validation set
- If <2% accuracy loss, adopt for cost savings

**Implementation:**
```python
resp = client.embeddings.create(
    model="text-embedding-3-large",
    input=texts,
    dimensions=1024  # 67% cost reduction
)
```

**Files to modify:**
- `backend/mapper.py`: Add `dimensions` parameter to embedding calls

**Success metric:** 30-50% reduction in embedding API costs with <2% accuracy loss

---

### Phase 2: Robust Foundations (2-3 weeks)
**Goal:** Reliable confidence scores and improved k-NN matching

#### 2.1 Weighted k-NN Voting
**Priority:** HIGH | **Effort:** MEDIUM | **Impact:** HIGH

- Replace top-1 k-NN with top-5 weighted voting
- Calculate consensus confidence from vote distribution
- Use adaptive threshold (75% consensus) instead of hard 0.93

**Implementation:**
```python
def weighted_knn_vote(query_vec, df_hist, hist_vecs, top_k=5):
    matches = get_similar_historical_entries(query_vec, df_hist, hist_vecs, top_k)

    votes = {}
    for match in matches:
        code = match['mapped_code']
        score = match['score']
        votes[code] = votes.get(code, 0) + score

    total_weight = sum(votes.values())
    best_code = max(votes, key=votes.get)
    consensus = votes[best_code] / total_weight

    return best_code, consensus
```

**Files to modify:**
- `backend/mapper.py`: Update `get_similar_historical_entries` logic

**Success metric:** 3-5% improvement in k-NN match rate

#### 2.2 Confidence Calibration
**Priority:** HIGH | **Effort:** MEDIUM | **Impact:** HIGH

- Split historical data: 80% train, 20% validation
- Collect raw scores vs. actual correctness on validation set
- Train isotonic regression calibrator
- Apply calibration to all confidence scores

**Implementation:**
```python
from sklearn.isotonic import IsotonicRegression

@st.cache_resource
def train_confidence_calibrator():
    # Load validation predictions and ground truth
    val_scores, val_correct = load_validation_data()
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_scores, val_correct)
    return calibrator

# At inference
raw_score = compute_score(...)
calibrated_conf = calibrator.predict([raw_score])[0]
```

**Files to modify:**
- `backend/mapper.py`: Add calibration module
- New file: `backend/calibration.py`

**Success metric:** Calibrated confidence scores within ±5% of actual accuracy

#### 2.3 Multi-Source Confidence Fusion
**Priority:** MEDIUM | **Effort:** MEDIUM | **Impact:** MEDIUM

- Combine signals: k-NN score, cross-encoder score, BM25 score, keyword matches
- Learn optimal weights via logistic regression on validation data
- Output single fused confidence score

**Implementation:**
```python
def fuse_confidence(knn_score, ce_score, bm25_score, keyword_count):
    # Learned weights from validation set
    confidence = (
        0.40 * knn_score +
        0.35 * calibrate_ce(ce_score) +
        0.15 * bm25_score +
        0.10 * min(keyword_count * 0.1, 0.5)
    )
    return min(confidence, 1.0)
```

**Files to modify:**
- `backend/mapper.py`: Replace simple confidence with fusion
- `backend/calibration.py`: Add fusion logic

**Success metric:** Confidence scores accurately predict success rate per bin

---

### Phase 3: Advanced Techniques (3-4 weeks)
**Goal:** State-of-the-art accuracy through fine-tuning and advanced features

#### 3.1 Fine-Tune Cross-Encoder
**Priority:** HIGH | **Effort:** HIGH | **Impact:** HIGH

- Create training dataset from 11k historical mappings:
  - Positive pairs: (carrier_desc, correct_AEB_desc)
  - Hard negatives: (carrier_desc, top-K wrong codes)
- Split: 80% train, 10% val, 10% test
- Fine-tune multilingual cross-encoder (3-5 epochs)
- Evaluate on held-out test set

**Training setup:**
```python
from sentence_transformers import CrossEncoder, InputExample

# Prepare data
train_samples = []
for _, row in df_hist.iterrows():
    carrier_desc = row['Description']
    correct_code = row['AEB Event Code']
    correct_desc = get_code_desc(correct_code)

    # Positive
    train_samples.append(InputExample(texts=[carrier_desc, correct_desc], label=1.0))

    # Hard negatives (similar but wrong codes)
    wrong_codes = get_top_k_wrong_codes(carrier_desc, k=3)
    for wrong_code in wrong_codes:
        wrong_desc = get_code_desc(wrong_code)
        train_samples.append(InputExample(texts=[carrier_desc, wrong_desc], label=0.0))

# Train
model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
model.fit(
    train_dataloader=DataLoader(train_samples, batch_size=16),
    epochs=3,
    warmup_steps=100
)
model.save('models/cross-encoder-finetuned')
```

**Files to create:**
- `scripts/train_cross_encoder.py`: Training script
- `models/`: Save trained models

**Files to modify:**
- `backend/mapper.py`: Load fine-tuned model

**Success metric:** 8-12% improvement over base cross-encoder

#### 3.2 Enhanced LLM Strategy
**Priority:** MEDIUM | **Effort:** MEDIUM | **Impact:** MEDIUM

- Implement multi-strategy prompting (3 parallel strategies)
- Add chain-of-thought reasoning (two-step: analysis → decision)
- Use contrastive historical examples (positive + negative)
- Add confidence self-assessment

**Implementation:**
```python
strategies = {
    "keyword": "Focus on exact keyword matches",
    "semantic": "Analyze business meaning and process stage",
    "historical": "Follow established historical patterns"
}

async def multi_strategy_llm(input_text, candidates, hist_examples):
    results = []
    for strategy_name, instruction in strategies.items():
        result = await llm_classify(
            input_text,
            candidates,
            strategy_instruction=instruction,
            hist_examples=format_contrastive_examples(hist_examples)
        )
        results.append(result)

    # Majority voting with confidence weighting
    return select_best_result(results)
```

**Files to modify:**
- `backend/mapper.py`: Update `classify_single_row` function
- New file: `backend/llm_strategies.py`

**Success metric:** 10-15% improvement in LLM fallback accuracy

#### 3.3 Historical Clustering
**Priority:** LOW | **Effort:** HIGH | **Impact:** MEDIUM

- Cluster 11k historical embeddings into semantic groups (DBSCAN)
- For each cluster, identify dominant AEB code and consistency
- Use cluster membership as additional signal

**Implementation:**
```python
from sklearn.cluster import DBSCAN

@st.cache_resource
def cluster_historical_data(hist_vecs, df_hist):
    clustering = DBSCAN(eps=0.15, min_samples=5).fit(hist_vecs)
    labels = clustering.labels_

    # Build cluster profiles
    cluster_profiles = {}
    for cluster_id in set(labels):
        if cluster_id == -1: continue

        cluster_mask = labels == cluster_id
        cluster_codes = df_hist[cluster_mask]['AEB Event Code'].value_counts()

        dominant_code = cluster_codes.index[0]
        consistency = cluster_codes.iloc[0] / len(cluster_codes)

        cluster_profiles[cluster_id] = {
            'code': dominant_code,
            'consistency': consistency
        }

    return cluster_profiles, labels

# At inference
cluster_id = find_nearest_cluster(query_vec, hist_vecs, cluster_labels)
if cluster_profiles[cluster_id]['consistency'] > 0.90:
    return cluster_profiles[cluster_id]['code']  # High confidence
```

**Files to modify:**
- `backend/mapper.py`: Add cluster-based matching stage

**Success metric:** 2-4% improvement through cluster patterns

---

### Phase 4: Continuous Learning (2-3 weeks)
**Goal:** System that improves over time through user feedback

#### 4.1 Active Learning Pipeline
**Priority:** MEDIUM | **Effort:** HIGH | **Impact:** HIGH (long-term)

- Identify high-uncertainty predictions for human review
- Build review queue ranked by learning value
- Capture user corrections and add to historical data
- Periodically retrain models with accumulated corrections

**Implementation:**
```python
class ActiveLearner:
    def identify_review_cases(self, predictions):
        review_queue = []
        for pred in predictions:
            uncertainty_score = self.calculate_uncertainty(pred)
            if uncertainty_score > threshold:
                review_queue.append({
                    'input': pred['input'],
                    'prediction': pred['code'],
                    'confidence': pred['confidence'],
                    'uncertainty_score': uncertainty_score,
                    'reason': pred['uncertainty_reason']
                })
        return sorted(review_queue, key=lambda x: x['uncertainty_score'], reverse=True)

    def calculate_uncertainty(self, pred):
        score = 0
        # Low margin between top-1 and top-2
        if pred['top1_score'] - pred['top2_score'] < 0.1:
            score += 10
        # Disagreement between methods
        if pred['knn_code'] != pred['ce_code']:
            score += 8
        # Rare code
        if pred['code_frequency'] < 0.01:
            score += 5
        return score

    def add_correction(self, input_text, wrong_code, correct_code):
        # Add to historical examples
        new_row = {
            'Description': input_text,
            'AEB Event Code': correct_code,
            'source': 'user_correction',
            'timestamp': datetime.now()
        }
        append_to_history(new_row)

        # Invalidate caches
        st.cache_resource.clear()
```

**Files to create:**
- `backend/active_learning.py`: Active learning module
- `data/corrections.jsonl`: Store user corrections

**Files to modify:**
- `app.py`: Add review UI in Step 4
- `backend/mapper.py`: Integrate uncertainty calculation

**Success metric:** Continuous accuracy improvement with user feedback

#### 4.2 Review UI in Streamlit
**Priority:** MEDIUM | **Effort:** MEDIUM | **Impact:** MEDIUM

- Add "Review Uncertain Cases" section in Step 4
- Show top-N uncertain predictions with:
  - Input description
  - Predicted code + confidence
  - Top-3 alternative candidates
  - Option to correct or approve
- Track corrections and retrain periodically

**UI Mock:**
```python
with st.expander("🔍 Review Uncertain Predictions", expanded=False):
    uncertain_cases = active_learner.identify_review_cases(df_final)

    if uncertain_cases:
        st.warning(f"{len(uncertain_cases)} predictions need review")

        for i, case in enumerate(uncertain_cases[:10]):  # Top 10
            st.markdown(f"**Case {i+1}:** {case['input']}")
            st.write(f"Predicted: {case['prediction']} (confidence: {case['confidence']:.2%})")
            st.write(f"Uncertainty reason: {case['reason']}")

            col1, col2 = st.columns([3, 1])
            with col1:
                correct_code = st.selectbox(
                    "Correct code:",
                    options=[case['prediction']] + case['alternatives'],
                    key=f"correct_{i}"
                )
            with col2:
                if st.button("✓ Confirm", key=f"confirm_{i}"):
                    active_learner.add_correction(
                        case['input'],
                        case['prediction'],
                        correct_code
                    )
                    st.success("Correction saved!")
```

**Files to modify:**
- `app.py`: Add review section in Step 4 results

**Success metric:** User adoption of review feature, 50+ corrections per month

#### 4.3 Automated Retraining Pipeline
**Priority:** LOW | **Effort:** MEDIUM | **Impact:** MEDIUM (long-term)

- Monthly: Retrain confidence calibrator with new validated data
- Quarterly: Fine-tune cross-encoder with accumulated corrections
- Track performance metrics over time (accuracy, cost, speed)

**Implementation:**
```bash
# Cron job or GitHub Actions workflow
# Run monthly
python scripts/retrain_calibrator.py

# Run quarterly
python scripts/finetune_cross_encoder.py --include-corrections
```

**Files to create:**
- `scripts/retrain_calibrator.py`: Automated calibration retraining
- `scripts/finetune_cross_encoder.py`: Automated fine-tuning
- `.github/workflows/monthly-retrain.yml`: CI/CD automation

**Success metric:** Automated improvement cycle without manual intervention

---

## Implementation Priorities

### Recommended Sequence

**Week 1-2: Phase 1 (Quick Wins)**
- Upgrade cross-encoder → Immediate 5% improvement
- Add BM25 scoring → Additional 3% improvement
- Keyword boosting → Additional 2% improvement
- Dimension reduction → 40% cost savings

**Week 3-5: Phase 2 (Foundations)**
- k-NN voting → 4% improvement, better historical usage
- Confidence calibration → Reliable confidence scores
- Multi-source fusion → Unified confidence metric

**Week 6-9: Phase 3 (Advanced)**
- Fine-tune cross-encoder → 10% improvement (biggest single gain)
- Enhanced LLM strategy → 12% improvement in LLM cases
- Historical clustering → 3% additional improvement

**Week 10-12: Phase 4 (Continuous Learning)**
- Active learning pipeline → Long-term accuracy growth
- Review UI → User engagement and feedback loop
- Automated retraining → Sustainable improvement

### Expected Cumulative Results

| Phase | Accuracy Gain | LLM Cost Reduction | Confidence Quality |
|-------|---------------|-------------------|-------------------|
| Baseline | 0% | 0% | Poor (uncalibrated) |
| Phase 1 | +10% | +40% | Poor |
| Phase 2 | +14% | +40% | Good (calibrated) |
| Phase 3 | +25% | +60% | Excellent |
| Phase 4 | +25%+ (grows) | +65% | Excellent |

## Technical Architecture Changes

### New Dependencies
```txt
rank-bm25>=0.2.2
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
torch>=2.0.0  # For fine-tuning
```

### New Files Structure
```
backend/
  ├── mapper.py (enhanced)
  ├── calibration.py (new)
  ├── active_learning.py (new)
  ├── llm_strategies.py (new)
  └── feature_engineering.py (new)

scripts/
  ├── train_cross_encoder.py (new)
  ├── retrain_calibrator.py (new)
  └── evaluate_models.py (new)

models/
  ├── cross-encoder-finetuned/ (new)
  └── calibrator.pkl (new)

data/
  ├── corrections.jsonl (new)
  └── validation_split.csv (new)

docs/plans/
  └── 2026-02-12-semantic-mapping-improvements-design.md (this file)
```

### Configuration Changes

Add to `app.py`:
```python
MAPPER_CONFIG = {
    "use_multilingual_ce": True,
    "use_bm25": True,
    "use_keyword_boost": True,
    "embedding_dimensions": 1024,
    "knn_voting_k": 5,
    "knn_consensus_threshold": 0.75,
    "confidence_calibration": True,
    "llm_strategy": "multi",  # "single" or "multi"
    "enable_active_learning": True
}
```

## Testing Strategy

### Validation Dataset
- Hold out 20% of historical data (2,200 examples)
- Stratified by AEB code to ensure coverage
- Never use for training, only for evaluation

### Metrics to Track
1. **Accuracy:** Percentage of correct predictions
2. **Confidence Calibration:** ECE (Expected Calibration Error)
3. **Coverage:** Percentage handled by each stage (k-NN, embedding, LLM)
4. **Cost:** API calls and cost per prediction
5. **Speed:** End-to-end processing time

### A/B Testing Approach
- Run old and new pipeline in parallel on validation set
- Compare metrics side-by-side
- Only deploy if new version shows consistent improvement

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Fine-tuning overfits | HIGH | Use proper train/val/test split, early stopping |
| API cost increase | MEDIUM | Start with dimension reduction, monitor costs |
| Slower inference | MEDIUM | Cache models, use batch processing, profile bottlenecks |
| User confusion with review UI | LOW | Clear instructions, tooltips, optional feature |
| Model drift over time | LOW | Automated monitoring, periodic retraining |

## Success Criteria

### Phase 1 Success
- [ ] 8-10% accuracy improvement on validation set
- [ ] 30%+ reduction in embedding API costs
- [ ] All tests pass, no regression in existing functionality

### Phase 2 Success
- [ ] 12-15% accuracy improvement
- [ ] Confidence scores calibrated within ±5% of actual accuracy
- [ ] Clear documentation of all changes

### Phase 3 Success
- [ ] 20-25% accuracy improvement
- [ ] 50%+ reduction in LLM fallback usage
- [ ] Fine-tuned models versioned and reproducible

### Phase 4 Success
- [ ] Active learning pipeline collecting user corrections
- [ ] Automated retraining working end-to-end
- [ ] Continuous accuracy improvement observable over 3 months

## Rollout Plan

### Development Environment
- Create feature branch: `feature/semantic-mapping-improvements`
- Use git worktree for isolated development
- Commit frequently with clear messages

### Testing
- Unit tests for each new module
- Integration tests for end-to-end pipeline
- Benchmark tests comparing old vs. new performance

### Deployment
- Gradual rollout: Test with small batches first
- Monitor accuracy and costs closely
- Keep old pipeline available for rollback
- Document all changes in CLAUDE.md

## Next Steps

1. **Review this design** with stakeholders
2. **Validate assumptions** on sample data
3. **Create implementation plan** using superpowers:writing-plans
4. **Set up git worktree** for isolated development
5. **Begin Phase 1** implementation

---

**Document Status:** Ready for review and implementation planning
