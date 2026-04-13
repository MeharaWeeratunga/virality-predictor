

# PrePubVirality
# Pre-Publication Virality Prediction of Scientific Research Papers

## Project Overview 📖

This project is a machine learning system that predicts whether a scientific research paper will go viral — attracting significant social media and news attention — using only information available **before publication**. No citation data, no post-publication metrics.

Virality is defined as a paper ranking in the **top 10% of Altmetric attention scores** within its publication year cohort. Altmetric scores aggregate attention from news outlets, Twitter/X, blogs, policy documents, and Mendeley readership.

The system is deployed as an interactive Streamlit web application where researchers can input a paper's title, abstract, and basic metadata to obtain real-time virality predictions with explanations.

---

## Objectives 🎯

- Predict whether a research paper will go viral at the time of publication
- Use **only pre-publication signals** — title, abstract, venue, author count, publication year
- Leverage fine-tuned **SPECTER2** domain-specific transformer embeddings for semantic understanding
- Provide an interpretable web interface with probability outputs and feature explanations
- Demonstrate fairness-aware design by removing author h-index features

---

## Key Features 🚀

### Single Paper Prediction ✅
Users enter:
- Paper title and abstract
- Research field (Computer Science or Medicine)
- Publication type, venue/journal, reference count, author count, publication year
- All metadata fields are **optional** — mean imputation handles missing values

The system outputs:
- Viral / Not Viral prediction
- Virality probability score
- Individual model agreement (XGBoost, Logistic Regression, Random Forest, Ensemble)
- Feature snapshot showing what signals drove the prediction
- Explanation of positive and negative signals

### Paper Comparison Mode ✅
Compare two papers side-by-side:
- Which paper is more likely to go viral
- Relative probability difference
- Per-model agreement breakdown

### Model Insights Tab ✅
Directly loaded from the saved model bundle:
- Full results table (ROC-AUC, PR-AUC, F1 per model)
- Ablation study results
- Frozen vs fine-tuned SPECTER2 comparison
- Year-stratified evaluation (2023–2025)
- Topic-level virality analysis (K-Means clusters)

### Example Papers ✅
Four pre-loaded examples covering viral Medicine RCTs and non-viral CS papers for quick demonstration.

---

## Machine Learning Approach 🧠

### Dataset
- **94,435 papers** from Semantic Scholar API
- **Fields**: Computer Science + Medicine
- **Years**: 2018–2026
- **Labels**: Altmetric attention score, top 10% per year cohort = viral
- **Altmetric hit rate**: 53.2% of papers have Altmetric scores

### Feature Set
| Feature Group | Features |
|---|---|
| Semantic | SPECTER2 embeddings (768-dim) from title + abstract |
| Authorship | num\_authors |
| Venue | is\_top\_venue (binary whitelist), top\_venue × reference\_count |
| Temporal | pub\_year\_centred (years from training cutoff) |
| Reference | reference\_count, reference\_count\_log |
| Title structure | title\_len, title\_has\_question, title\_has\_colon, title\_has\_number |
| Abstract structure | abstract\_len, abstract\_has\_eq, abstract\_has\_table |

**Note**: Author h-index was deliberately excluded on fairness grounds — it penalises early-career researchers and papers by authors with limited publication history.

### SPECTER2 Fine-Tuning
SPECTER2 (`allenai/specter2_base`) is fine-tuned on virality labels using:
- Top 2 transformer layers + pooler unfrozen (13.6% of 110M parameters)
- 5 epochs, AdamW, cosine scheduler with warmup
- Class-weighted BCE loss (9:1 negative:positive ratio)
- Best checkpoint selected by validation PR-AUC

### Models
| Model | Role |
|---|---|
| XGBoost | Raw embeddings + metadata, raw probabilities (no calibration wrapper) |
| Random Forest | Raw embeddings + metadata, balanced subsample |
| Logistic Regression | PCA(100) embeddings + metadata |
| Baseline | Metadata only — quantifies embedding contribution |
| **Ensemble** | **XGB×0.4 + LR×0.4 + RF×0.2** |

### Labelling Methodology
- **Per-cohort labels**: viral threshold computed within each publication year independently, preventing label drift across years
- **2026 exclusion**: papers published in 2026 are excluded from primary test metrics — at the March 2026 Altmetric snapshot they had ≤3 months of exposure, making viral labels structurally unreliable. They are evaluated separately for context (ROC ≈ 0.785, expected degradation)
- **Mature papers**: test papers < 12 months old at Altmetric snapshot excluded from primary metrics
- **Temporal split**: train ≤ 2022, test 2023–2025 — strictly no future data leakage

---

## Results 📊

### Primary Metrics (Test 2023–2025, n=33,564 · Mature subset n=26,306)

| Model | ROC-AUC (all) | PR-AUC (all) | ROC-AUC (mature) | PR-AUC (mature) | F1 |
|---|---|---|---|---|---|
| Baseline (meta only) | 0.8263 | 0.4076 | 0.8320 | 0.4095 | 0.405 |
| Logistic Regression | 0.9060 | 0.6086 | 0.9128 | 0.6177 | 0.542 |
| Random Forest | 0.9039 | 0.5853 | 0.9110 | 0.5933 | 0.543 |
| **XGBoost** | **0.9076** | **0.6142** | **0.9143** | **0.6211** | **0.504** |
| **Ensemble** | **0.9073** | **0.6117** | **0.9142** | **0.6202** | **0.530** |

- **Embedding PR-AUC lift over baseline**: +50.7% relative
- **PR-AUC vs random**: 6.1× random baseline (random = 0.10)
- **Fine-tuning gain**: +6.0% PR-AUC relative over frozen SPECTER2

### Key Findings
- SPECTER2 embeddings are the dominant signal; embeddings alone outperform metadata alone on PR-AUC
- Top SHAP features: num\_authors, pub\_type\_Journal, reference\_count\_log, reference\_count, is\_top\_venue
- Topic virality spread: 77.2 pp across 20 semantic clusters (clinical trials 77.4% viral vs CS graph papers 0.2%)
- Year-stratified: ROC declines 0.920 → 0.913 → 0.889 from 2023 to 2025; precision approaches the 0.30 floor in 2025

---

## Project Structure 📁

```
virality-predictor/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── viral_predictor_v12_altmetric_no_hindex.pkl   # Model bundle
│
├── backend/
│   ├── __init__.py
│   ├── model_loader.py             # Bundle + SPECTER2 loading with st.cache_resource
│   └── predictor.py                # Full inference pipeline with mean imputation
│
└── frontend/
    ├── __init__.py
    ├── styles.py                   # Light theme CSS
    ├── components.py               # Prediction card, metrics, explanation
    └── tabs.py                     # Predict, Compare, Model Insights, Examples tabs
```

---

## Setup & Running 🛠️

### Prerequisites
```bash
pip install -r requirements.txt
```

### Model Bundle
Place the following in the same directory as `app.py`:
- `viral_predictor_v12_altmetric_no_hindex.pkl`
- SPECTER2 model files at the paths stored in the bundle (`specter_finetuned_path` or `specter_local_path`)

### Run
```bash
streamlit run app.py
```

---

## Dependencies
```
streamlit>=1.32.0
torch>=2.0.0
transformers>=4.38.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
```

---

## Known Limitations ⚠️

- **Field scope**: Trained on Computer Science and Medicine only. Predictions for other fields are unreliable.
- **Venue/reference bias**: Structural metadata features (venue prestige, reference count) are strong predictors but may disadvantage early-career researchers. Mean imputation is used when these are unknown.
- **Temporal degradation**: Performance degrades on 2025 papers (ROC 0.889 vs 0.920 in 2023) due to shorter attention accumulation windows. 2026 papers are excluded from primary metrics for the same reason.
- **Irreducible noise**: Papers that go viral due to news cycle timing or social amplification by specific influencers cannot be predicted from content alone. The false negative fingerprint shows these are often single-PI, non-top-venue papers.

---

## Fairness Design Decisions 🤝

| Decision | Rationale |
|---|---|
| H-index removed | Penalises early-career researchers; career stage ≠ paper quality |
| Per-cohort labels | CS and Medicine have structurally different attention patterns |
| Mean imputation for unknowns | Setting missing fields to 0 falsely signals low quality |
| Optional metadata fields | Researchers who don't know their venue yet are not penalised |

---

## FYP Project · 2025
*Computer Science + Medicine · 94,435 papers · SPECTER2 fine-tuned · XGBoost ROC-AUC 0.9076 · Ensemble ROC-AUC 0.9073 (primary test 2023–2025)*