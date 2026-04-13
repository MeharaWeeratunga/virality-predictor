"""
Prediction logic for model bundle.

NUMERIC_META (16 features — must match pipeline exactly):
  num_authors, reference_count, reference_count_log,
  pub_year_centred, top_venue_x_refcount,
  title_len, title_has_question, title_has_colon, title_has_number,
  abstract_len, abstract_has_eq, abstract_has_table,
  is_top_venue,
  title_impact_words, abstract_impact_words, abstract_readability
"""
import re
import numpy as np
import torch

# Impact word vocabulary (mirrors pipeline)
_IMPACT_WORDS = {
    # Trial / study design
    'randomized', 'randomised', 'trial', 'trials', 'placebo',
    'controlled', 'blinded', 'blind', 'multicenter', 'multicentre',
    'multinational', 'clinical', 'phase',
    # Outcomes
    'mortality', 'survival', 'death', 'risk', 'outcome', 'outcomes',
    'efficacy', 'safety', 'benefit', 'reduction', 'prevention',
    'cardiovascular', 'cancer', 'diabetes', 'obesity', 'stroke',
    # Scale / impact
    'global', 'worldwide', 'national', 'largest', 'first', 'novel',
    'major', 'significant', 'critical', 'important', 'breakthrough',
    'multi', 'large', 'human', 'world',
    # Public health
    'pandemic', 'covid', 'epidemic', 'burden', 'crisis', 'emergency',
    'antimicrobial', 'resistance', 'infection',
    # Treatment
    'treatment', 'therapy', 'drug', 'vaccine', 'intervention',
    'semaglutide', 'tirzepatide',
    # CS/AI
    'large', 'language', 'intelligence', 'benchmark', 'foundation',
}

# Field-stratified kNN viral rate fallbacks
_KNN_VR_BY_FIELD = {
    "Medicine":         0.18,
    "Computer Science": 0.03,
    "Unknown":          0.10,   # global train mean
}

def _n_authors(authors_raw) -> int:
    if isinstance(authors_raw, list):
        return len(authors_raw)
    return len([x for x in str(authors_raw).split(',') if x.strip()])


def _count_impact(text: str) -> int:
    words = re.findall(r'[a-z]+', str(text).lower())
    return sum(1 for w in words if w in _IMPACT_WORDS)


def _avg_word_len(text: str) -> float:
    words = re.findall(r'[a-z]+', str(text).lower())
    return float(np.mean([len(w) for w in words])) if words else 5.0


def _is_top_venue(venue_str: str, tw_norm_set) -> int:
    v = re.sub(r'[^a-z0-9 ]', ' ', str(venue_str).lower())
    return int(any(tw in v for tw in tw_norm_set))


def _embed(title: str, abstract: str, tok, mdl, device: str) -> np.ndarray:
    text = f"{title.strip()} {abstract.strip()}" or "untitled"
    inputs = tok(
        text, padding=True, truncation=True,
        max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = mdl(**inputs)
    return out.pooler_output.squeeze().cpu().numpy()   # (768,)


def predict_virality(
    paper: dict,
    bundle: dict,
    tok,
    mdl,
    device: str,
    use_ensemble: bool = True,
) -> dict:
    """
    Predict virality for a single paper.

    paper keys (all optional except title + abstract):
        title, abstract, authors, field, pub_type, venue,
        reference_count, pub_year
    """
    if bundle is None or tok is None or mdl is None:
        raise RuntimeError("Model or SPECTER2 not loaded.")

    title_s  = str(paper.get("title",    "")).strip()
    abstr_s  = str(paper.get("abstract", "")).strip()

    # SPECTER2 embedding
    emb = _embed(title_s, abstr_s, tok, mdl, device)   # (768,)

    # Training set means - used for missing value handling
    _means = bundle.get("feature_means", {
        "reference_count": 57.0,
        "num_authors":      7.6,
    })

    # Reference count means by publication type - used only when value is None
    _ref_means_by_type = {
        "Journal":    65.0,
        "Conference": 35.0,
        "Preprint":   45.0,
        "Unknown":    57.0,   # overall mean
    }
    _mean_refs = float(_ref_means_by_type.get(
        paper.get("pub_type", "Unknown"), 57.0
    ))

    # Venue normalisation 
    tw_norm = frozenset(
        re.sub(r'[^a-z0-9 ]', ' ', tw.lower())
        for tw in bundle.get("top_venue_whitelist", [])
    )
    venue_s     = str(paper.get("venue", paper.get("publicationVenue", "")))
    venue_known = venue_s.strip() != ""
    is_tv       = _is_top_venue(venue_s, tw_norm) if venue_known else None

    # Scalar features
    # reference_count: impute to pub-type mean ONLY if not provided (None)
    _ref_raw  = paper.get("reference_count")
    ref_count = float(_ref_raw) if _ref_raw is not None else _mean_refs
    ref_log   = float(np.log1p(ref_count))

    # num_authors: impute only if None
    _auth_raw = paper.get("authors")
    if _auth_raw is None:
        n_auth = round(float(_means.get("num_authors", 7.6)))
    else:
        n_auth = _n_authors(_auth_raw)

    train_cutoff = int(bundle.get("train_year_cutoff", 2022))
    pub_year     = int(paper.get("pub_year", 2024))
    pub_year_ctr = float(pub_year - train_cutoff)

    # is_top_venue: if venue unknown use neutral signal
    if is_tv is None:
        is_tv     = 0
        top_x_ref = 0.11 * ref_log   # mean venue rate as proxy
    else:
        top_x_ref = float(is_tv) * ref_log

    # Linguistic features
    title_impact   = _count_impact(title_s)
    abstr_impact   = _count_impact(abstr_s)
    readability    = _avg_word_len(abstr_s)

    # kNN viral rate
    field    = str(paper.get("field",    "Unknown"))
    knn_vr = float(_KNN_VR_BY_FIELD.get(field, 0.10))


    # Build feature lookup - all possible features
    _feature_lookup = {
        "num_authors":           n_auth,
        "reference_count":       ref_count,
        "reference_count_log":   ref_log,
        "pub_year_centred":      pub_year_ctr,
        "top_venue_x_refcount":  top_x_ref,
        "title_len":             len(title_s.split()),
        "title_has_question":    int("?" in title_s),
        "title_has_colon":       int(":" in title_s),
        "title_has_number":      int(bool(re.search(r'\d', title_s))),
        "abstract_len":          len(abstr_s.split()),
        "abstract_has_eq":       int("$" in abstr_s or "equation" in abstr_s.lower()),
        "abstract_has_table":    int("table" in abstr_s.lower() or "fig" in abstr_s.lower()),
        "is_top_venue":          is_tv,
        "title_impact_words":    title_impact,
        "abstract_impact_words": abstr_impact,
        "abstract_readability":  readability,
        "knn_viral_rate":        knn_vr,
    }

    # Use the feature order saved in the bundle
    meta_numeric = bundle.get("meta_numeric", list(_feature_lookup.keys()))
    num_vals = [_feature_lookup.get(f, 0.0) for f in meta_numeric]

    # Validate feature count
    scaler   = bundle["scaler"]
    expected = scaler.n_features_in_
    if len(num_vals) != expected:
        raise ValueError(
            f"Feature count mismatch: built {len(num_vals)} from bundle's "
            f"meta_numeric={meta_numeric}, scaler expects {expected}. "
            f"Ensure the bundle matches the predictor version."
        )

    num_sc  = scaler.transform(np.array(num_vals).reshape(1, -1))

    # OHE
    ohe     = bundle["ohe"]
    pub_type = paper.get("pub_type", "Unknown")
    field    = paper.get("field",    "Unknown")
    cat_arr  = ohe.transform([[pub_type, field]])

    meta_arr = np.hstack([num_sc, cat_arr])              # (1, n_meta)
    X_tree   = np.hstack([emb.reshape(1, -1), meta_arr]) # (1, 768+n_meta)

    # Individual model probabilities
    p_xgb = bundle["xgb_model"].predict_proba(X_tree)[0][1]
    p_rf  = bundle["rf_model"].predict_proba(X_tree)[0][1]

    pca = bundle.get("pca")
    emb_pca = pca.transform(emb.reshape(1, -1)) if pca else emb.reshape(1, -1)
    X_lr    = np.hstack([emb_pca, meta_arr])
    p_lr    = bundle["lr_model"].predict_proba(X_lr)[0][1]

    # Training weights
    ew = bundle.get(
        "ensemble_weights",
        {"XGBoost": 0.4, "Logistic Regression": 0.4, "Random Forest": 0.2}
    )
    p_ens = (
        ew["XGBoost"]             * p_xgb +
        ew["Logistic Regression"] * p_lr  +
        ew["Random Forest"]       * p_rf
    )

    thresholds = bundle.get("thresholds", {})
    if use_ensemble and "Ensemble" in thresholds:
        proba_use  = p_ens
        thresh_use = thresholds["Ensemble"]
        model_used = "Ensemble"
    else:
        proba_use  = p_xgb
        thresh_use = thresholds.get("XGBoost", 0.5)
        model_used = "XGBoost"

    prediction = "VIRAL" if proba_use >= thresh_use else "NOT VIRAL"

    return {
        "prediction":        prediction,
        "viral_probability": round(float(proba_use), 4),
        "threshold":         round(float(thresh_use), 4),
        "model_used":        model_used,
        "individual_probas": {
            "XGBoost":             round(float(p_xgb), 4),
            "Logistic Regression": round(float(p_lr),  4),
            "Random Forest":       round(float(p_rf),  4),
            "Ensemble ((XGB×0.4+LR×0.4+RF×0.2))": round(float(p_ens), 4),
        },
        "key_features": {
            "num_authors":           n_auth,
            "reference_count":       ref_count,
            "is_top_venue":          bool(is_tv),
            "pub_year":              pub_year,
            "title_len":             len(title_s.split()),
            "abstract_len":          len(abstr_s.split()),
            "title_has_colon":       int(":" in title_s),
            "title_impact_words":    title_impact,
            "abstract_impact_words": abstr_impact,
            "abstract_readability":  round(readability, 2),
            "knn_viral_rate":        round(knn_vr, 3),
            "refs_imputed":          (paper.get("reference_count") is None or paper.get("reference_count") == 0),
            "authors_imputed":       (paper.get("authors") is None or _n_authors(paper.get("authors", [])) == 0),
        },
    }