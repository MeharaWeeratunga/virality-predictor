"""
Reusable UI components
"""
import streamlit as st


def render_prediction_card(result: dict):
    """Large prediction result card."""
    is_viral = result["prediction"] == "VIRAL"
    css_cls  = "pred-viral" if is_viral else "pred-safe"
    icon     = "🔥" if is_viral else "📄"
    label    = "VIRAL" if is_viral else "NOT VIRAL"
    sub_txt  = (
        "High viral potential detected - pre-publication signals are strong."
        if is_viral else
        "Limited viral reach predicted based on pre-publication signals."
    )
    prob = result["viral_probability"]
    thresh = result["threshold"]
    model  = result["model_used"]

    st.markdown(f"""
    <div class="{css_cls}">
        <div class="pred-icon">{icon}</div>
        <div>
            <p class="pred-label">{label}</p>
            <p class="pred-sub">{sub_txt}</p>
            <p class="pred-sub" style="margin-top:0.4rem;">
                Probability <strong style="color:var(--text)">{prob:.1%}</strong>
                &nbsp;·&nbsp; threshold {thresh:.3f}
                &nbsp;·&nbsp; {model}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_model_metrics(result: dict):
    """4-up metric grid for individual model probabilities."""
    ip = result["individual_probas"]
    ens_key = [k for k in ip if "Ensemble" in k][0]
    items = [
        ("Ensemble",     ip[ens_key]),
        ("XGBoost",      ip["XGBoost"]),
        ("Log. Reg.",    ip["Logistic Regression"]),
        ("Rand. Forest", ip["Random Forest"]),
    ]
    cells = "".join(
        f'<div class="metric-box">'
        f'<div class="metric-val">{v:.0%}</div>'
        f'<div class="metric-name">{n}</div>'
        f'</div>'
        for n, v in items
    )
    st.markdown(
        f'<div class="metric-row">{cells}</div>',
        unsafe_allow_html=True
    )

    # Probability bars per model
    st.markdown('<p style="font-size:0.78rem;color:var(--muted);margin-bottom:0.7rem;">INDIVIDUAL MODEL PROBABILITIES</p>', unsafe_allow_html=True)
    for name, val in items:
        if val >= 0.65:
            fill_cls = "prob-fill-viral"
        elif val >= 0.35:
            fill_cls = "prob-fill-mid"
        else:
            fill_cls = "prob-fill-safe"
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">{name}</div>
            <div class="prob-track">
                <div class="{fill_cls}" style="width:{val*100:.1f}%"></div>
            </div>
            <div class="prob-val">{val:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_feature_breakdown(result: dict):
    """Chip grid of key features used in the prediction."""
    kf = result["key_features"]

    def chip(name, val, highlight=False):
        cls = 'chip-val highlight' if highlight else 'chip-val'
        return (
            f'<div class="chip">'
            f'<div class="chip-name">{name}</div>'
            f'<div class="{cls}">{val}</div>'
            f'</div>'
        )

    chips = "".join([
        chip("Authors",          kf["num_authors"]),
        chip("References",       int(kf["reference_count"])),
        chip("Top Venue",        "✓" if kf["is_top_venue"] else "✗",
             highlight=kf["is_top_venue"]),
        chip("Pub Year",         kf["pub_year"]),
        chip("Title Words",      kf["title_len"]),
        chip("Abstract Words",   kf["abstract_len"]),
        chip("Impact Words (T)", kf["title_impact_words"],
             highlight=kf["title_impact_words"] > 2),
        chip("Impact Words (A)", kf["abstract_impact_words"],
             highlight=kf["abstract_impact_words"] > 5),
        chip("Readability",      f"{kf['abstract_readability']:.1f} avg chars/word"),
        chip("KNN Viral Rate",   f"{kf['knn_viral_rate']:.1%}"),
    ])
    st.markdown(
        f'<div class="chip-grid">{chips}</div>',
        unsafe_allow_html=True
    )


def render_explanation(result: dict):
    """Rule-based textual explanation of the prediction."""
    kf   = result["key_features"]
    prob = result["viral_probability"]
    good, bad, tips = [], [], []

    if kf["is_top_venue"]:
        good.append("Published in a top-tier venue (Nature, NEJM, NeurIPS…)")
    else:
        bad.append("Venue not in top-tier whitelist - lower baseline attention")
        tips.append("Target higher-impact journals or conferences if possible")

    if kf["num_authors"] >= 10:
        good.append(f"Large author team ({kf['num_authors']} authors) - broader network reach")
    elif kf["num_authors"] <= 2:
        bad.append("Small author count - narrower promotion network")

    if kf["reference_count"] >= 50:
        good.append(f"Well-referenced paper ({int(kf['reference_count'])} refs) - signals scholarly depth")
    elif kf["reference_count"] < 15:
        bad.append("Low reference count - may signal narrow scope")
        tips.append("Ensure comprehensive literature review")

    if kf["title_impact_words"] >= 2:
        good.append(f"Title contains {kf['title_impact_words']} high-attention keywords (e.g. 'trial', 'global', 'mortality')")
    else:
        bad.append("Title lacks high-attention keywords")
        tips.append("Consider adding concrete impact terms: 'global', 'randomized', 'mortality' etc.")

    if kf["abstract_impact_words"] >= 8:
        good.append(f"Abstract is keyword-rich ({kf['abstract_impact_words']} impact terms)")
    elif kf["abstract_impact_words"] <= 3:
        bad.append("Abstract has few high-attention impact words")

    if kf["title_has_colon"]:
        good.append("Colon in title - common in high-impact structured titles")

    if kf["abstract_len"] >= 200:
        good.append(f"Detailed abstract ({kf['abstract_len']} words) - richer signal for the model")
    elif kf["abstract_len"] < 80:
        bad.append("Short abstract - model has limited text signal")
        tips.append("Expand abstract to 150+ words for better prediction accuracy")

    if prob >= 0.65:
        good.append(f"Strong ensemble agreement - all models above threshold")
    elif prob < 0.35:
        bad.append("Weak ensemble signal - models disagree or collectively low")

    if kf.get("refs_imputed"):
        tips.append("Reference count was not provided and was imputed - supply it for a sharper prediction")
    if kf.get("authors_imputed"):
        tips.append("Author count was not provided and was imputed - supply it for a sharper prediction")

    st.markdown("")

    with st.expander("🔍 Explanation", expanded=True):
        if good:
            st.markdown("**✅ What helped**")
            for g in good:
                st.markdown(f"- {g}")
        if bad:
            st.markdown("**🔴 What held it back**")
            for b in bad:
                st.markdown(f"- {b}")
        if tips:
            st.markdown("**💡 How to improve**")
            for t in tips:
                st.markdown(f"- {t}")

        st.caption(
            "Explanation is rule-based (SHAP top features: pub_type, "
            "num_authors, reference_count, is_top_venue, impact words). "
            "It is indicative, not exhaustive."
        )


def render_sidebar(bundle: dict) -> dict:
    """Render sidebar and return user settings."""
    st.markdown("### ⚙️ Settings")

    use_ensemble = st.toggle("Use Ensemble model", value=True,
        help="Ensemble (XGB×0.4 + LR×0.4 + RF×0.2)")

    st.markdown("---")
    st.markdown("### 📊 Model Info")

    if bundle:
        res = bundle.get("results_summary", {})
        ens = res.get("Ensemble", {})
        roc = ens.get("roc_mature", ens.get("roc", None))
        pr  = ens.get("pr_mature",  ens.get("pr",  None))

        col1, col2 = st.columns(2)
        col1.metric("ROC-AUC",  f"{roc:.4f}" if roc else "-", help="Mature papers")
        col2.metric("PR-AUC",   f"{pr:.4f}"  if pr  else "-", help="Mature papers")

        st.caption(
            f"Fields: Computer Science + Medicine  \n"
            f"Train ≤ {bundle.get('train_year_cutoff', 2022)} | "
            f"Snapshot: {bundle.get('eval_cutoff', '-')}  \n"
            f"Labels: Altmetric top-10% per year cohort  \n"
            f"Embeddings: SPECTER2 fine-tuned"
        )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This tool predicts whether a research paper will go viral "
        "using only signals available **before publication**: title, "
        "abstract, venue, reference count, and author count. "
        "No post-publication data is used."
    )

    return {"use_ensemble": use_ensemble}