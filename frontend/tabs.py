"""
Tab renderers
"""
import streamlit as st
from .components import (
    render_prediction_card,
    render_model_metrics,
    render_feature_breakdown,
    render_explanation,
)


# Helpers
_FIELDS    = ["Computer Science", "Medicine", "Unknown"]
_PUB_TYPES = ["Journal", "Conference", "Preprint", "Unknown"]

_EXAMPLES = {
    "🏥 Clinical Trial (Medicine - viral 🔥)": {
        "title":           "Tirzepatide vs Semaglutide for Weight Reduction in Adults with Obesity: A Randomised Controlled Trial",
        "abstract":        "Importance: Although tirzepatide and semaglutide were shown to reduce weight in randomized clinical trials, data from head-to-head comparisons in populations with overweight or obesity are not yet available. Objective: To compare on-treatment weight loss and rates of gastrointestinal adverse events (AEs) among adults with overweight or obesity receiving tirzepatide or semaglutide labeled for type 2 diabetes (T2D) in a clinical setting. Design, setting, and participants: In this cohort study, adults with overweight or obesity receiving semaglutide or tirzepatide between May 2022 and September 2023 were identified using electronic health record (EHR) data linked to dispensing information from a collective of US health care systems. On-treatment weight outcomes through November 3, 2023, were assessed. Adults with overweight or obesity and regular care in the year before initiation, no prior glucagon-like peptide 1 receptor agonist receptor agonist use, a prescription within 60 days prior to initiation, and an available baseline weight were identified. The analysis was completed on April 3, 2024. Exposures: Tirzepatide or semaglutide in formulations labeled for T2D, on or off label. Main outcomes and measures: On-treatment weight change in a propensity score-matched population, assessed as hazard of achieving 5% or greater, 10% or greater, and 15% or greater weight loss, and percentage change in weight at 3, 6, and 12 months. Hazards of gastrointestinal AEs were compared. Results: Among 41 222 adults meeting the study criteria (semaglutide, 32 029; tirzepatide, 9193), 18 386 remained after propensity score matching. The mean (SD) age was 52.0 (12.9) years, 12 970 were female (70.5%), 14 182 were white (77.1%), 2171 Black (11.8%), 354 Asian (1.9%), 1679 were of other or unknown race, and 9563 (52.0%) had T2D. The mean (SD) baseline weight was 110 (25.8) kg. Follow-up was ended by discontinuation for 5140 patients (55.9%) receiving tirzepatide and 4823 (52.5%) receiving semaglutide. Patients receiving tirzepatide were significantly more likely to achieve weight loss (≥5%; hazard ratio [HR], 1.76, 95% CI, 1.68, 1.84; ≥10%; HR, 2.54; 95% CI, 2.37, 2.73; and ≥15%; HR, 3.24; 95% CI, 2.91, 3.61). On-treatment changes in weight were larger for patients receiving tirzepatide at 3 months (difference, -2.4%; 95% CI -2.5% to -2.2%), 6 months (difference, -4.3%; 95% CI, -4.7% to -4.0%), and 12 months (difference, -6.9%; 95% CI, -7.9% to -5.8%). Rates of gastrointestinal AEs were similar between groups. Conclusions and relevance: In this population of adults with overweight or obesity, use of tirzepatide was associated with significantly greater weight loss than semaglutide. Future study is needed to understand differences in other important outcomes.",
        "authors":         "Smith J, Patel R, Johnson M, Chen L, Williams A, Brown K, Davis E, Wilson T",
        "field":           "Medicine",
        "pub_type":        "Journal",
        "venue":           "New England Journal of Medicine",
        "reference_count": 58,
        "pub_year":        2024,
        "altmetric_img":   "assets/altmetric_1.png",
    },
    "🧬 Medicine - Not viral": {
        "title":           "In Vitro Enzymatic Studies Reveal pH and Temperature Sensitive Properties of the CLIC Proteins",
        "abstract":        "Chloride intracellular ion channel (CLIC) proteins exist as both soluble and integral membrane proteins, with CLIC1 capable of shifting between two distinct structural conformations. New evidence has emerged indicating that members of the CLIC family act as moonlighting proteins, referring to the ability of a single protein to carry out multiple functions. In addition to their ion channel activity, CLIC family members possess oxidoreductase enzymatic activity and share significant structural and sequence homology, along with varying overlaps in their tissue distribution and cellular localization. In this study, the 2-hydroxyethyl disulfide (HEDS) assay system was used to characterize kinetic properties, as well as the temperature and pH profiles of three CLIC protein family members (CLIC1, CLIC3, CLIC4). We also assessed the effects of the drugs rapamycin and amphotericin B, on the three CLIC proteins' enzymatic activity in the HEDS assay. Our results demonstrate CLIC1 to be highly heat-sensitive, with optimal enzymatic activity observed at neutral pH7 and at a temperature of 37 °C, while CLIC3 had higher oxidoreductase activity in more acidic pH5 and was found to be relatively heat stable. CLIC4, like CLIC1, was temperature sensitive with optimal enzymatic activity observed at 37 °C; however, it showed optimal activity in more alkaline conditions of pH8. Our current study demonstrates individual differences in the enzymatic activity between the three CLIC proteins, suggesting each CLIC protein is likely regulated in discrete ways, involving changes in the subcellular milieu and microenvironment.",
        "authors":         "Amani Alghalayini, Khondker Rufaka Hossain ,Saba Moghaddasi ,Daniel R. Turkewitz ,Claudia D’Amario ,Michael Wallach, Stella M. Valenzuela",
        "field":           "Medicine",
        "pub_type":        "Article",
        "venue":           "Biomolecules",
        "reference_count": 6,
        "pub_year":        2023,
        "altmetric_img":   "assets/altmetric_2.png",
    },
    "🤖 LLM / AI Paper (CS - viral 🔥)": {
        "title":           "Large language models encode clinical knowledge",
        "abstract":        "Large language models (LLMs) have demonstrated impressive capabilities, but the bar for clinical applications is high. Attempts to assess the clinical knowledge of models typically rely on automated evaluations based on limited benchmarks. Here, to address these limitations, we present MultiMedQA, a benchmark combining six existing medical question answering datasets spanning professional medicine, research and consumer queries and a new dataset of medical questions searched online, HealthSearchQA. We propose a human evaluation framework for model answers along multiple axes including factuality, comprehension, reasoning, possible harm and bias. In addition, we evaluate Pathways Language Model1 (PaLM, a 540-billion parameter LLM) and its instruction-tuned variant, Flan-PaLM2 on MultiMedQA. Using a combination of prompting strategies, Flan-PaLM achieves state-of-the-art accuracy on every MultiMedQA multiple-choice dataset (MedQA3, MedMCQA4, PubMedQA5 and Measuring Massive Multitask Language Understanding (MMLU) clinical topics6), including 67.6% accuracy on MedQA (US Medical Licensing Exam-style questions), surpassing the prior state of the art by more than 17%. However, human evaluation reveals key gaps. To resolve this, we introduce instruction prompt tuning, a parameter-efficient approach for aligning LLMs to new domains using a few exemplars. The resulting model, Med-PaLM, performs encouragingly, but remains inferior to clinicians. We show that comprehension, knowledge recall and reasoning improve with model scale and instruction prompt tuning, suggesting the potential utility of LLMs in medicine. Our human evaluations reveal limitations of today’s models, reinforcing the importance of both evaluation frameworks and method development in creating safe, helpful LLMs for clinical applications.",
        "authors":         "Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne, Paul Gamble, Chris Kelly, Abubakr Babiker, Nathanael Schärli, Aakanksha Chowdhery, Philip Mansfield, Dina Demner-Fushman, Blaise Agüera y Arcas, Dale Webster, Greg S. Corrado, Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle Barral, Christopher Semturs, Alan Karthikesalingam & Vivek Natarajan",
        "field":           "Computer Science",
        "pub_type":        "article",
        "venue":           "nature",
        "reference_count": 94,
        "pub_year":        2023,
        "altmetric_img":   "assets/altmetric_3.png",
    },
    "🔢 Mathematics (CS - Not viral)": {
        "title":           "A Practical Analysis of Oblivious Sorting Algorithms for Secure Multi-party Computation",
        "abstract":        "Cryptographic secure computing methods like secure multi-party computation, circuit garbling and homomorphic encryption are becoming practical enough to be usable in applications. Such applications need special data-independent sorting algorithms to preserve privacy. In this paper, we describe the design and implementation of four different oblivious sorting algorithms. We improve two earlier designs based on sorting networks and quicksort with the capability of sorting matrices. We also propose two new designs—a naive comparison-based sort with a low round count and an oblivious radix sort algorithm that does not require any private comparisons. For all these algorithms, we present thorough complexity and performance analysis including detailed breakdown of running-time, network and memory usage.",        "authors":         "Muhammad Shafiq, Zhaoquan Gu,",
        "field":           "Computer Science",
        "pub_type":        "Conference",
        "venue":           "Springer",
        "reference_count": 26,
        "pub_year":        2014,
        "altmetric_img":   "assets/altmetric_4.png",
    },
}


# Tab 1: Single prediction
def render_predict_tab(predict_fn, settings: dict):
    st.header("Enter Paper Details")

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        title = st.text_input(
            "Title *",
            placeholder="e.g. Tirzepatide for Obesity: a Randomised Controlled Trial",
        )
        abstract = st.text_area(
            "Abstract *",
            height=210,
            placeholder="Paste full abstract here (≥50 words recommended)…",
        )

    with col_r:
        field    = st.selectbox("Research Field *", _FIELDS)
        pub_type = st.selectbox("Publication Type *", _PUB_TYPES)
        venue    = st.text_input(
            "Venue",
            placeholder="e.g. New England Journal of Medicine",
        )
        ref_count = st.number_input("Reference Count", min_value=0, max_value=1000, value=30)
        n_authors = st.number_input("Number of Authors", min_value=1, max_value=500, value=4)
        pub_year  = st.number_input("Publication Year", min_value=2000, max_value=2030, value=2024)

    run = st.button("🚀 Predict Virality", type="primary", width='stretch')

    if run:
        if not title.strip() or not abstract.strip():
            st.error("❌ Title and abstract are required.")
            return
        if len(abstract.split()) < 30:
            st.warning("⚠️ Abstract is very short - predictions are more reliable with 50+ words.")

        with st.spinner("Embedding paper with SPECTER2…"):
            result = predict_fn({
                "title":           title,
                "abstract":        abstract,
                "authors":         [f"A{i}" for i in range(n_authors)],
                "field":           field,
                "pub_type":        pub_type,
                "venue":           venue,
                "reference_count": float(ref_count),
                "pub_year":        int(pub_year),
            })

        st.markdown('<p class="section-head">Result</p>', unsafe_allow_html=True)
        render_prediction_card(result)

        st.markdown('<p class="section-head">Model Agreement</p>', unsafe_allow_html=True)
        render_model_metrics(result)

        st.markdown('<p class="section-head">Feature Snapshot</p>', unsafe_allow_html=True)
        render_feature_breakdown(result)

        render_explanation(result)

        st.caption(
            "⚠️ This model was trained on CS + Medicine papers (2018-2026). "
            "Predictions outside these domains are less reliable. "
            "Use as a research tool, not a submission decision."
        )


# Tab 2: Compare two papers
def render_compare_tab(predict_fn, settings: dict):
    st.header("Compare Two Papers")
    st.caption("Enter both papers below to see which has higher viral potential.")

    colA, colB = st.columns(2, gap="large")

    def paper_form(col, label):
        with col:
            st.markdown(f"**{label}**")
            title    = st.text_input("Title *",    key=f"title_{label}")
            abstract = st.text_area("Abstract *", key=f"abstract_{label}", height=160)
            field    = st.selectbox("Field",    _FIELDS,    key=f"field_{label}")
            pub_type = st.selectbox("Type",     _PUB_TYPES, key=f"type_{label}")
            venue    = st.text_input("Venue",   key=f"venue_{label}")
            refs     = st.number_input("Refs",  min_value=0, max_value=1000, value=30, key=f"refs_{label}")
            authors  = st.number_input("Authors", min_value=1, max_value=500, value=4, key=f"auth_{label}")
            year     = st.number_input("Year",  min_value=2000, max_value=2030, value=2024, key=f"year_{label}")
        return {
            "title": title, "abstract": abstract, "field": field,
            "pub_type": pub_type, "venue": venue,
            "reference_count": float(refs),
            "authors": [f"A{i}" for i in range(authors)],
            "pub_year": int(year),
        }

    data_a = paper_form(colA, "📄 Paper A")
    data_b = paper_form(colB, "📄 Paper B")

    if st.button("⚖️ Compare Papers", type="primary", width='stretch'):
        if not all([data_a["title"], data_a["abstract"], data_b["title"], data_b["abstract"]]):
            st.error("❌ Fill in title and abstract for both papers.")
            return

        with st.spinner("Analysing both papers…"):
            res_a = predict_fn(data_a)
            res_b = predict_fn(data_b)

        st.markdown('<p class="section-head">Comparison</p>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.markdown("**Paper A**")
            render_prediction_card(res_a)
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            pa, pb = res_a["viral_probability"], res_b["viral_probability"]
            ratio  = pa / max(pb, 1e-6)
            if abs(ratio - 1) < 0.1:
                st.info("~Equal")
            elif ratio > 1:
                st.success(f"A wins\n{ratio:.1f}×")
            else:
                st.success(f"B wins\n{1/ratio:.1f}×")
        with c3:
            st.markdown("**Paper B**")
            render_prediction_card(res_b)


# Tab 3: Model insights
def render_insights_tab(bundle: dict):
    st.markdown('<p class="section-head">Model Performance</p>', unsafe_allow_html=True)

    if not bundle:
        st.warning("Bundle not loaded.")
        return

    res   = bundle.get("results_summary", {})
    order = ["Baseline (meta only)", "Logistic Regression", "Random Forest", "XGBoost", "Ensemble"]

    rows = []
    for name in order:
        if name not in res:
            continue
        r = res[name]
        rows.append({
            "Model":       name,
            "ROC (all)":   f"{r.get('roc',      float('nan')):.4f}",
            "PR (all)":    f"{r.get('pr',       float('nan')):.4f}",
            "ROC (mature)":f"{r.get('roc_mature',float('nan')):.4f}",
            "PR (mature)": f"{r.get('pr_mature', float('nan')):.4f}",
            "F1":          f"{r.get('f1',        float('nan')):.4f}",
        })

    import pandas as pd
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.caption(
        "2026 papers (9,758) excluded - labels unreliable at <3 months exposure. "
        "Mature = papers ≥12 months old at Altmetric snapshot."
    )

    # Ablation
    abl = bundle.get("ablation_results", {})
    if abl:
        st.markdown('<p class="section-head">Ablation Study</p>', unsafe_allow_html=True)
        abl_rows = [
            {"Configuration": k.replace("\n", " "),
             "ROC-AUC": f"{v['roc']:.4f}",
             "PR-AUC":  f"{v['pr']:.4f}"}
            for k, v in abl.items()
        ]
        st.dataframe(pd.DataFrame(abl_rows), width='stretch', hide_index=True)

    # Fine-tuning gain
    ft = bundle.get("ft_ablation_results", {})
    if ft:
        st.markdown('<p class="section-head">Fine-Tuning Gain</p>', unsafe_allow_html=True)
        ft_rows = [
            {"Configuration": k.replace("\n", " "),
             "ROC-AUC": f"{v['roc']:.4f}",
             "PR-AUC":  f"{v['pr']:.4f}"}
            for k, v in ft.items()
        ]
        st.dataframe(pd.DataFrame(ft_rows), width='stretch', hide_index=True)

    # Stratified eval
    strat = bundle.get("stratified_eval", [])
    if strat:
        st.markdown('<p class="section-head">Year-Stratified Evaluation (XGBoost)</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(strat), width='stretch', hide_index=True)

    # Topic virality
    topics = bundle.get("topic_virality", [])
    if topics:
        st.markdown('<p class="section-head">Topic-Level Virality (K-Means clusters)</p>', unsafe_allow_html=True)
        topic_df = pd.DataFrame(topics)[["label", "n_test", "viral_rate", "roc", "pr", "mean_refs"]]
        topic_df["viral_rate"] = topic_df["viral_rate"].map(lambda x: f"{x:.1%}")
        topic_df["roc"]        = topic_df["roc"].map(lambda x: f"{x:.4f}")
        topic_df["pr"]         = topic_df["pr"].map(lambda x: f"{x:.4f}")
        st.dataframe(topic_df, width='stretch', hide_index=True)


# Tab 4: Examples
def render_examples_tab(predict_fn, settings: dict):
    st.markdown('<p class="section-head">Pre-Loaded Examples</p>', unsafe_allow_html=True)
    st.caption("Select an example paper and run the predictor instantly.")

    choice = st.selectbox("Choose an example", list(_EXAMPLES.keys()))
    ex = _EXAMPLES[choice]

    with st.expander("📋 Paper details", expanded=True):
        st.markdown(f"**Title:** {ex['title']}")
        st.markdown(f"**Field:** {ex['field']} · **Type:** {ex['pub_type']} · **Venue:** {ex['venue']}")
        st.markdown(f"**Authors:** {ex['authors']} · **Refs:** {ex['reference_count']} · **Year:** {ex['pub_year']}")
        st.markdown(f"**Abstract:** {ex['abstract'][:300]}…")
        
        alt_img = ex.get("altmetric_img")

    if alt_img: st.markdown("**Altmetric Attention (Proof):**")

    try:
        st.image(alt_img, width=300)
    except:
        st.warning("Altmetric screenshot not found.")

    st.caption("Altmetric score evidence - used to validate viral / non-viral classification.")

    if st.button("🔬 Run Prediction on this Example", type="primary", width='stretch'):
        paper = {
            "title":           ex["title"],
            "abstract":        ex["abstract"],
            "authors":         ex["authors"].split(","),
            "field":           ex["field"],
            "pub_type":        ex["pub_type"],
            "venue":           ex["venue"],
            "reference_count": float(ex["reference_count"]),
            "pub_year":        int(ex["pub_year"]),
        }
        with st.spinner("Running prediction…"):
            result = predict_fn(paper)

        st.markdown("---")
        st.header("📊 Prediction Results")
        render_prediction_card(result)
        render_model_metrics(result)
        render_feature_breakdown(result)
        render_explanation(result)