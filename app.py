"""
Pre-Publication Virality Predictor
Streamlit app · CS + Medicine · Altmetric labels · SPECTER2 fine-tuned
"""
import streamlit as st

# Page config
st.set_page_config(
    page_title="Virality Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend
from backend.model_loader import load_bundle, load_specter
from backend.predictor    import predict_virality

# Frontend
from frontend.styles     import get_css
from frontend.components import render_sidebar
from frontend.tabs       import (
    render_predict_tab,
    render_compare_tab,
    render_insights_tab,
    render_examples_tab,
)

# Inject CSS
st.markdown(get_css(), unsafe_allow_html=True)

# Load models
BUNDLE_PATH = "viral_predictor_v12_altmetric_no_hindex.pkl"

bundle = load_bundle(BUNDLE_PATH)
if bundle is None:
    st.error(
        "⚠️ Could not load model bundle. "
        f"Ensure `{BUNDLE_PATH}` is in the same directory as `app.py`."
    )
    st.stop()

tok, mdl, device = load_specter(bundle)
if tok is None:
    st.error("⚠️ SPECTER2 failed to load. Check Drive paths in bundle.")
    st.stop()

# Sidebar
with st.sidebar:
    settings = render_sidebar(bundle)

# Prediction wrapper
def _predict(paper: dict) -> dict:
    return predict_virality(
        paper      = paper,
        bundle     = bundle,
        tok        = tok,
        mdl        = mdl,
        device     = device,
        use_ensemble = settings.get("use_ensemble", True),
    )

# Masthead
st.markdown("""
<div class="masthead">
    <h1 class="masthead-title">🔬 PrePubVirality</h1>
    <p class="masthead-sub">
        Predict if your research paper will go viral before publication
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Prediction",
    "⚖️ Compare Papers",
    "📊 Model Insights",
    "📚 Examples",
])

with tab1:
    render_predict_tab(_predict, settings)

with tab2:
    render_compare_tab(_predict, settings)

with tab3:
    render_insights_tab(bundle)

with tab4:
    render_examples_tab(_predict, settings)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:var(--muted);font-size:1rem;font-weight:600">'
    'PrePubVirality: Research Paper Virality Predictor | FYP 2026'
    '</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align:center;color:var(--muted);font-size:0.78rem;">'
    'CS + Medicine · 94,435 papers · '
    'XGBoost ROC-AUC 0.9076 · Ensemble ROC-AUC 0.9073 (primary test 2023–2025)'
    '</p>',
    unsafe_allow_html=True
)
