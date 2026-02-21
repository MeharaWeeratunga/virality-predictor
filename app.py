"""
PrePubVirality
Main Streamlit Application
"""
import streamlit as st

# Backend imports
from backend.model_loader import load_model, load_specter, get_model_components
from backend.predictor import predict_virality

# Frontend imports
from frontend.styles import get_custom_css
from frontend.components import render_sidebar
from frontend.tabs import (
    render_single_prediction_tab,
    render_comparison_tab,
    render_examples_tab
)

# Page config
st.set_page_config(
    page_title="PrePubVirality: Research Paper Virality Predictor",
    page_icon="🔬",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Load models
saved_model = load_model()
tokenizer, specter_model, device = load_specter()
model_components = get_model_components(saved_model)

# Check if models loaded successfully
if saved_model is None or tokenizer is None or model_components is None:
    st.error("⚠️ Models failed to load. Please ensure viral_predictor.pkl is in the same directory.")
    st.stop()

# Header
st.markdown(
    '<div class="main-header">🔬 Research Paper Virality Predictor</div>', 
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Predict if your research paper will go viral before publication</div>', 
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    threshold_mode = render_sidebar('optimal')

# Create prediction function wrapper
def predict_fn(paper_data, threshold_mode='optimal'):
    return predict_virality(
        paper_data,
        model_components,
        tokenizer,
        specter_model,
        device,
        threshold_mode
    )

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "⚖️ Compare Papers", "📚 Examples"])

with tab1:
    render_single_prediction_tab(predict_fn, threshold_mode)

with tab2:
    render_comparison_tab(predict_fn)

with tab3:
    render_examples_tab()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>PrePubVirality: Research Paper Virality Predictor</strong> | FYP Project 2025</p>
    <p>Model trained on 1,565 arXiv papers | Primary domains: Physics, Mathematics, Astronomy</p>
</div>
""", unsafe_allow_html=True)