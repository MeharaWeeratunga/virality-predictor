"""
Model loading utilities for bundle.

Bundle keys (viral_predictor_v12_altmetric_no_hindex.pkl):
  xgb_model, rf_model, lr_model, baseline_model
  pca, scaler, ohe
  thresholds, ensemble_weights
  meta_numeric, cat_cols, top_venue_whitelist
  specter_finetuned_path, specter_local_path, specter_model_name
  train_year_cutoff, virality_percentile, eval_cutoff
  results_summary, ablation_results, stratified_eval, topic_virality
"""
import pickle
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import os


@st.cache_resource(show_spinner="Loading model bundle…")
def load_bundle(model_path: str = "viral_predictor_v12_altmetric_no_hindex.pkl"):
    """Load the v12 model bundle from disk."""
    if not os.path.exists(model_path):
        st.error(f"❌ Bundle not found: `{model_path}`")
        return None
    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except Exception as e:
        st.error(f"❌ Failed to load bundle: {e}")
        return None


@st.cache_resource(show_spinner="Loading SPECTER2…")
def load_specter(_bundle: dict):
    """
    Load SPECTER2 tokenizer + model.
    Priority: fine-tuned → local frozen → HuggingFace hub.
    """
    if _bundle is None:
        return None, None, "cpu"

    ft_path  = _bundle.get("specter_finetuned_path")
    loc_path = _bundle.get("specter_local_path")
    hub_name = _bundle.get("specter_model_name", "allenai/specter2_base")

    src = (
        ft_path  if (ft_path  and os.path.isdir(ft_path))  else
        loc_path if (loc_path and os.path.isdir(loc_path)) else
        hub_name
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(src)
        mdl = AutoModel.from_pretrained(src).to(device)
        mdl.eval()
        return tok, mdl, device
    except Exception as e:
        st.error(f"❌ SPECTER2 load failed: {e}")
        return None, None, "cpu"