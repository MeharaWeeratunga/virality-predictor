"""
Model loading utilities
"""
import pickle
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch


@st.cache_resource
def load_model(model_path='viral_predictor.pkl'):
    """Load the trained prediction model"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_specter():
    """Load SPECTER2 model for embeddings"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoModel.from_pretrained("allenai/specter2_base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading SPECTER2: {e}")
        return None, None, "cpu"


def get_model_components(saved_model):
    """Extract components from saved model"""
    if saved_model is None:
        return None
    
    return {
        'clf': saved_model['model'],
        'pca': saved_model['pca'],
        'le': saved_model['label_encoder'],
        'threshold': saved_model['threshold'],
        'optimal_threshold': saved_model['optimal_prediction_threshold'],
        'balanced_threshold': saved_model['balanced_prediction_threshold']
    }