"""
Backend module for Research Paper Virality Predictor
"""
from .model_loader import load_model, load_specter
from .predictor import predict_virality
from .embeddings import get_specter_embedding

__all__ = [
    'load_model',
    'load_specter', 
    'predict_virality',
    'get_specter_embedding'
]