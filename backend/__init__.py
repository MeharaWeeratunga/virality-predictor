"""
Backend - Pre-Publication Virality Predictor
"""
from .model_loader import load_bundle, load_specter
from .predictor import predict_virality

__all__ = ['load_bundle', 'load_specter', 'predict_virality']