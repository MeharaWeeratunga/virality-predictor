"""
Frontend module for Research Paper Virality Predictor UI
"""
from .styles import get_custom_css
from .components import display_prediction_result, display_metrics
from .tabs import render_single_prediction_tab, render_comparison_tab, render_examples_tab

__all__ = [
    'get_custom_css',
    'display_prediction_result',
    'display_metrics',
    'render_single_prediction_tab',
    'render_comparison_tab',
    'render_examples_tab'
]