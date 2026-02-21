"""
Paper virality prediction logic
"""
import numpy as np
from .embeddings import get_specter_embedding


def predict_virality(
    paper_data,
    model_components,
    tokenizer,
    specter_model,
    device,
    threshold_mode='optimal'
):
    """
    Predict if a paper will go viral
    
    Args:
        paper_data: dict with 'title', 'abstract', 'authors', 'categories'
        model_components: dict with model parts (clf, pca, le, thresholds)
        tokenizer: SPECTER2 tokenizer
        specter_model: SPECTER2 model
        device: computation device
        threshold_mode: 'optimal', 'balanced', or 'conservative'
        
    Returns:
        dict with prediction results
    """
    # Extract model components
    clf = model_components['clf']
    pca = model_components['pca']
    le = model_components['le']
    threshold = model_components['threshold']
    optimal_threshold = model_components['optimal_threshold']
    balanced_threshold = model_components['balanced_threshold']
    
    # Select threshold
    if threshold_mode == 'optimal':
        pred_threshold = optimal_threshold
    elif threshold_mode == 'balanced':
        pred_threshold = balanced_threshold
    else:
        pred_threshold = 0.5
    
    # Generate embedding
    text = paper_data['title'] + " " + paper_data['abstract']
    emb = get_specter_embedding(text, tokenizer, specter_model, device)
    
    # Extract metadata
    if isinstance(paper_data['authors'], str):
        num_authors = len([a for a in paper_data['authors'].split(',') if a.strip()])
    else:
        num_authors = paper_data['authors']
    
    abstract_len = len(paper_data['abstract'].split())
    cat = paper_data.get('categories', 'unknown')
    
    try:
        cat_enc = le.transform([cat])[0]
    except:
        cat_enc = 0
    
    # Combine features
    features = np.concatenate([emb, [num_authors, abstract_len, cat_enc]])
    emb_pca = pca.transform(features[:768].reshape(1, -1))
    features_final = np.hstack([emb_pca, features[768:].reshape(1, -1)])
    
    # Predict
    probability = clf.predict_proba(features_final)[0]
    prediction = 1 if probability[1] >= pred_threshold else 0
    expected_score = threshold * (probability[1] / 0.5)
    
    return {
        'prediction': 'VIRAL' if prediction == 1 else 'NOT VIRAL',
        'viral_probability': float(probability[1]),
        'not_viral_probability': float(probability[0]),
        'confidence': float(max(probability)),
        'expected_attention_score': float(expected_score),
        'threshold_used': float(threshold),
        'prediction_threshold': float(pred_threshold),
        'threshold_mode': threshold_mode,
        'metadata': {
            'num_authors': num_authors,
            'abstract_length': abstract_len,
            'category': cat
        }
    }