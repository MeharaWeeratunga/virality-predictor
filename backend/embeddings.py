"""
Text embedding generation
"""
import torch


def get_specter_embedding(text, tokenizer, model, device):
    """
    Generate SPECTER2 embedding for given text
    
    Args:
        text: Input text (title + abstract)
        tokenizer: SPECTER2 tokenizer
        model: SPECTER2 model
        device: CPU or CUDA
        
    Returns:
        numpy array of embedding
    """
    inputs = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.pooler_output.squeeze().cpu().numpy()