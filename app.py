import streamlit as st
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import time

# Page config
st.set_page_config(
    page_title="Research Paper Virality Predictor",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
    }
    .viral {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .not-viral {
        background-color: #d1ecf1;
        border-color: #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    with open('viral_predictor.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_specter():
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoModel.from_pretrained("allenai/specter2_base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

# Load models
try:
    saved_model = load_model()
    tokenizer, specter_model, device = load_specter()
    
    clf = saved_model['model']
    pca = saved_model['pca']
    le = saved_model['label_encoder']
    threshold = saved_model['threshold']
    optimal_threshold = saved_model['optimal_prediction_threshold']
    balanced_threshold = saved_model['balanced_prediction_threshold']
    
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# Prediction function
def get_specter_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = specter_model(**inputs)
    return outputs.pooler_output.squeeze().cpu().numpy()

def predict_virality(paper_data, threshold_mode='optimal'):
    # Select threshold
    if threshold_mode == 'optimal':
        pred_threshold = optimal_threshold
    elif threshold_mode == 'balanced':
        pred_threshold = balanced_threshold
    else:
        pred_threshold = 0.5
    
    # Generate embedding
    text = paper_data['title'] + " " + paper_data['abstract']
    emb = get_specter_embedding(text)
    
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

# ========================
# UI STARTS HERE
# ========================

# Header
st.markdown('<div class="main-header">🔬 Research Paper Virality Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict if your research paper will go viral before publication</div>', unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Models failed to load. Please ensure viral_predictor.pkl is in the same directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    threshold_mode = st.radio(
        "Prediction Mode",
        ['optimal', 'balanced', 'conservative'],
        help="""
        • **Optimal**: Best F1 score (catches 80% of viral papers)
        • **Balanced**: 50% recall with higher precision
        • **Conservative**: High precision (fewer false positives)
        """
    )
    
    st.markdown("---")
    
    st.header("📊 Model Info")
    st.metric("Training Papers", "1,565")
    st.metric("ROC-AUC Score", "0.65")
    st.metric("Primary Domains", "Physics, Math")
    
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.write("""
    This model predicts research paper virality using:
    - **SPECTER2** embeddings
    - Title & abstract content
    - Author count
    - Research category
    """)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "⚖️ Compare Papers", "📚 Examples"])

# TAB 1: Single Prediction
with tab1:
    st.header("Enter Paper Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        title = st.text_input(
            "Paper Title *",
            placeholder="e.g., Attention Is All You Need",
            help="Enter the full title of your research paper"
        )
        
        abstract = st.text_area(
            "Abstract *",
            height=200,
            placeholder="Paste your paper abstract here (at least 50 words recommended)...",
            help="The abstract should be at least 50 words for accurate predictions"
        )
    
    with col2:
        num_authors = st.number_input(
            "Number of Authors *",
            min_value=1,
            max_value=50,
            value=3,
            help="Total number of authors on the paper"
        )
        
        category = st.selectbox(
            "Primary Category *",
            [
                'cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.NE',
                'physics.comp-ph', 'quant-ph', 'cond-mat.stat-mech',
                'math.CO', 'math.NT', 'math.AG',
                'astro-ph', 'astro-ph.HE',
                'q-bio.NC', 'q-bio.QM',
                'stat.ML', 'stat.AP'
            ],
            help="Select the primary arXiv category"
        )
        
        st.info("**Note:** Model performs best on physics/math papers")
    
    if st.button("🚀 Predict Virality", type="primary", use_container_width=True):
        if not title or not abstract:
            st.error("❌ Please fill in both title and abstract!")
        elif len(abstract.split()) < 30:
            st.warning("⚠️ Abstract seems short. Predictions are more accurate with longer abstracts (50+ words).")
        else:
            with st.spinner("🔮 Analyzing your paper..."):
                time.sleep(1)  # Dramatic effect
                
                result = predict_virality({
                    'title': title,
                    'abstract': abstract,
                    'authors': num_authors,
                    'categories': category
                }, threshold_mode=threshold_mode)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Main prediction box
            if result['prediction'] == 'VIRAL':
                st.markdown(f"""
                <div class="prediction-box viral">
                    <h2 style="margin:0; color: #856404;">🔥 VIRAL</h2>
                    <p style="margin:0.5rem 0 0 0; color: #856404;">This paper has high viral potential!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box not-viral">
                    <h2 style="margin:0; color: #0c5460;">📄 NOT VIRAL</h2>
                    <p style="margin:0.5rem 0 0 0; color: #0c5460;">This paper may have limited viral reach.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Viral Probability",
                    f"{result['viral_probability']:.1%}",
                    help="Likelihood of becoming viral based on model analysis"
                )
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{result['confidence']:.1%}",
                    help="Model's confidence in this prediction"
                )
            
            with col3:
                st.metric(
                    "Expected Attention Score",
                    f"{result['expected_attention_score']:.1f}",
                    help="Predicted Altmetric attention score"
                )
            
            # Details
            with st.expander("🔍 Detailed Analysis"):
                st.write(f"""
                **Threshold Information:**
                - Prediction mode: **{result['threshold_mode'].title()}**
                - Threshold used: **{result['prediction_threshold']:.3f}**
                - Virality cutoff: Papers with attention score > **{result['threshold_used']:.1f}**
                
                **Paper Metadata:**
                - Number of authors: **{result['metadata']['num_authors']}**
                - Abstract length: **{result['metadata']['abstract_length']} words**
                - Category: **{result['metadata']['category']}**
                
                **Interpretation:**
                {f"✅ The model predicts this paper **will go viral** with {result['viral_probability']:.1%} probability." if result['prediction'] == 'VIRAL' else f"❌ The model predicts this paper **will not go viral** (only {result['viral_probability']:.1%} probability)."}
                
                Expected Altmetric attention score: **{result['expected_attention_score']:.1f}** 
                (Viral threshold: **{result['threshold_used']:.1f}**)
                """)
            
            # Disclaimer
            st.info("""
            ⚠️ **Disclaimer**: This model was trained primarily on physics and mathematics papers. 
            Predictions for other domains (especially CS/AI) may be less accurate. 
            Use as a reference tool, not definitive prediction.
            """)

# TAB 2: Compare Papers
with tab2:
    st.header("Compare Two Papers")
    st.write("See which paper is more likely to go viral")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Paper A")
        title_a = st.text_input("Title A", key="title_a")
        abstract_a = st.text_area("Abstract A", height=150, key="abstract_a")
        authors_a = st.number_input("Authors A", 1, 50, 3, key="authors_a")
        cat_a = st.selectbox("Category A", ['cs.AI', 'cs.LG', 'quant-ph', 'math.NT', 'astro-ph'], key="cat_a")
    
    with col2:
        st.subheader("📄 Paper B")
        title_b = st.text_input("Title B", key="title_b")
        abstract_b = st.text_area("Abstract B", height=150, key="abstract_b")
        authors_b = st.number_input("Authors B", 1, 50, 3, key="authors_b")
        cat_b = st.selectbox("Category B", ['cs.AI', 'cs.LG', 'quant-ph', 'math.NT', 'astro-ph'], key="cat_b")
    
    if st.button("⚖️ Compare Papers", type="primary"):
        if not all([title_a, abstract_a, title_b, abstract_b]):
            st.error("Please fill in all fields for both papers!")
        else:
            with st.spinner("Analyzing both papers..."):
                result_a = predict_virality({
                    'title': title_a,
                    'abstract': abstract_a,
                    'authors': authors_a,
                    'categories': cat_a
                }, threshold_mode='optimal')
                
                result_b = predict_virality({
                    'title': title_b,
                    'abstract': abstract_b,
                    'authors': authors_b,
                    'categories': cat_b
                }, threshold_mode='optimal')
            
            st.markdown("---")
            
            # Comparison
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric("Paper A Probability", f"{result_a['viral_probability']:.1%}")
            
            with col2:
                ratio = result_a['viral_probability'] / result_b['viral_probability']
                if ratio > 1.2:
                    st.success(f"📈 Paper A is **{ratio:.1f}x** more likely to be viral")
                elif ratio < 0.8:
                    st.success(f"📈 Paper B is **{1/ratio:.1f}x** more likely to be viral")
                else:
                    st.info("📊 Both papers have similar viral potential")
            
            with col2:
                st.metric("Paper B Probability", f"{result_b['viral_probability']:.1%}")

# TAB 3: Examples
with tab3:
    st.header("📚 Example Papers")
    st.write("Try these pre-filled examples")
    
    examples = {
        "Quantum Physics Paper": {
            'title': 'Quantum Entanglement in Many-Body Systems',
            'abstract': 'We investigate quantum entanglement properties in many-body quantum systems at finite temperature. Using a combination of analytical methods and numerical simulations, we demonstrate novel scaling behavior of entanglement entropy near critical points. Our results have implications for quantum information theory and condensed matter physics.',
            'authors': 4,
            'categories': 'quant-ph'
        },
        "Machine Learning Paper": {
            'title': 'Deep Learning for Natural Language Understanding',
            'abstract': 'Recent advances in deep learning have revolutionized natural language processing. We propose a novel architecture combining transformers with graph neural networks to improve language understanding tasks. Experiments on benchmark datasets show significant improvements over baseline methods. Our approach achieves state-of-the-art results on several challenging tasks.',
            'authors': 6,
            'categories': 'cs.LG'
        },
        "Mathematics Paper": {
            'title': 'On Prime Number Distribution in Arithmetic Progressions',
            'abstract': 'We present new results on the distribution of prime numbers in arithmetic progressions. Building on classical analytic number theory techniques, we establish sharper bounds for the error term in the prime number theorem. The proof relies on sophisticated estimates for L-functions and exponential sums.',
            'authors': 2,
            'categories': 'math.NT'
        }
    }
    
    selected_example = st.selectbox("Select an example", list(examples.keys()))
    
    if st.button("📥 Load Example"):
        example = examples[selected_example]
        st.session_state.example_loaded = example
        st.success(f"✅ Loaded: {selected_example}")
        st.write("Go to the **Single Prediction** tab to see the prediction!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Research Paper Virality Predictor</strong> | FYP Project 2025</p>
    <p>Model trained on 1,565 arXiv papers | Primary domains: Physics, Mathematics, Astronomy</p>
</div>
""", unsafe_allow_html=True)