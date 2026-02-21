"""
Reusable UI components
"""
import streamlit as st


def display_prediction_result(result):
    """Display prediction result box"""
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


def display_metrics(result):
    """Display prediction metrics in columns"""
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


def display_detailed_analysis(result):
    """Display detailed analysis in expander"""
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


def render_sidebar(threshold_mode):
    """Render sidebar content"""
    st.header("⚙️ Settings")
    
    mode = st.radio(
        "Prediction Mode",
        ['optimal', 'balanced', 'conservative'],
        index=['optimal', 'balanced', 'conservative'].index(threshold_mode),
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
    
    return mode


def get_example_papers():
    """Return dictionary of example papers"""
    return {
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