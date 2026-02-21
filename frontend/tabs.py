"""
Tab content for the Streamlit app
"""
import streamlit as st
import time
from .components import (
    display_prediction_result, 
    display_metrics, 
    display_detailed_analysis,
    get_example_papers
)


def render_single_prediction_tab(predict_fn, threshold_mode):
    """Render the single prediction tab"""
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
                time.sleep(1)
                
                result = predict_fn({
                    'title': title,
                    'abstract': abstract,
                    'authors': num_authors,
                    'categories': category
                }, threshold_mode=threshold_mode)
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            display_prediction_result(result)
            display_metrics(result)
            display_detailed_analysis(result)
            
            st.info("""
            ⚠️ **Disclaimer**: This model was trained primarily on physics and mathematics papers. 
            Predictions for other domains (especially CS/AI) may be less accurate. 
            Use as a reference tool, not definitive prediction.
            """)


def render_comparison_tab(predict_fn):
    """Render the paper comparison tab"""
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
                result_a = predict_fn({
                    'title': title_a,
                    'abstract': abstract_a,
                    'authors': authors_a,
                    'categories': cat_a
                }, threshold_mode='optimal')
                
                result_b = predict_fn({
                    'title': title_b,
                    'abstract': abstract_b,
                    'authors': authors_b,
                    'categories': cat_b
                }, threshold_mode='optimal')
            
            st.markdown("---")
            
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
            
            with col3:
                st.metric("Paper B Probability", f"{result_b['viral_probability']:.1%}")


def render_examples_tab():
    """Render the examples tab"""
    st.header("📚 Example Papers")
    st.write("Try these pre-filled examples")
    
    examples = get_example_papers()
    selected_example = st.selectbox("Select an example", list(examples.keys()))
    
    if st.button("📥 Load Example"):
        example = examples[selected_example]
        st.session_state.example_loaded = example
        st.success(f"✅ Loaded: {selected_example}")
        st.write("Go to the **Single Prediction** tab to see the prediction!")