# Pre-Publication Virality Prediction of Scientific Research Papers
# Project Overview 📖

This project is a research-based machine learning prototype that predicts the virality of scientific research papers at the time of publication, using only features available before citation data exists.

Traditional academic impact prediction relies heavily on citation counts, which take months or years to accumulate. This system addresses that limitation by using semantic embeddings from paper abstracts combined with lightweight metadata to estimate whether a paper will gain significant attention shortly after publication.

The system is deployed as an interactive Streamlit web application where users can input a research paper’s title and abstract and obtain real-time virality predictions.

# Objectives 🎯

🔹 Predict whether a research paper will become viral at the time of publication.

🔹 Use only pre-publication information (no citation-based features).

🔹 Leverage SPECTER2 embeddings for semantic understanding of paper content.

🔹 Provide an interactive web interface for researchers and users.

🔹 Support interpretability through probability outputs and attention score estimation.

# Key Features 🚀
## Single Paper Prediction ✅

Users can enter:

🔹Paper title

🔹Abstract

🔹Number of authors

🔹Research category

The system outputs:

🔹Viral / Not Viral prediction

🔹Virality probability score

🔹Expected attention score

🔹Confidence interpretation

## Paper Comparison Mode ✅

Users can compare two papers and observe:

🔹Which paper is more likely to go viral

🔹Relative virality probability difference

🔹Expected attention score comparison

## Threshold-Based Prediction ✅

The prototype supports different prediction modes:

🔹Optimal Threshold (default)

🔹Balanced Threshold

🔹Conservative Threshold

This allows flexible decision-making based on user preference.

## Efficient Model Loading ✅

The backend uses Streamlit caching (st.cache_resource) to ensure:

🔹SPECTER2 is loaded only once

🔹Random Forest model is loaded only once

🔹Faster real-time predictions

# Machine Learning Approach 🧠

The model is trained using a multimodal feature set:

🔹 Semantic Features

Paper abstracts are converted into embeddings using SPECTER2

Embeddings capture contextual and scientific meaning of research papers

🔹 Metadata Features

Number of authors

Abstract length

Research category encoding

🔹 Dimensionality Reduction

PCA is applied to reduce the embedding dimension while preserving semantic information

🔹 Classifier

Random Forest Classifier (trained on combined features)

# Model Info
- **Training data**: 1,565 arXiv papers
- **Domains**: Physics (78%), Mathematics, Astronomy
- **ROC-AUC**: 0.67
- **Optimal threshold**: 0.373