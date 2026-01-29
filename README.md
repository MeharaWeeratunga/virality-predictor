# Research Paper Virality Predictor

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Model File
- Download `viral_predictor.pkl` from Google Drive
- Place it in the same folder as `app.py`

### 3. Run the App
```bash
streamlit run app.py
```

## Project Structure
```
├── app.py                  # Main Streamlit application
├── viral_predictor.pkl     # Trained model (download from Drive)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Features
- Single paper prediction
- Compare two papers
- Multiple threshold modes (optimal/balanced/conservative)
- Pre-loaded examples

## Model Info
- **Training data**: 1,565 arXiv papers
- **Domains**: Physics (78%), Mathematics, Astronomy
- **ROC-AUC**: 0.65
- **Optimal threshold**: 0.373