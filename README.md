# TSLA Stock Price Prediction

Predicting Tesla stock price movement using machine learning and sentiment analysis.

## Project Phases

- [x] Phase 1: Baseline model with technical indicators
- [x] Phase 2: Add sentiment analysis from financial news
- [x] Phase 3: Advanced models (XGBoost)
- [x] Phase 4: Deployment with Streamlit

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### Run Individual Phases
```bash
# Phase 1: Baseline Logistic Regression
python src/phase1_baseline.py

# Phase 2: With Sentiment Analysis
python src/phase2_with_sentiment.py

# Phase 3: XGBoost Model
python src/phase3_xgboost.py
```

### Run Interactive Dashboard
```bash
streamlit run app.py
```

## Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository: `dalzouby/Projects`
6. Set Main file path: `app.py`
7. Click "Deploy"

The app will be live at: `https://your-app-name.streamlit.app`

### Local Deployment
```bash
streamlit run app.py
```

## Results

See individual phase scripts for detailed results and visualizations.
