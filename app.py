import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="TSLA Stock Predictor",
    page_icon="📈",
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
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📈 TSLA Stock Price Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Configuration")
ticker = st.sidebar.selectbox("Select Stock", ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN"], index=0)
period = st.sidebar.selectbox("Historical Period", ["1y", "2y", "5y"], index=1)
train_button = st.sidebar.button("🚀 Train Model", type="primary")

# Helper functions
@st.cache_data
def load_data(ticker, period):
    """Load stock data"""
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def engineer_features(data):
    """Create all features"""
    # Basic features
    data['Returns'] = data['Close'].pct_change()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['Price_to_MA'] = data['Close'] / data['MA_20']
    
    # Additional indicators
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    # Derived sentiment
    data['Price_Momentum'] = (data['MA_5'] - data['MA_20']) / data['MA_20']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_Sentiment'] = (data['RSI'] - 50) / 50
    
    data['Volume_Sentiment'] = data['Returns'] * data['Volume_Ratio']
    data['Sentiment'] = (
        0.4 * data['Price_Momentum'] + 
        0.3 * data['Volume_Sentiment'] + 
        0.3 * data['RSI_Sentiment']
    ).clip(-1, 1)
    
    # Additional features
    data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Close_Open_Change'] = (data['Close'] - data['Open']) / data['Open']
    data['Returns_Lag1'] = data['Returns'].shift(1)
    data['Volume_Lag1'] = data['Volume_Ratio'].shift(1)
    
    # Target
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return data.dropna()

def train_models(X_train, y_train):
    """Train both models"""
    # Logistic Regression
    model_lr = LogisticRegression(random_state=42, max_iter=1000)
    model_lr.fit(X_train, y_train)
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    )
    model_xgb.fit(X_train, y_train)
    
    return model_lr, model_xgb

# Main app
if train_button:
    with st.spinner(f"Loading {ticker} data..."):
        # Load and prepare data
        data = load_data(ticker, period)
        data = engineer_features(data)
        
        # Features
        features = [
            'Returns', 'MA_20', 'Volatility', 'Price_to_MA',
            'Price_Momentum', 'Volume_Sentiment', 'RSI_Sentiment', 'Sentiment',
            'High_Low_Range', 'Close_Open_Change', 'Returns_Lag1', 'Volume_Lag1'
        ]
        
        X = data[features]
        y = data['Target']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
    with st.spinner("Training models..."):
        # Train models
        model_lr, model_xgb = train_models(X_train, y_train)
        
        # Predictions
        y_pred_lr = model_lr.predict(X_test)
        y_pred_xgb = model_xgb.predict(X_test)
        
        # Accuracies
        lr_acc = accuracy_score(y_test, y_pred_lr)
        xgb_acc = accuracy_score(y_test, y_pred_xgb)
        
        # Confusion matrices
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    
    st.success("✅ Models trained successfully!")
    
    # Display results
    st.markdown("## 📊 Model Performance")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🔴 Logistic Regression",
            value=f"{lr_acc:.1%}",
            delta=f"{(lr_acc - 0.5):.1%} vs random"
        )
    
    with col2:
        st.metric(
            label="🟢 XGBoost",
            value=f"{xgb_acc:.1%}",
            delta=f"{(xgb_acc - lr_acc):.1%} improvement"
        )
    
    with col3:
        improvement = xgb_acc - lr_acc
        st.metric(
            label="💡 Performance Gain",
            value=f"{improvement:+.1%}",
            delta="Better" if improvement > 0 else "Worse"
        )
    
    # Confusion Matrices
    st.markdown("### Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Logistic Regression**")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='RdYlGn', ax=ax1, 
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_title(f'Accuracy: {lr_acc:.1%}')
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        st.markdown("**XGBoost**")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='RdYlGn', ax=ax2,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title(f'Accuracy: {xgb_acc:.1%}')
        st.pyplot(fig2)
        plt.close()
    
    # Feature Importance
    st.markdown("### 🎯 Feature Importance (XGBoost)")
    feature_imp = pd.DataFrame({
        'Feature': features,
        'Importance': model_xgb.feature_importances_
    }).sort_values('Importance', ascending=False).head(8)
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.barh(range(len(feature_imp)), feature_imp['Importance'], color='steelblue', alpha=0.7)
    ax3.set_yticks(range(len(feature_imp)))
    ax3.set_yticklabels(feature_imp['Feature'])
    ax3.set_xlabel('Importance Score')
    ax3.set_title('Top 8 Most Important Features')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig3)
    plt.close()
    
    # Price predictions visualization
    st.markdown("### 📈 Predictions on Price Chart")
    
    test_dates = data.index[split_idx:]
    test_prices = data['Close'].iloc[split_idx:]
    
    tab1, tab2 = st.tabs(["Logistic Regression", "XGBoost"])
    
    with tab1:
        correct_lr = (y_pred_lr == y_test)
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        ax4.plot(test_dates, test_prices, label=f'{ticker} Price', color='blue', alpha=0.6, linewidth=2)
        ax4.scatter(test_dates[correct_lr], test_prices[correct_lr], 
                   color='green', label='Correct Prediction', alpha=0.6, s=30)
        ax4.scatter(test_dates[~correct_lr], test_prices[~correct_lr], 
                   color='red', label='Wrong Prediction', alpha=0.6, s=30)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price ($)')
        ax4.set_title(f'{ticker} Price with Logistic Regression Predictions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig4)
        plt.close()
    
    with tab2:
        correct_xgb = (y_pred_xgb == y_test)
        fig5, ax5 = plt.subplots(figsize=(14, 6))
        ax5.plot(test_dates, test_prices, label=f'{ticker} Price', color='blue', alpha=0.6, linewidth=2)
        ax5.scatter(test_dates[correct_xgb], test_prices[correct_xgb], 
                   color='green', label='Correct Prediction', alpha=0.6, s=30)
        ax5.scatter(test_dates[~correct_xgb], test_prices[~correct_xgb], 
                   color='red', label='Wrong Prediction', alpha=0.6, s=30)
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Price ($)')
        ax5.set_title(f'{ticker} Price with XGBoost Predictions')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig5)
        plt.close()
    
    # Latest prediction
    st.markdown("### 🔮 Latest Prediction")
    
    latest_features = X.iloc[-1:].values
    latest_pred_xgb = model_xgb.predict(latest_features)[0]
    latest_proba = model_xgb.predict_proba(latest_features)[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Tomorrow's Prediction**")
        prediction_text = "📈 UP" if latest_pred_xgb == 1 else "📉 DOWN"
        st.markdown(f"<h2 style='text-align: center; color: {'green' if latest_pred_xgb == 1 else 'red'};'>{prediction_text}</h2>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Confidence**")
        confidence = max(latest_proba) * 100
        st.markdown(f"<h2 style='text-align: center;'>{confidence:.1f}%</h2>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Current Price**")
        current_price = data['Close'].iloc[-1]
        st.markdown(f"<h2 style='text-align: center;'>${current_price:.2f}</h2>", unsafe_allow_html=True)
    
    # Classification report
    with st.expander("📋 Detailed Classification Report (XGBoost)"):
        report = classification_report(y_test, y_pred_xgb, target_names=['Down', 'Up'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
    
    # Raw data
    with st.expander("📊 View Raw Data"):
        st.dataframe(data.tail(20))

else:
    st.info("👈 Configure settings in the sidebar and click **Train Model** to get started!")
    
    # Show example visualization
    st.markdown("### 💡 What This App Does:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Data Collection")
        st.write("Fetches historical stock data and engineers 12+ technical features including momentum, volatility, and sentiment indicators.")
    
    with col2:
        st.markdown("#### 2️⃣ Model Training")
        st.write("Trains both Logistic Regression and XGBoost models to predict next-day price movements (Up/Down).")
    
    with col3:
        st.markdown("#### 3️⃣ Predictions")
        st.write("Provides accuracy metrics, feature importance, and visual predictions on price charts.")
    
    st.markdown("---")
    st.markdown("**Built with:** Python, Streamlit, XGBoost, yfinance, scikit-learn")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built as part of a 4-phase ML project | Phase 4: Interactive Dashboard"
    "</div>", 
    unsafe_allow_html=True
)