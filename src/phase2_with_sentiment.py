import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ============================================
# PHASE 2: DERIVED SENTIMENT ANALYSIS
# ============================================
# Instead of API news (which has limitations), we'll create
# "derived sentiment" from price action and volume - a common
# technique in quantitative finance
print("="*60)
print("PHASE 2: STOCK PREDICTION WITH DERIVED SENTIMENT")
print("="*60)

ticker = "TSLA"

# ============================================
# STEP 1: Get Stock Data
# ============================================
print("\n[1/5] Downloading TSLA stock data...")
data = yf.download(ticker, period="2y", interval="1d")

# Flatten multi-level columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ============================================
# STEP 2: Create Technical Features
# ============================================
print("[2/5] Creating technical features...")

# Basic features from Phase 1
data['Returns'] = data['Close'].pct_change()
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['Volatility'] = data['Returns'].rolling(window=20).std()
data['Price_to_MA'] = data['Close'] / data['MA_20']

# Additional technical indicators
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']

# ============================================
# STEP 3: Create DERIVED SENTIMENT
# ============================================
print("[3/5] Creating derived sentiment indicators...")

# Sentiment Indicator 1: Price Momentum
# Positive when price is rising strongly, negative when falling
data['Price_Momentum'] = (data['MA_5'] - data['MA_20']) / data['MA_20']

# Sentiment Indicator 2: Volume Sentiment
# High volume on up days = positive sentiment
# High volume on down days = negative sentiment
data['Volume_Sentiment'] = data['Returns'] * data['Volume_Ratio']

# Sentiment Indicator 3: RSI (Relative Strength Index)
# Classic momentum indicator: >70 = overbought, <30 = oversold
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data['Close'])
data['RSI_Sentiment'] = (data['RSI'] - 50) / 50  # Normalize to -1 to 1

# Combined Sentiment Score (weighted average)
data['Sentiment'] = (
    0.4 * data['Price_Momentum'] + 
    0.3 * data['Volume_Sentiment'] + 
    0.3 * data['RSI_Sentiment']
)

# Normalize sentiment to -1 to 1 range
data['Sentiment'] = data['Sentiment'].clip(-1, 1)

# Target variable
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop NaN values
data = data.dropna()

print(f"Stock data shape: {data.shape}")
print(f"\nSentiment statistics:")
print(data['Sentiment'].describe())

# ============================================
# STEP 4: Train Models (With and Without Sentiment)
# ============================================
print("\n[4/5] Training models...")

# Features without sentiment (baseline from Phase 1)
features_base = ['Returns', 'MA_20', 'Volatility', 'Price_to_MA']

# Features with all sentiment indicators
features_with_sentiment = features_base + [
    'Price_Momentum', 'Volume_Sentiment', 'RSI_Sentiment', 'Sentiment'
]

X_base = data[features_base]
X_sentiment = data[features_with_sentiment]
y = data['Target']

# Train-test split (80/20)
split_idx = int(len(X_base) * 0.8)

# Baseline model (without sentiment)
X_train_base = X_base[:split_idx]
X_test_base = X_base[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

model_base = LogisticRegression(random_state=42, max_iter=1000)
model_base.fit(X_train_base, y_train)
y_pred_base = model_base.predict(X_test_base)

# Model with sentiment
X_train_sent = X_sentiment[:split_idx]
X_test_sent = X_sentiment[split_idx:]

model_sentiment = LogisticRegression(random_state=42, max_iter=1000)
model_sentiment.fit(X_train_sent, y_train)
y_pred_sent = model_sentiment.predict(X_test_sent)

# ============================================
# STEP 5: Evaluate and Compare
# ============================================
print("\n[5/5] Evaluating models...")

base_acc = accuracy_score(y_test, y_pred_base)
sent_acc = accuracy_score(y_test, y_pred_sent)

print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)

print(f"\n📊 BASELINE MODEL (Phase 1 features only):")
print(f"   Test Accuracy: {base_acc:.2%}")
print(f"   Features: {', '.join(features_base)}")

print(f"\n🎯 MODEL WITH SENTIMENT (Phase 2 - all features):")
print(f"   Test Accuracy: {sent_acc:.2%}")
print(f"   Added features: Price Momentum, Volume Sentiment, RSI")

improvement = sent_acc - base_acc
print(f"\n💡 IMPROVEMENT: {improvement:+.2%}")

if improvement > 0:
    print("   ✅ Sentiment analysis helped!")
elif improvement == 0:
    print("   ➡️  No change in accuracy")
else:
    print("   ⚠️  Sentiment didn't improve predictions")

print("\n" + "-"*60)
print("Baseline Model - Confusion Matrix:")
cm_base = confusion_matrix(y_test, y_pred_base)
print(cm_base)
print(f"Predicting 'Up': {cm_base[1, 1]} correct out of {cm_base[1, 0] + cm_base[1, 1]} total")

print("\nSentiment Model - Confusion Matrix:")
cm_sent = confusion_matrix(y_test, y_pred_sent)
print(cm_sent)
print(f"Predicting 'Up': {cm_sent[1, 1]} correct out of {cm_sent[1, 0] + cm_sent[1, 1]} total")

print("\n" + "-"*60)
print("Sentiment Model - Detailed Classification Report:")
print(classification_report(y_test, y_pred_sent, target_names=['Down', 'Up'], zero_division=0))

# Feature importance
print("\n" + "-"*60)
print("Feature Importance (with Sentiment):")
feature_importance = pd.DataFrame({
    'Feature': features_with_sentiment,
    'Coefficient': model_sentiment.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(feature_importance)

# ============================================
# STEP 6: Visualizations
# ============================================
print("\nCreating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Plot 1: Sentiment over time (full data)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(data.index, data['Sentiment'], color='purple', linewidth=1.5, alpha=0.7)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(data.index, 0, data['Sentiment'], 
                  where=(data['Sentiment'] > 0), alpha=0.3, color='green', label='Positive Sentiment')
ax1.fill_between(data.index, 0, data['Sentiment'], 
                  where=(data['Sentiment'] < 0), alpha=0.3, color='red', label='Negative Sentiment')
ax1.set_title('Derived Sentiment Score Over Time', fontsize=14, fontweight='bold')
ax1.set_ylabel('Sentiment Score')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Sentiment components
ax2 = fig.add_subplot(gs[1, 0])
test_dates = data.index[split_idx:]
ax2.plot(test_dates, data['Price_Momentum'].iloc[split_idx:], label='Price Momentum', alpha=0.7)
ax2.plot(test_dates, data['Volume_Sentiment'].iloc[split_idx:], label='Volume Sentiment', alpha=0.7)
ax2.plot(test_dates, data['RSI_Sentiment'].iloc[split_idx:], label='RSI Sentiment', alpha=0.7)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Sentiment Components (Test Period)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sentiment Value')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Accuracy comparison
ax3 = fig.add_subplot(gs[1, 1])
models = ['Baseline\n(Phase 1)', 'With Sentiment\n(Phase 2)']
accuracies = [base_acc * 100, sent_acc * 100]
colors = ['#ff6b6b', '#4ecdc4']
bars = ax3.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
ax3.set_ylabel('Accuracy (%)', fontweight='bold')
ax3.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 100])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 4: Price with baseline predictions
ax4 = fig.add_subplot(gs[2, :])
test_prices = data['Close'].iloc[split_idx:]
correct_base = (y_pred_base == y_test)

ax4.plot(test_dates, test_prices, label='TSLA Price', color='blue', alpha=0.6, linewidth=2)
ax4.scatter(test_dates[correct_base], test_prices[correct_base], 
           color='green', label='Correct', alpha=0.5, s=20)
ax4.scatter(test_dates[~correct_base], test_prices[~correct_base], 
           color='red', label='Wrong', alpha=0.5, s=20)
ax4.set_title('Phase 1 Baseline Model Predictions', fontsize=12, fontweight='bold')
ax4.set_ylabel('Price ($)', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Price with sentiment model predictions
ax5 = fig.add_subplot(gs[3, :])
correct_sent = (y_pred_sent == y_test)

ax5.plot(test_dates, test_prices, label='TSLA Price', color='blue', alpha=0.6, linewidth=2)
ax5.scatter(test_dates[correct_sent], test_prices[correct_sent], 
           color='green', label='Correct', alpha=0.5, s=20)
ax5.scatter(test_dates[~correct_sent], test_prices[~correct_sent], 
           color='red', label='Wrong', alpha=0.5, s=20)
ax5.set_title('Phase 2 Model with Sentiment Predictions', fontsize=12, fontweight='bold')
ax5.set_ylabel('Price ($)', fontweight='bold')
ax5.set_xlabel('Date', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.savefig('results/tsla_phase2_derived_sentiment.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'results/tsla_phase2_derived_sentiment.png'")
plt.show()

print("\n" + "="*60)
print("Phase 2 Complete! 🎉")
print("="*60)
print("\nWhat we did:")
print("- Created 'derived sentiment' from price action and volume")
print("- Added Price Momentum, Volume Sentiment, and RSI indicators")
print("- These mimic real market sentiment without API limitations")
print(f"\nResults:")
print(f"- Baseline accuracy: {base_acc:.2%}")
print(f"- With sentiment: {sent_acc:.2%}")
print(f"- Change: {improvement:+.2%}")
print("\nNext: Phase 3 will use XGBoost for better predictions!")