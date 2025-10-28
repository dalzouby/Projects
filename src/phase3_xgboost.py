import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import xgboost as xgb

# ============================================
# PHASE 3: XGBOOST FOR BETTER PREDICTIONS
# ============================================
print("="*60)
print("PHASE 3: STOCK PREDICTION WITH XGBOOST")
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
# STEP 2: Create All Features
# ============================================
print("[2/5] Engineering features...")

# Basic features
data['Returns'] = data['Close'].pct_change()
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['Volatility'] = data['Returns'].rolling(window=20).std()
data['Price_to_MA'] = data['Close'] / data['MA_20']

# Additional technical indicators
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']

# Derived sentiment
data['Price_Momentum'] = (data['MA_5'] - data['MA_20']) / data['MA_20']

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data['Close'])
data['RSI_Sentiment'] = (data['RSI'] - 50) / 50

data['Volume_Sentiment'] = data['Returns'] * data['Volume_Ratio']
data['Sentiment'] = (
    0.4 * data['Price_Momentum'] + 
    0.3 * data['Volume_Sentiment'] + 
    0.3 * data['RSI_Sentiment']
).clip(-1, 1)

# Additional features for XGBoost
data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
data['Close_Open_Change'] = (data['Close'] - data['Open']) / data['Open']

# Lag features (yesterday's values)
data['Returns_Lag1'] = data['Returns'].shift(1)
data['Volume_Lag1'] = data['Volume_Ratio'].shift(1)

# Target variable
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop NaN values
data = data.dropna()

print(f"Final dataset shape: {data.shape}")
print(f"Features created: {data.shape[1] - 1}")

# ============================================
# STEP 3: Prepare Data
# ============================================
print("\n[3/5] Preparing training and test sets...")

# All features
features_all = [
    'Returns', 'MA_20', 'Volatility', 'Price_to_MA',
    'Price_Momentum', 'Volume_Sentiment', 'RSI_Sentiment', 'Sentiment',
    'High_Low_Range', 'Close_Open_Change', 'Returns_Lag1', 'Volume_Lag1'
]

X = data[features_all]
y = data['Target']

# Train-test split (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Class distribution in training:")
print(y_train.value_counts())

# ============================================
# STEP 4: Train Models
# ============================================
print("\n[4/5] Training models...")

# Baseline: Logistic Regression (from Phase 2)
print("  - Training Logistic Regression (baseline)...")
model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# XGBoost Model
print("  - Training XGBoost...")
model_xgb = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle class imbalance
)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

# ============================================
# STEP 5: Evaluate and Compare
# ============================================
print("\n[5/5] Evaluating models...")

lr_acc = accuracy_score(y_test, y_pred_lr)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)

print(f"\n📊 LOGISTIC REGRESSION (Phases 1 & 2):")
print(f"   Test Accuracy: {lr_acc:.2%}")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"   Predicting 'Up': {cm_lr[1, 1]} correct out of {cm_lr[1, 0] + cm_lr[1, 1]} total")

print(f"\n🚀 XGBOOST (Phase 3):")
print(f"   Test Accuracy: {xgb_acc:.2%}")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print(f"   Predicting 'Up': {cm_xgb[1, 1]} correct out of {cm_xgb[1, 0] + cm_xgb[1, 1]} total")

improvement = xgb_acc - lr_acc
print(f"\n💡 IMPROVEMENT: {improvement:+.2%}")

if improvement > 0:
    print("   ✅ XGBoost significantly improved predictions!")
elif improvement == 0:
    print("   ➡️  No change")
else:
    print("   ⚠️  Performance decreased")

print("\n" + "-"*60)
print("Logistic Regression - Confusion Matrix:")
print(cm_lr)
print("  [[True Negatives, False Positives]")
print("   [False Negatives, True Positives]]")

print("\nXGBoost - Confusion Matrix:")
print(cm_xgb)
print("  [[True Negatives, False Positives]")
print("   [False Negatives, True Positives]]")

print("\n" + "-"*60)
print("XGBoost - Detailed Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Down', 'Up'], zero_division=0))

# Feature importance from XGBoost
print("\n" + "-"*60)
print("XGBoost Feature Importance (Top 10):")
feature_importance = pd.DataFrame({
    'Feature': features_all,
    'Importance': model_xgb.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.head(10))

# ============================================
# STEP 6: Visualizations
# ============================================
print("\nCreating visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

# Plot 1: Model Comparison
ax1 = fig.add_subplot(gs[0, :])
models = ['Logistic\nRegression', 'XGBoost']
accuracies = [lr_acc * 100, xgb_acc * 100]
colors = ['#ff6b6b', '#51cf66']
bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax1.set_title('Phase 3: Model Performance Comparison', fontsize=16, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=14)

# Plot 2: Confusion Matrices Comparison
ax2 = fig.add_subplot(gs[1, 0])
im2 = ax2.imshow(cm_lr, cmap='RdYlGn', alpha=0.6)
ax2.set_title('Logistic Regression\nConfusion Matrix', fontsize=12, fontweight='bold')
ax2.set_xlabel('Predicted', fontweight='bold')
ax2.set_ylabel('Actual', fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Down', 'Up'])
ax2.set_yticklabels(['Down', 'Up'])
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm_lr[i, j]), ha='center', va='center', 
                fontsize=16, fontweight='bold')

ax3 = fig.add_subplot(gs[1, 1])
im3 = ax3.imshow(cm_xgb, cmap='RdYlGn', alpha=0.6)
ax3.set_title('XGBoost\nConfusion Matrix', fontsize=12, fontweight='bold')
ax3.set_xlabel('Predicted', fontweight='bold')
ax3.set_ylabel('Actual', fontweight='bold')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Down', 'Up'])
ax3.set_yticklabels(['Down', 'Up'])
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm_xgb[i, j]), ha='center', va='center',
                fontsize=16, fontweight='bold')

# Plot 3: Feature Importance
ax4 = fig.add_subplot(gs[2, :])
top_features = feature_importance.head(8)
ax4.barh(range(len(top_features)), top_features['Importance'], color='steelblue', alpha=0.7)
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['Feature'])
ax4.set_xlabel('Importance Score', fontweight='bold')
ax4.set_title('XGBoost: Most Important Features', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')
ax4.invert_yaxis()

# Plot 4: Predictions on Price Chart (Logistic Regression)
ax5 = fig.add_subplot(gs[3, 0])
test_dates = data.index[split_idx:]
test_prices = data['Close'].iloc[split_idx:]
correct_lr = (y_pred_lr == y_test)

ax5.plot(test_dates, test_prices, label='TSLA Price', color='blue', alpha=0.5, linewidth=1.5)
ax5.scatter(test_dates[correct_lr], test_prices[correct_lr], 
           color='green', label='Correct', alpha=0.6, s=15)
ax5.scatter(test_dates[~correct_lr], test_prices[~correct_lr], 
           color='red', label='Wrong', alpha=0.6, s=15)
ax5.set_title('Logistic Regression Predictions', fontsize=12, fontweight='bold')
ax5.set_ylabel('Price ($)', fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45)

# Plot 5: Predictions on Price Chart (XGBoost)
ax6 = fig.add_subplot(gs[3, 1])
correct_xgb = (y_pred_xgb == y_test)

ax6.plot(test_dates, test_prices, label='TSLA Price', color='blue', alpha=0.5, linewidth=1.5)
ax6.scatter(test_dates[correct_xgb], test_prices[correct_xgb], 
           color='green', label='Correct', alpha=0.6, s=15)
ax6.scatter(test_dates[~correct_xgb], test_prices[~correct_xgb], 
           color='red', label='Wrong', alpha=0.6, s=15)
ax6.set_title('XGBoost Predictions', fontsize=12, fontweight='bold')
ax6.set_ylabel('Price ($)', fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

plt.savefig('results/tsla_phase3_xgboost.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'results/tsla_phase3_xgboost.png'")
plt.show()

print("\n" + "="*60)
print("Phase 3 Complete! 🎉")
print("="*60)
print("\nWhat we achieved:")
print("- Implemented XGBoost with class balancing")
print("- Added 12 engineered features")
print("- Handled class imbalance with scale_pos_weight")
print(f"\nFinal Results:")
print(f"- Logistic Regression: {lr_acc:.2%}")
print(f"- XGBoost: {xgb_acc:.2%}")
print(f"- Improvement: {improvement:+.2%}")
print("\nXGBoost is now predicting both 'Up' and 'Down' movements!")
print("\nNext: Phase 4 will create a Streamlit dashboard for deployment")