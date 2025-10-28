import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 1: Download TSLA data (2 years)
print("Downloading TSLA data...")
ticker = "TSLA"
data = yf.download(ticker, period="2y", interval="1d")

# Flatten multi-level columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Step 2: Feature Engineering
print("\nCreating features...")

# Calculate daily returns
data['Returns'] = data['Close'].pct_change()

# 20-day moving average
data['MA_20'] = data['Close'].rolling(window=20).mean()

# Volatility (20-day rolling standard deviation of returns)
data['Volatility'] = data['Returns'].rolling(window=20).std()

# Price relative to moving average
data['Price_to_MA'] = data['Close'] / data['MA_20']

# Target: Next day's direction (1 = up, 0 = down)
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop rows with NaN values
data = data.dropna()

print(f"Dataset shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())

# Step 3: Prepare features and target
features = ['Returns', 'MA_20', 'Volatility', 'Price_to_MA']
X = data[features]
y = data['Target']

print(f"\nClass distribution:")
print(y.value_counts())
print(f"Baseline accuracy (always predict majority class): {y.value_counts().max() / len(y):.2%}")

# Step 4: Train-test split (80/20)
# Important: Don't shuffle for time series!
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Step 5: Train the model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Step 7: Evaluate
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
print("\n  [[True Negatives, False Positives]")
print("   [False Negatives, True Positives]]")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Down', 'Up']))

# Step 8: Feature importance
print("\nFeature Importance (Coefficients):")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)
print(feature_importance)

# Step 9: Visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Price with predictions
test_dates = data.index[split_idx:]
test_prices = data['Close'].iloc[split_idx:]
correct_predictions = (y_pred_test == y_test)

axes[0].plot(test_dates, test_prices, label='TSLA Price', color='blue', alpha=0.7)
axes[0].scatter(test_dates[correct_predictions], test_prices[correct_predictions], 
                color='green', label='Correct Prediction', alpha=0.5, s=10)
axes[0].scatter(test_dates[~correct_predictions], test_prices[~correct_predictions], 
                color='red', label='Wrong Prediction', alpha=0.5, s=10)
axes[0].set_title('TSLA Stock Price with Prediction Results')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Cumulative correct predictions
cumulative_correct = correct_predictions.cumsum()
axes[1].plot(test_dates, cumulative_correct, color='green')
axes[1].set_title('Cumulative Correct Predictions Over Time')
axes[1].set_ylabel('Cumulative Correct')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tsla_phase1_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'tsla_phase1_results.png'")
plt.show()

print("\n" + "="*50)
print("Phase 1 Complete! 🎉")
print("="*50)
print("\nNext steps for Phase 2:")
print("- Add sentiment analysis from financial news")
print("- Compare performance with and without sentiment")