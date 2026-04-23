import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# =========================
# Load Data
# =========================
train_df = pd.read_csv("train_data_150k.csv")
test_df = pd.read_csv("test_data_60k.csv")

# Combine temporarily for consistent label encoding to avoid unseen labels in test set
train_df['is_train'] = True
test_df['is_train'] = False
df = pd.concat([train_df, test_df], ignore_index=True)

# =========================
# Feature Engineering & Label Encoding
# =========================
le = LabelEncoder()
if 'season' in df.columns:
    df['season_enc'] = le.fit_transform(df['season'].astype(str))

# =========================
# Features & Target
# =========================
features = ['season_enc', 'temperature', 'rainfall', 'humidity', 'soil_moisture', 'demand_index']
# Ensure we only use columns that exist
features = [f for f in features if f in df.columns]

# Split back into train and test
train_data = df[df['is_train'] == True]
test_data = df[df['is_train'] == False]

X_train = train_data[features]
y_train = train_data['price']

X_test = test_data[features]
y_test = test_data['price']

print(f"Total features : {len(features)}")
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# =========================
# Train Model
# =========================
print("\n⏳ Training XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)
print("✅ Training complete!")

# =========================
# Model Evaluation (Train & Test)
# =========================
# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Training Metrics
train_mae  = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2   = r2_score(y_train, y_train_pred)

# Testing Metrics
test_mae  = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2   = r2_score(y_test, y_test_pred)

print("\n" + "="*50)
print("       MODEL EVALUATION RESULTS (FOR PPT)")
print("="*50)
print(" 📊 TRAINING DATA METRICS:")
print(f"  - R²   (Accuracy/Variation)   :  {train_r2:.4f}")
print(f"  - MAE  (Mean Absolute Error)  : ₹{train_mae:.2f}")
print(f"  - RMSE (Root Mean Sq. Error)  : ₹{train_rmse:.2f}")
print("-" * 50)
print(" 🎯 TESTING DATA METRICS (Unseen Data):")
print(f"  - R²   (Accuracy/Variation)   :  {test_r2:.4f}")
print(f"  - MAE  (Mean Absolute Error)  : ₹{test_mae:.2f}")
print(f"  - RMSE (Root Mean Sq. Error)  : ₹{test_rmse:.2f}")
print("="*50)

# =========================
# What these mean
# =========================
print(f"\n📊 Interpretation (on unseen test data):")
print(f"  - On average, prediction is off by ₹{test_mae:.0f}")
print(f"  - Model explains {test_r2*100:.1f}% of price variation")
print(f"  - RMSE penalizes large errors more: ₹{test_rmse:.0f}")

# =========================
# Actual vs Predicted Sample
# =========================
print("\n📋 Sample Predictions (first 5 test rows):")
print("-"*45)
print(f"{'Actual':>12}  {'Predicted':>12}  {'Error':>10}")
print("-"*45)
for actual, pred in zip(list(y_test[:5]), y_test_pred[:5]):
    error = abs(actual - pred)
    print(f"₹{actual:>10.0f}  ₹{pred:>10.0f}  ₹{error:>8.0f}")
print("-"*45)

# =========================
# Graphical Analysis
# =========================
feat_df = pd.DataFrame({
    'Feature':    features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

# Create a figure with 1 row and 3 columns for 3 graphs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Feature Importance Chart
axes[0].barh(feat_df['Feature'], feat_df['Importance'], color='#4CAF50')
axes[0].set_xlabel('Importance Score')
axes[0].set_title('Feature Importance')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# 2. Actual vs Predicted Scatter Plot
# Sample a smaller subset so it doesn't take forever to plot
plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred}).sample(min(1000, len(y_test)), random_state=42)
axes[1].scatter(plot_df['Actual'], plot_df['Predicted'], alpha=0.5, color='#2196F3')
axes[1].plot([plot_df['Actual'].min(), plot_df['Actual'].max()], 
             [plot_df['Actual'].min(), plot_df['Actual'].max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title('Actual vs Predicted Prices')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# 3. Residual Distribution
residuals = y_test - y_test_pred
axes[2].hist(residuals, bins=50, color='#FF9800', edgecolor='k', alpha=0.7)
axes[2].set_xlabel('Prediction Error (Residuals)')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Error Distribution')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('model_analysis_graphs.png', dpi=150)
print("\n📊 Saved 'model_analysis_graphs.png' with Feature Importance, Actual vs Expected, and Residuals charts!")
plt.close()
print("✅ Done! Open 'model_analysis_graphs.png' to view the charts.")
