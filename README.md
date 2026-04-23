# Crop Price Forecasting & Market Advisory System

An end-to-end ML pipeline that trains **three regression models** on agricultural data and generates a comprehensive comparison dashboard.

## Models
| Model | Preprocessing |
|-------|--------------|
| Linear Regression | One-hot encoding + StandardScaler |
| Random Forest | LabelEncoder (no scaling needed) |
| XGBoost | LabelEncoder (no scaling needed) |

## Dataset
| File | Rows | Columns |
|------|------|---------|
| `train_data_150k.csv` | 150,000 | season, temperature, rainfall, humidity, soil_moisture, demand_index, price |
| `test_data_60k.csv`   | 60,000  | same |

## How to Run

```bash
# 1. Install dependencies
pip install scikit-learn xgboost pandas numpy matplotlib joblib

# 2. Run the full pipeline
python main.py
```

## Outputs
| File | Description |
|------|-------------|
| `final_model_comparison.png` | Master dashboard (metrics + scatter plots + summary table) |
| `lr_train_sorted.png` / `lr_test_sorted.png` / `lr_scatter.png` | LR plots |
| `rf_train_sorted.png` / `rf_test_sorted.png` / `rf_scatter.png` | RF plots |
| `xgboost_train_sorted.png` / `xgboost_test_sorted.png` / `xgboost_scatter.png` | XGBoost plots |
| `rf_feature_importance.png` / `xgb_feature_importance.png` | Feature importance charts |
| `lr_model.pkl`, `lr_scaler.pkl` | Saved LR artefacts |
| `rf_model.pkl` | Saved RF model |
| `xgb_model.json` | Saved XGBoost model |

## Advisory Logic
| Predicted change vs Actual | Signal |
|----------------------------|--------|
| ≥ +10 % | **HOLD** |
| +2 % – +10 % | **WAIT** |
| < +2 % | **SELL** |

## Metrics Reported
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Both **training** and **testing** metrics are reported for every model.