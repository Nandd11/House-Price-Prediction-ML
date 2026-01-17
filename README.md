# House Price Prediction (Linear Regression)

## Problem Statement
HomeVista Properties wants to automate house pricing using Machine Learning. The goal is to predict the **SalePrice** of houses based on property features such as zoning, lot configuration, building type, year built, basement area, and overall condition.

## Dataset
- **Target column:** `SalePrice`
- **Type:** Supervised regression problem
- **Preprocessing done:** missing value handling, removal of irrelevant columns, one-hot encoding

## Approach / Workflow
1. Load and explore dataset
2. Data cleaning
   - Drop `Id`
   - Fill missing `SalePrice` with mean
   - Drop rows with remaining null values
3. Feature encoding
   - One-hot encoding for categorical columns (e.g., `MSZoning`, `LotConfig`, `BldgType`, `Exterior1st`)
4. Train-test split (80/20)
5. Train baseline **Linear Regression** model
6. Evaluate with MAE, RMSE, and R²
7. Visualize predictions and residuals

## Model Used
- Linear Regression (`sklearn.linear_model.LinearRegression`)

## Results (Test Set)
- **R² Score:** `0.3741`
- **MAE:** `30829.94`
- **RMSE:** `41138.56`

> Note: This is a baseline model. The goal is to show correct ML workflow and evaluation.

## Visualizations
- Correlation heatmap (top correlated features)
- Actual vs Predicted plot
- Residual distribution

All plots are available in the [`images/`](./images) folder.

## How to Run
### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run notebook
```bash
jupyter notebook
```

## Repository Structure
```text
House-Price-Prediction-ML/
├── house_price_predictor.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── HousePricePrediction.csv
└── images/
    ├── correlation_heatmap_top.png
    ├── actual_vs_predicted.png
    ├── residual_distribution.png
    └── metrics.txt
```

## Future Improvements
- Try Ridge/Lasso Regression
- Use Random Forest / XGBoost for better accuracy
- Feature scaling + better feature selection
- Handle missing values without dropping rows
- Hyperparameter tuning and cross-validation
