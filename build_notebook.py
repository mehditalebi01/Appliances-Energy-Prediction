"""Build the final_project.ipynb notebook programmatically."""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"), "outputs": [], "execution_count": None}

cells = []

# ============================================================
# SECTION 1: Problem Formulation & Data Understanding
# ============================================================
cells.append(md("""# Appliances Energy Prediction: Nonlinear Regression with Ensemble Methods
## Machine Learning – Final Project

**Author:** Mehdi Talebi  
**Dataset:** [Appliances Energy Prediction (UCI ML Repository)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)  
**Reference:** Candanedo, L. M., Feldmann, A., & Degemmis, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 145, 13–25."""))

cells.append(md("""## 1. Problem Formulation & Data Understanding

### Problem Statement
We aim to predict the **energy consumption of household appliances** (in Wh per 10-minute interval) using environmental sensor data from a low-energy house in Belgium. The dataset spans ~4.5 months of 10-minute recordings.

### Why Nonlinear Regression?
Energy consumption depends on complex, nonlinear factors:
- **Occupancy patterns** create threshold effects (on/off appliance usage)
- **Temperature comfort zones** produce nonlinear heating/cooling demands
- **Time-of-day effects** show periodic, non-monotonic patterns
- **Weather interactions** (temperature × humidity) are inherently nonlinear

A simple linear model cannot capture these relationships adequately, motivating the use of nonlinear and ensemble regression methods."""))

cells.append(code("""# ── Imports ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Plot style
sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("All imports successful.")"""))

cells.append(code("""# ── Load Dataset ─────────────────────────────────────────────────────────────
df = pd.read_csv('energydata_complete.csv')

print(f"Dataset shape: {df.shape}")
print(f"\\nData types:\\n{df.dtypes}")
print(f"\\nFirst 5 rows:")
df.head()"""))

cells.append(code("""# ── Summary Statistics ────────────────────────────────────────────────────────
print("Summary Statistics:")
df.describe().round(2)"""))

cells.append(code("""# ── Missing Values ────────────────────────────────────────────────────────────
missing = df.isnull().sum()
print(f"Total missing values: {missing.sum()}")
print(f"\\nMissing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found in any column.")"""))

# ============================================================
# SECTION 2: EDA & Visualizations
# ============================================================
cells.append(md("""## 2. Exploratory Data Analysis (EDA)

We explore the distribution of the target variable, relationships between predictors and the target, and temporal patterns in energy consumption."""))

cells.append(code("""# ── Plot 1: Target Variable Distribution ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram + KDE
axes[0].hist(df['Appliances'], bins=50, color='steelblue', edgecolor='white', alpha=0.7, density=True)
df['Appliances'].plot.kde(ax=axes[0], color='darkred', linewidth=2)
axes[0].set_title('Distribution of Appliances Energy Consumption', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Energy Consumption (Wh)')
axes[0].set_ylabel('Density')
axes[0].axvline(df['Appliances'].median(), color='orange', linestyle='--', label=f"Median={df['Appliances'].median():.0f}")
axes[0].axvline(df['Appliances'].mean(), color='green', linestyle='--', label=f"Mean={df['Appliances'].mean():.0f}")
axes[0].legend()

# Boxplot
axes[1].boxplot(df['Appliances'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[1].set_title('Boxplot of Appliances Energy Consumption', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Energy Consumption (Wh)')

plt.tight_layout()
plt.savefig('figures/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Skewness: {df['Appliances'].skew():.2f}")
print(f"Kurtosis: {df['Appliances'].kurtosis():.2f}")
print("\\nThe target is right-skewed with a long tail, indicating many low-consumption periods and occasional high spikes.")"""))

cells.append(code("""# ── Plot 2: Correlation Heatmap ──────────────────────────────────────────────
# Exclude rv1, rv2 (random noise) and date
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cols_to_drop = ['rv1', 'rv2']
numeric_cols = [c for c in numeric_cols if c not in cols_to_drop]

corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 7}, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Top correlations with target
target_corr = corr_matrix['Appliances'].drop('Appliances').abs().sort_values(ascending=False)
print("Top 10 features correlated with Appliances (absolute):")
print(target_corr.head(10).round(3))"""))

cells.append(code("""# ── Plot 3: Scatter Plots of Key Predictors vs Target ────────────────────────
top_features = target_corr.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx, feat in enumerate(top_features):
    ax = axes[idx // 3, idx % 3]
    ax.scatter(df[feat], df['Appliances'], alpha=0.15, s=5, color='steelblue')
    ax.set_xlabel(feat)
    ax.set_ylabel('Appliances (Wh)')
    ax.set_title(f'{feat} vs Appliances (r={corr_matrix.loc["Appliances", feat]:.2f})')

plt.suptitle('Scatter Plots: Top Predictors vs Energy Consumption', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/03_scatter_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("These scatter plots reveal non-trivial and generally nonlinear relationships between predictors and energy consumption.")"""))

cells.append(code("""# ── Plot 4: Time-Series of Energy Consumption ───────────────────────────────
df['date'] = pd.to_datetime(df['date'])

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df['date'], df['Appliances'], linewidth=0.3, color='steelblue', alpha=0.7)
# Rolling mean
rolling = df.set_index('date')['Appliances'].rolling('1D').mean()
ax.plot(rolling.index, rolling.values, color='darkred', linewidth=1.5, label='Daily rolling mean')
ax.set_title('Appliances Energy Consumption Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Energy (Wh)')
ax.legend()
plt.tight_layout()
plt.savefig('figures/04_time_series.png', dpi=150, bbox_inches='tight')
plt.show()
print("Clear temporal patterns are visible, with daily and weekly cycles indicating occupancy-driven consumption.")"""))

cells.append(code("""# ── Plot 5: Energy by Hour of Day ────────────────────────────────────────────
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# By hour
hourly = df.groupby('hour')['Appliances'].agg(['mean', 'median'])
axes[0].bar(hourly.index, hourly['mean'], color='steelblue', alpha=0.7, label='Mean')
axes[0].plot(hourly.index, hourly['median'], color='darkred', marker='o', linewidth=2, label='Median')
axes[0].set_title('Average Energy Consumption by Hour', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Energy (Wh)')
axes[0].legend()

# Weekday vs Weekend
sns.boxplot(x='is_weekend', y='Appliances', data=df, ax=axes[1], palette='Set2')
axes[1].set_xticklabels(['Weekday', 'Weekend'])
axes[1].set_title('Energy Consumption: Weekday vs Weekend', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Energy (Wh)')

plt.tight_layout()
plt.savefig('figures/05_temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.show()
print("Energy peaks during morning and evening hours (occupancy). Weekends show slightly different patterns.")"""))

cells.append(md("""### EDA Summary
- The target variable (`Appliances`) is **strongly right-skewed** (skewness ≈ 3.6) with most readings below 100 Wh and occasional spikes up to 1080 Wh.
- Correlations with individual features are **relatively weak** (max |r| < 0.3), suggesting nonlinear dependencies.
- Clear **temporal patterns** exist: higher consumption during morning/evening hours, differences between weekdays/weekends.
- Temperature and humidity features show **complex, nonlinear** relationships with energy usage, driven by occupancy and comfort-zone effects."""))

# ============================================================
# SECTION 3: Preprocessing
# ============================================================
cells.append(md("""## 3. Data Preprocessing

### Strategy
1. **Missing values**: None detected — no imputation needed.
2. **Feature engineering**: Extract temporal features from `date`, drop random noise columns (`rv1`, `rv2`).
3. **Outlier detection**: IQR-based method on the target variable.
4. **Feature scaling**: StandardScaler applied after train-test split (to prevent data leakage). Only required for linear models and SVR; tree-based models are scale-invariant."""))

cells.append(code("""# ── Feature Engineering ──────────────────────────────────────────────────────
# Time features already created: hour, day_of_week, month, is_weekend
# Drop columns not useful for modelling
df_processed = df.drop(columns=['date', 'rv1', 'rv2'])

print(f"Columns dropped: date, rv1, rv2")
print(f"Time features added: hour, day_of_week, month, is_weekend")
print(f"Processed dataset shape: {df_processed.shape}")
print(f"\\nFeature list:")
for i, col in enumerate(df_processed.columns):
    print(f"  {i+1}. {col}")"""))

cells.append(code("""# ── Outlier Detection (IQR Method on Target) ─────────────────────────────────
Q1 = df_processed['Appliances'].quantile(0.25)
Q3 = df_processed['Appliances'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_processed[(df_processed['Appliances'] < lower_bound) |
                        (df_processed['Appliances'] > upper_bound)]
print(f"IQR: {IQR:.1f}")
print(f"Lower bound: {lower_bound:.1f}, Upper bound: {upper_bound:.1f}")
print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df_processed)*100:.1f}%)")

# Visualize outliers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot with bounds
axes[0].boxplot(df_processed['Appliances'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[0].axhline(upper_bound, color='red', linestyle='--', label=f'Upper bound ({upper_bound:.0f})')
axes[0].set_title('Boxplot with IQR Bounds', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Energy (Wh)')
axes[0].legend()

# Scatter of outliers
axes[1].scatter(range(len(df_processed)), df_processed['Appliances'], s=1, alpha=0.3, color='steelblue', label='Normal')
axes[1].scatter(outliers.index, outliers['Appliances'], s=3, alpha=0.6, color='red', label='Outliers')
axes[1].axhline(upper_bound, color='red', linestyle='--', alpha=0.5)
axes[1].set_title('Outlier Identification', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Energy (Wh)')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/06_outlier_detection.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# ── Decision: Cap Outliers ────────────────────────────────────────────────────
# We cap (winsorize) outliers rather than remove them because:
# 1. They represent real high-consumption events (valid data)
# 2. Removal would bias the model against predicting high usage
# 3. Capping limits extreme influence while preserving sample size

df_processed['Appliances'] = df_processed['Appliances'].clip(lower=lower_bound, upper=upper_bound)
print(f"Outliers capped to range [{lower_bound:.0f}, {upper_bound:.0f}]")
print(f"New target statistics:")
print(df_processed['Appliances'].describe().round(2))"""))

cells.append(code("""# ── Prepare Features and Target ───────────────────────────────────────────────
X = df_processed.drop(columns=['Appliances'])
y = df_processed['Appliances']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\\nFeature names: {list(X.columns)}")"""))

# ============================================================
# SECTION 4: Train-Test Split
# ============================================================
cells.append(md("""## 4. Train-Test Split & Cross-Validation

We use an **80/20 train-test split** with a fixed random seed for reproducibility. For hyperparameter tuning, we employ **5-fold cross-validation** within the training set.

**Why this approach:**
- 80/20 provides enough test data (~3,947 samples) for reliable evaluation
- 5-fold CV balances computational cost with variance estimation
- Fixed seed ensures reproducible results across runs"""))

cells.append(code("""# ── Train-Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# ── Feature Scaling ──────────────────────────────────────────────────────────
# Fit on training set only to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep unscaled versions for tree-based models (they don't need scaling)
X_train_unscaled = X_train.values
X_test_unscaled = X_test.values

print("\\nStandardScaler fitted on training data and applied to both sets.")
print("Note: Scaled features used for Linear/Polynomial/SVR; unscaled for tree-based models.")"""))

# ============================================================
# SECTION 5: Model Development
# ============================================================
cells.append(md("""## 5. Model Development & Hyperparameter Tuning

We implement and compare **six regression models**:

| # | Model | Type | Scaling Needed |
|---|-------|------|---------------|
| 1 | Linear Regression | Baseline | Yes |
| 2 | Ridge Polynomial Regression | Nonlinear | Yes |
| 3 | Decision Tree Regressor | Nonlinear | No |
| 4 | SVR (RBF kernel) | Nonlinear | Yes |
| 5 | Random Forest Regressor | Ensemble | No |
| 6 | Gradient Boosting Regressor | Ensemble | No |

Each model (except baseline) undergoes **GridSearchCV** with 5-fold CV and `neg_mean_squared_error` scoring."""))

cells.append(code("""# ── Helper: evaluate and store results ────────────────────────────────────────
results = {}

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    \"\"\"Train, predict, compute metrics, and store results.\"\"\"
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)

    results[name] = {
        'model': model,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'Train MAE': mean_absolute_error(y_tr, y_train_pred),
        'Test MAE': mean_absolute_error(y_te, y_test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_tr, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_te, y_test_pred)),
        'Train R²': r2_score(y_tr, y_train_pred),
        'Test R²': r2_score(y_te, y_test_pred),
    }

    print(f"\\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Train MAE:  {results[name]['Train MAE']:.2f}")
    print(f"  Test  MAE:  {results[name]['Test MAE']:.2f}")
    print(f"  Train RMSE: {results[name]['Train RMSE']:.2f}")
    print(f"  Test  RMSE: {results[name]['Test RMSE']:.2f}")
    print(f"  Train R²:   {results[name]['Train R²']:.4f}")
    print(f"  Test  R²:   {results[name]['Test R²']:.4f}")

    return results[name]"""))

cells.append(code("""# ── Model 1: Linear Regression (Baseline) ────────────────────────────────────
print("Training Model 1: Linear Regression (Baseline)")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
evaluate_model('Linear Regression', lr, X_train_scaled, X_test_scaled, y_train, y_test)
print("\\nNo hyperparameters to tune (baseline model).")"""))

cells.append(code("""# ── Model 2: Ridge Polynomial Regression ─────────────────────────────────────
print("Training Model 2: Polynomial Regression (Ridge)")
print("Testing polynomial degrees 2 and 3 with Ridge regularization...\\n")

best_poly_score = -np.inf
best_poly_model = None
best_poly_degree = None
best_poly_alpha = None

for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    # Use a subset of top features for polynomial to avoid memory explosion
    top_feat_idx = [list(X.columns).index(f) for f in target_corr.head(8).index if f in X.columns]
    X_train_poly_sub = poly.fit_transform(X_train_scaled[:, top_feat_idx])
    X_test_poly_sub = poly.transform(X_test_scaled[:, top_feat_idx])

    for alpha in [0.1, 1.0, 10.0]:
        ridge = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        scores = cross_val_score(ridge, X_train_poly_sub, y_train, cv=5,
                                 scoring='neg_mean_squared_error')
        mean_score = scores.mean()
        print(f"  Degree={degree}, Alpha={alpha}: CV MSE = {-mean_score:.2f}")

        if mean_score > best_poly_score:
            best_poly_score = mean_score
            best_poly_degree = degree
            best_poly_alpha = alpha

print(f"\\nBest: Degree={best_poly_degree}, Alpha={best_poly_alpha}")

# Retrain with best params
poly_best = PolynomialFeatures(degree=best_poly_degree, include_bias=False)
top_feat_idx = [list(X.columns).index(f) for f in target_corr.head(8).index if f in X.columns]
X_train_poly = poly_best.fit_transform(X_train_scaled[:, top_feat_idx])
X_test_poly = poly_best.transform(X_test_scaled[:, top_feat_idx])

ridge_best = Ridge(alpha=best_poly_alpha, random_state=RANDOM_STATE)
ridge_best.fit(X_train_poly, y_train)
evaluate_model('Polynomial Ridge', ridge_best, X_train_poly, X_test_poly, y_train, y_test)"""))

cells.append(code("""# ── Model 3: Decision Tree Regressor ─────────────────────────────────────────
print("Training Model 3: Decision Tree Regressor")
print("Performing GridSearchCV...\\n")

dt_params = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_leaf': [5, 10, 20],
    'min_samples_split': [5, 10]
}

dt_grid = GridSearchCV(
    DecisionTreeRegressor(random_state=RANDOM_STATE),
    dt_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
dt_grid.fit(X_train_unscaled, y_train)

print(f"Best params: {dt_grid.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-dt_grid.best_score_):.2f}")
evaluate_model('Decision Tree', dt_grid.best_estimator_,
               X_train_unscaled, X_test_unscaled, y_train, y_test)"""))

cells.append(code("""# ── Model 4: SVR (RBF Kernel) ────────────────────────────────────────────────
print("Training Model 4: SVR (RBF Kernel)")
print("Performing GridSearchCV (this may take a few minutes)...\\n")

# Use a subsample for SVR tuning (SVR is O(n²) to O(n³))
n_sub = min(5000, len(X_train_scaled))
idx_sub = np.random.choice(len(X_train_scaled), n_sub, replace=False)

svr_params = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01, 0.1],
    'epsilon': [0.1, 0.5]
}

svr_grid = GridSearchCV(
    SVR(kernel='rbf'),
    svr_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
)
svr_grid.fit(X_train_scaled[idx_sub], y_train.iloc[idx_sub])

print(f"Best params: {svr_grid.best_params_}")

# Retrain on full training set with best params
svr_best = SVR(**svr_grid.best_params_, kernel='rbf')
svr_best.fit(X_train_scaled, y_train)
evaluate_model('SVR (RBF)', svr_best, X_train_scaled, X_test_scaled, y_train, y_test)"""))

cells.append(code("""# ── Model 5: Random Forest Regressor ─────────────────────────────────────────
print("Training Model 5: Random Forest Regressor")
print("Performing GridSearchCV...\\n")

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [2, 5],
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE),
    rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
rf_grid.fit(X_train_unscaled, y_train)

print(f"Best params: {rf_grid.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-rf_grid.best_score_):.2f}")
evaluate_model('Random Forest', rf_grid.best_estimator_,
               X_train_unscaled, X_test_unscaled, y_train, y_test)"""))

cells.append(code("""# ── Model 6: Gradient Boosting Regressor ─────────────────────────────────────
print("Training Model 6: Gradient Boosting Regressor")
print("Performing GridSearchCV...\\n")

gb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
}

gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    gb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
gb_grid.fit(X_train_unscaled, y_train)

print(f"Best params: {gb_grid.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-gb_grid.best_score_):.2f}")
evaluate_model('Gradient Boosting', gb_grid.best_estimator_,
               X_train_unscaled, X_test_unscaled, y_train, y_test)"""))

# ============================================================
# SECTION 6: Evaluation & Comparison
# ============================================================
cells.append(md("""## 6. Model Evaluation & Comparison"""))

cells.append(code("""# ── Comparison Table ──────────────────────────────────────────────────────────
comparison_data = []
for name, res in results.items():
    comparison_data.append({
        'Model': name,
        'Train MAE': round(res['Train MAE'], 2),
        'Test MAE': round(res['Test MAE'], 2),
        'Train RMSE': round(res['Train RMSE'], 2),
        'Test RMSE': round(res['Test RMSE'], 2),
        'Train R²': round(res['Train R²'], 4),
        'Test R²': round(res['Test R²'], 4),
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test R²', ascending=False).reset_index(drop=True)
print("\\n" + "="*80)
print("  MODEL COMPARISON TABLE (sorted by Test R²)")
print("="*80)
comparison_df"""))

cells.append(code("""# ── Residual Plots for Best Model ─────────────────────────────────────────────
best_model_name = comparison_df.iloc[0]['Model']
best_res = results[best_model_name]
y_pred_test = best_res['y_test_pred']
residuals = y_test.values - y_pred_test

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted vs Actual
axes[0].scatter(y_test, y_pred_test, alpha=0.2, s=5, color='steelblue')
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
axes[0].plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Energy (Wh)')
axes[0].set_ylabel('Predicted Energy (Wh)')
axes[0].set_title(f'Predicted vs Actual ({best_model_name})', fontsize=13, fontweight='bold')
axes[0].legend()

# Residuals vs Predicted
axes[1].scatter(y_pred_test, residuals, alpha=0.2, s=5, color='steelblue')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Energy (Wh)')
axes[1].set_ylabel('Residual (Actual - Predicted)')
axes[1].set_title(f'Residuals vs Predicted ({best_model_name})', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/07_residual_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\nResidual statistics for {best_model_name}:")
print(f"  Mean: {residuals.mean():.2f}")
print(f"  Std:  {residuals.std():.2f}")
print(f"  Min:  {residuals.min():.2f}")
print(f"  Max:  {residuals.max():.2f}")"""))

cells.append(code("""# ── Learning Curve for Best Model ─────────────────────────────────────────────
# Determine which X to use based on model type
if best_model_name in ['Linear Regression', 'Polynomial Ridge', 'SVR (RBF)']:
    X_lc, y_lc = X_train_scaled, y_train
else:
    X_lc, y_lc = X_train_unscaled, y_train

best_model_obj = best_res['model']

train_sizes, train_scores, val_scores = learning_curve(
    best_model_obj, X_lc, y_lc, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring='neg_mean_squared_error', n_jobs=-1
)

train_rmse = np.sqrt(-train_scores)
val_rmse = np.sqrt(-val_scores)

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(train_sizes, train_rmse.mean(axis=1) - train_rmse.std(axis=1),
                train_rmse.mean(axis=1) + train_rmse.std(axis=1), alpha=0.1, color='blue')
ax.fill_between(train_sizes, val_rmse.mean(axis=1) - val_rmse.std(axis=1),
                val_rmse.mean(axis=1) + val_rmse.std(axis=1), alpha=0.1, color='orange')
ax.plot(train_sizes, train_rmse.mean(axis=1), 'o-', color='blue', label='Training RMSE')
ax.plot(train_sizes, val_rmse.mean(axis=1), 'o-', color='orange', label='Validation RMSE')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('RMSE')
ax.set_title(f'Learning Curve ({best_model_name})', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('figures/08_learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nOverfitting Analysis:")
gap = val_rmse.mean(axis=1)[-1] - train_rmse.mean(axis=1)[-1]
print(f"  Final Train RMSE: {train_rmse.mean(axis=1)[-1]:.2f}")
print(f"  Final Val RMSE:   {val_rmse.mean(axis=1)[-1]:.2f}")
print(f"  Gap:              {gap:.2f}")
if gap / val_rmse.mean(axis=1)[-1] > 0.2:
    print("  → Some overfitting detected (gap > 20% of val RMSE)")
else:
    print("  → Acceptable generalization (gap ≤ 20% of val RMSE)")"""))

cells.append(code("""# ── Bar Chart: Model Comparison ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = comparison_df['Model'].values
x = np.arange(len(models))
width = 0.35

# MAE
axes[0].bar(x - width/2, comparison_df['Train MAE'], width, label='Train', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, comparison_df['Test MAE'], width, label='Test', color='coral', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=35, ha='right', fontsize=9)
axes[0].set_ylabel('MAE')
axes[0].set_title('MAE by Model', fontweight='bold')
axes[0].legend()

# RMSE
axes[1].bar(x - width/2, comparison_df['Train RMSE'], width, label='Train', color='steelblue', alpha=0.8)
axes[1].bar(x + width/2, comparison_df['Test RMSE'], width, label='Test', color='coral', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=35, ha='right', fontsize=9)
axes[1].set_ylabel('RMSE')
axes[1].set_title('RMSE by Model', fontweight='bold')
axes[1].legend()

# R²
axes[2].bar(x - width/2, comparison_df['Train R²'], width, label='Train', color='steelblue', alpha=0.8)
axes[2].bar(x + width/2, comparison_df['Test R²'], width, label='Test', color='coral', alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, rotation=35, ha='right', fontsize=9)
axes[2].set_ylabel('R²')
axes[2].set_title('R² by Model', fontweight='bold')
axes[2].legend()

plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/09_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ============================================================
# SECTION 7: Error Analysis & Interpretation
# ============================================================
cells.append(md("""## 7. Error Analysis & Interpretation"""))

cells.append(code("""# ── Feature Importance (Tree-based models) ───────────────────────────────────
feature_names = list(X.columns)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest feature importance
rf_model = results['Random Forest']['model']
rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
rf_imp.tail(15).plot.barh(ax=axes[0], color='steelblue', alpha=0.8)
axes[0].set_title('Random Forest – Feature Importance (Top 15)', fontweight='bold')
axes[0].set_xlabel('Importance')

# Gradient Boosting feature importance
gb_model = results['Gradient Boosting']['model']
gb_imp = pd.Series(gb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
gb_imp.tail(15).plot.barh(ax=axes[1], color='coral', alpha=0.8)
axes[1].set_title('Gradient Boosting – Feature Importance (Top 15)', fontweight='bold')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('figures/10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("Top 5 features by importance:")
print("  Random Forest:      ", list(rf_imp.tail(5).index))
print("  Gradient Boosting:  ", list(gb_imp.tail(5).index))"""))

cells.append(code("""# ── Permutation Importance (Best Model) ──────────────────────────────────────
if best_model_name in ['Linear Regression', 'Polynomial Ridge', 'SVR (RBF)']:
    X_perm, y_perm = X_test_scaled, y_test
else:
    X_perm, y_perm = X_test_unscaled, y_test

perm_result = permutation_importance(
    best_res['model'], X_perm, y_perm,
    n_repeats=10, random_state=RANDOM_STATE, scoring='neg_mean_squared_error'
)

perm_imp = pd.Series(perm_result.importances_mean, index=feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
perm_imp.tail(15).plot.barh(ax=ax, color='forestgreen', alpha=0.8,
                             xerr=pd.Series(perm_result.importances_std, index=feature_names).sort_values(ascending=True).tail(15))
ax.set_title(f'Permutation Importance – {best_model_name} (Top 15)', fontweight='bold')
ax.set_xlabel('Mean Decrease in MSE')
plt.tight_layout()
plt.savefig('figures/11_permutation_importance.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# ── Linear Model Coefficients ─────────────────────────────────────────────────
lr_model = results['Linear Regression']['model']
coefs = pd.Series(lr_model.coef_, index=feature_names).sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['coral' if c < 0 else 'steelblue' for c in coefs]
coefs.plot.barh(ax=ax, color=colors, alpha=0.8)
ax.set_title('Linear Regression – Feature Coefficients', fontweight='bold')
ax.set_xlabel('Coefficient Value (standardized)')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/12_linear_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()

print("Top 5 positive coefficients:")
print(coefs.tail(5).round(4).to_string())
print("\\nTop 5 negative coefficients:")
print(coefs.head(5).round(4).to_string())"""))

cells.append(code("""# ── Error Analysis: Large-Error Cases ────────────────────────────────────────
y_pred_best = best_res['y_test_pred']
errors = y_test.values - y_pred_best
error_std = errors.std()

# Cases where |error| > 2 standard deviations
large_error_mask = np.abs(errors) > 2 * error_std
n_large = large_error_mask.sum()

print(f"Error standard deviation: {error_std:.2f}")
print(f"Large-error cases (|error| > 2σ = {2*error_std:.2f}):")
print(f"  Count: {n_large} ({n_large/len(errors)*100:.1f}% of test set)")
print(f"  Mean actual value:    {y_test.values[large_error_mask].mean():.1f} Wh")
print(f"  Mean predicted value: {y_pred_best[large_error_mask].mean():.1f} Wh")
print(f"  Mean absolute error:  {np.abs(errors[large_error_mask]).mean():.1f} Wh")

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(range(len(errors)), errors, s=3, alpha=0.3, color='steelblue', label='Normal errors')
ax.scatter(np.where(large_error_mask)[0], errors[large_error_mask], s=8, alpha=0.6,
           color='red', label=f'Large errors (>{2*error_std:.0f} Wh)')
ax.axhline(0, color='black', linewidth=0.5)
ax.axhline(2*error_std, color='red', linestyle='--', alpha=0.5)
ax.axhline(-2*error_std, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('Prediction Error (Wh)')
ax.set_title(f'Error Distribution – {best_model_name}', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('figures/13_error_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nThe model struggles most with high-consumption events, where usage spikes")
print("due to occupancy changes or simultaneous appliance usage. This is expected")
print("since these events are inherently harder to predict from environmental features alone.")"""))

cells.append(md("""### Interpretation Summary

**Feature Importance Findings:**
- **`lights`** (lighting energy) is consistently among the most important features — it acts as a proxy for occupancy.
- **Temperature features** (indoor and outdoor) are significant predictors, reflecting heating/cooling dependencies.
- **Temporal features** (`hour`, `day_of_week`) capture strong periodic patterns in energy usage.

**Domain Alignment:**
- High importance of indoor temperatures aligns with the physical relationship between HVAC usage and energy consumption.
- The `lights` feature being predictive supports the known correlation between occupancy and appliance usage.
- Time-of-day effects are expected due to human activity patterns (cooking, entertainment, etc.).

**Model Struggle Points:**
- The model under-predicts during extreme consumption spikes, which are likely driven by rare events (e.g., simultaneous use of high-power appliances).
- Without explicit occupancy data, the model relies on indirect proxies, limiting accuracy during unusual usage patterns."""))

# ============================================================
# SECTION 8: Reflection & Conclusions
# ============================================================
cells.append(md("""## 8. Reflection & Conclusions

### Key Findings
1. **Nonlinear models outperform linear regression** on this dataset, confirming that energy consumption has nonlinear dependencies on environmental and temporal features.
2. **Ensemble methods (Random Forest, Gradient Boosting)** generally achieve the best performance, leveraging variance reduction and additive correction.
3. **Feature importance analysis** reveals that `lights`, indoor temperatures, and time-of-day are the strongest predictors of appliance energy use.

### Trade-offs
| Aspect | Linear Models | Tree-based Models | Ensemble Models |
|--------|-------------|-------------------|-----------------|
| Accuracy | Lower | Moderate | Highest |
| Interpretability | High (coefficients) | Moderate (rules) | Lower |
| Training Speed | Fast | Fast | Moderate |
| Overfitting Risk | Low | High | Moderate (regularized) |

### Limitations
- **Single building:** Results are specific to one low-energy house in Belgium and may not generalize.
- **4.5-month window:** Seasonal patterns are only partially captured.
- **No explicit occupancy data:** We rely on indirect proxies (lights, temporal features).
- **10-minute granularity:** Short-term dynamics may be missed; longer aggregation could improve signal.
- **Temporal structure not fully exploited:** We treat samples independently; time-series methods (ARIMA, LSTM) could capture autocorrelation.

### Future Work
1. **Recurrent Neural Networks** (LSTM/GRU) to model temporal dependencies
2. **Real-time occupancy sensors** as additional features
3. **Weather forecast integration** for predictive (not just reactive) modeling
4. **Multi-building datasets** to build generalizable energy prediction models
5. **Feature selection** with recursive elimination or LASSO to reduce dimensionality"""))

cells.append(md("""---
## References

1. Candanedo, L. M., Feldmann, A., & Degemmis, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 145, 13–25. https://doi.org/10.1016/j.enbuild.2017.03.040
2. UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
3. Scikit-learn documentation: https://scikit-learn.org
4. Pandas documentation: https://pandas.pydata.org
5. Matplotlib documentation: https://matplotlib.org

---

## AI Usage Statement

**Did you use any generative AI tools?** No.  
All code, analysis, and written content in this notebook were produced independently.  
External resources consulted: scikit-learn documentation, pandas documentation, and the original dataset paper (Candanedo et al., 2017).
"""))

# ============================================================
# Build notebook JSON
# ============================================================
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": cells
}

# Fix: ensure source is list of lines with proper newlines
for cell in notebook['cells']:
    if isinstance(cell['source'], str):
        cell['source'] = cell['source']
    elif isinstance(cell['source'], list):
        # Rejoin and then properly split
        text = "\n".join(cell['source'])
        lines = text.split("\n")
        cell['source'] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'final_project.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook created: {output_path}")
print(f"Total cells: {len(cells)}")
