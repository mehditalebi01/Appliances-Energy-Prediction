# 🔌 Appliances Energy Prediction

**Nonlinear Regression with Ensemble Methods** — Machine Learning Final Project

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Summary

This project develops a complete supervised ML pipeline for predicting household appliance energy consumption (Wh) using environmental sensor data from a low-energy house in Belgium. We compare **6 regression models** — from linear baselines to ensemble methods — demonstrating that nonlinear approaches significantly outperform linear models on this task.

**Best Model:** Random Forest Regressor — **R² = 0.734, MAE = 14.66 Wh**

### Dataset
- **Source:** [UCI ML Repository – Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Size:** 19,735 observations × 29 features (10-minute intervals, ~4.5 months)
- **Reference:** Candanedo et al. (2017), *Energy and Buildings*, 145, 13–25

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/mehditalebi01/Appliances-Energy-Prediction.git
cd Appliances-Energy-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook final_project.ipynb
```
Execute all cells from top to bottom. The notebook is fully self-contained and reproducible.

---

## 📁 Project Structure

```
├── final_project.ipynb     # Main notebook (primary deliverable)
├── report/
│   └── report.md           # Project report (3-4 pages)
├── figures/                # All generated plots (13 figures)
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_scatter_plots.png
│   ├── 04_time_series.png
│   ├── 05_temporal_patterns.png
│   ├── 06_outlier_detection.png
│   ├── 07_residual_plots.png
│   ├── 08_learning_curve.png
│   ├── 09_model_comparison.png
│   ├── 10_feature_importance.png
│   ├── 11_permutation_importance.png
│   ├── 12_linear_coefficients.png
│   └── 13_error_analysis.png
├── energydata_complete.csv # Dataset
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Results Summary

| Model | Test MAE | Test RMSE | Test R² |
|-------|----------|-----------|---------|
| **Random Forest** | **14.66** | **22.11** | **0.734** |
| Gradient Boosting | 15.71 | 23.16 | 0.708 |
| SVR (RBF) | 16.16 | 26.44 | 0.620 |
| Decision Tree | 17.13 | 27.66 | 0.584 |
| Linear Regression | 26.42 | 35.58 | 0.311 |
| Polynomial Ridge | 26.62 | 36.15 | 0.289 |

**Key insight:** Nonlinear/ensemble methods improve R² from ~0.30 (linear) to ~0.73 (Random Forest), confirming strong nonlinear relationships in energy consumption data.

---

## 🔬 Methodology

1. **EDA:** 6+ plots examining distributions, correlations, temporal patterns, and nonlinear relationships
2. **Preprocessing:** Feature engineering (time features), IQR outlier capping, StandardScaler (after train-test split)
3. **Models:** Linear Regression → Polynomial Ridge → Decision Tree → SVR → Random Forest → Gradient Boosting
4. **Tuning:** GridSearchCV with 5-fold CV for all models
5. **Evaluation:** MAE, RMSE, R², residual analysis, learning curves, feature importance, permutation importance

---

## 🔄 Reproducibility

- **Random seed:** `42` (fixed throughout)
- **Train-test split:** 80/20
- **Cross-validation:** 5-fold
- **Backend:** Matplotlib Agg (for non-interactive environments)
- **No hardcoded paths** — all paths are relative

To reproduce from scratch:
```bash
pip install -r requirements.txt
jupyter notebook final_project.ipynb
# Run all cells sequentially
```

---

## 📝 Report & Figures

- **Report:** [`report/report.md`](report/report.md) — 3-4 page project report with all required sections
- **Figures:** [`figures/`](figures/) — 13 publication-quality plots covering EDA, model evaluation, and interpretation

---

## 📚 References

1. Candanedo, L. M., Feldmann, A., & Degemmis, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 145, 13–25. [DOI](https://doi.org/10.1016/j.enbuild.2017.03.040)
2. [UCI ML Repository – Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
3. [Scikit-learn Documentation](https://scikit-learn.org)
