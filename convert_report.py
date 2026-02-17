"""Convert report to PDF using fpdf2 write_html method."""
from fpdf import FPDF

html_content = """
<h1 align="center">Appliances Energy Prediction</h1>
<h3 align="center">Nonlinear Regression with Ensemble Methods</h3>
<p align="center"><i>Machine Learning Final Project | Mehdi Talebi | February 2026</i></p>
<br><br>

<h2>1. Introduction</h2>
<p>This project predicts household appliance energy consumption (Wh per 10-minute interval) using environmental sensor data from a low-energy house in Belgium. The dataset spans ~4.5 months with ~19,700 observations and 25 features including indoor/outdoor temperature, humidity, weather conditions, and time-derived variables.</p>
<p>Energy consumption exhibits inherently nonlinear patterns driven by occupancy thresholds, temperature comfort zones, and time-of-day effects. Linear models cannot capture these dependencies, motivating nonlinear and ensemble regression methods.</p>
<p><b>Dataset:</b> UCI ML Repository (Candanedo et al., 2017). 19,735 observations, 29 columns, target: Appliances (Wh). No missing values.</p>

<h2>2. Methods</h2>
<h3>EDA Summary</h3>
<p>Target is strongly right-skewed (skewness ~3.6), most readings below 100 Wh. Individual feature correlations are weak (max |r| &lt; 0.3). Clear temporal patterns with energy peaks during morning/evening hours.</p>

<h3>Preprocessing</h3>
<ul>
<li>Feature engineering: Extracted hour, day_of_week, month, is_weekend from timestamp</li>
<li>Dropped: date (after extraction), rv1, rv2 (random noise variables)</li>
<li>Outlier treatment: IQR-based capping on target variable</li>
<li>Scaling: StandardScaler fitted on training data only (prevents data leakage)</li>
</ul>

<h3>Models and Hyperparameter Tuning</h3>
<p>Six models trained with GridSearchCV (5-fold CV, neg_mean_squared_error scoring):</p>
<ul>
<li>Linear Regression (baseline, no tuning)</li>
<li>Ridge Polynomial Regression (degree=2, alpha=0.1)</li>
<li>Decision Tree Regressor (max_depth=20, min_samples_leaf=5)</li>
<li>SVR with RBF kernel (C=100, gamma=0.1, epsilon=0.5)</li>
<li>Random Forest Regressor (200 trees, min_samples_leaf=2)</li>
<li>Gradient Boosting Regressor (200 estimators, max_depth=7, lr=0.1)</li>
</ul>

<h2>3. Results</h2>
<h3>Model Comparison Table</h3>
<table border="1">
<thead><tr>
<th width="25%">Model</th><th width="11%">Tr MAE</th><th width="11%">Te MAE</th>
<th width="13%">Tr RMSE</th><th width="13%">Te RMSE</th><th width="11%">Tr R2</th><th width="11%">Te R2</th>
</tr></thead>
<tbody>
<tr><td><b>Random Forest</b></td><td>7.02</td><td><b>14.66</b></td><td>10.89</td><td><b>22.11</b></td><td>0.936</td><td><b>0.734</b></td></tr>
<tr><td>Gradient Boosting</td><td>10.70</td><td>15.71</td><td>14.68</td><td>23.16</td><td>0.883</td><td>0.708</td></tr>
<tr><td>SVR (RBF)</td><td>12.29</td><td>16.16</td><td>22.38</td><td>26.44</td><td>0.729</td><td>0.620</td></tr>
<tr><td>Decision Tree</td><td>9.53</td><td>17.13</td><td>15.62</td><td>27.66</td><td>0.868</td><td>0.584</td></tr>
<tr><td>Linear Regression</td><td>26.41</td><td>26.42</td><td>35.86</td><td>35.58</td><td>0.304</td><td>0.311</td></tr>
<tr><td>Polynomial Ridge</td><td>26.00</td><td>26.62</td><td>35.38</td><td>36.15</td><td>0.322</td><td>0.289</td></tr>
</tbody>
</table>

<h3>Key Findings</h3>
<ul>
<li><b>Random Forest</b> achieves best test performance: R2=0.734, MAE=14.66 Wh</li>
<li>Ensemble methods consistently outperform individual models</li>
<li>Nonlinear models improve R2 from ~0.31 (linear) to ~0.73 (ensemble)</li>
<li>Decision Tree shows overfitting (Train R2=0.87 vs Test R2=0.58); Random Forest mitigates this via bagging</li>
</ul>
<p><b>Feature Importance:</b> Top predictive features are lights (occupancy proxy), indoor temperatures (T6, T3, T8), and temporal features (hour, day_of_week).</p>

<h2>4. Discussion</h2>
<h3>Error Analysis</h3>
<p>The best model struggles with high-consumption spikes. Large-error cases (~5% of test set) are biased toward under-prediction of peak consumption. Residuals show near-zero mean (unbiased) with mild heteroscedasticity at higher predicted values.</p>

<h3>Limitations</h3>
<ul>
<li>Single building: Results specific to one low-energy house in Belgium</li>
<li>4.5-month window: Incomplete seasonal coverage</li>
<li>No explicit occupancy data: Model relies on indirect proxies</li>
<li>Independence assumption: Ignores temporal autocorrelation</li>
</ul>

<h3>Future Work</h3>
<ul>
<li>Time-series models (LSTM, GRU) for temporal dependencies</li>
<li>Real-time occupancy sensors as additional predictors</li>
<li>Weather forecast integration for anticipatory energy management</li>
<li>Multi-building generalization with transfer learning</li>
</ul>

<h2>References</h2>
<p><font size="8">[1] Candanedo, L.M., Feldmann, A., and Degemmis, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. Energy and Buildings, 145, 13-25.<br>
[2] UCI ML Repository: archive.ics.uci.edu/dataset/374<br>
[3] Scikit-learn documentation: scikit-learn.org<br>
[4] Pandas: pandas.pydata.org | Matplotlib: matplotlib.org</font></p>

<h2>AI Usage Statement</h2>
<p><b>Did you use any generative AI tools?</b> No. All code, analysis, and written content were produced independently. External resources consulted: scikit-learn documentation, pandas documentation, and the original dataset paper (Candanedo et al., 2017).</p>
"""

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.write_html(html_content)
pdf.output('report/report.pdf')
print(f'PDF report created: report/report.pdf ({pdf.pages_count} pages)')
