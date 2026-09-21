Customer Churn Prediction

Predicting telecom customer churn with XGBoost and explaining the drivers with SHAP.

Dataset

IBM Telco Customer Churn sample data: 7,043 customers, 21 features (source).
After removing 11 rows with blank TotalCharges: 7,032 rows. Churn rate: 26.6%.

Approach
Cleaned data (numeric conversion, dropped missing values, label-encoded categoricals)
80/20 train/test split
Trained XGBoost and compared it against a logistic regression baseline
Evaluated with ROC-AUC, precision, recall, and 5-fold cross-validation
Handled class imbalance with scale_pos_weight
Explained predictions with SHAP
Results (test set)
Model	ROC-AUC	Precision (churn)	Recall (churn)
Logistic Regression	0.83	-	-
XGBoost	0.83	0.63	0.50
XGBoost (class-weighted)	0.83	0.50	0.80

5-fold CV ROC-AUC: XGBoost 0.847 ± 0.005, Logistic Regression 0.845 ± 0.006

Note: the unweighted XGBoost row is from the notebook (non-stratified split); the other rows are from day2_additions.py (stratified split), so small differences are expected.

Key findings
Month-to-month customers churn at 42.7%, versus 11.3% on one-year and 2.8% on two-year contracts.
Customers in their first 12 months churn at 48.3%, versus 17.5% for longer-tenured customers.
SHAP ranks Contract, tenure, and MonthlyCharges as the top three drivers, consistent with the segment rates above.
Class-weighting raised churn recall from 0.50 to 0.80, at the cost of precision (0.63 to 0.50).
A logistic regression baseline matched XGBoost, so the simpler model is a reasonable choice for interpretability.
Recommendation: focus retention offers on new month-to-month customers, for example discounts to move them to longer contracts. These are correlations, not proof that contract type causes churn.
Charts
Show Image
Show Image
Show Image
Tech stack

Python, Pandas, Scikit-Learn, XGBoost, SHAP, Matplotlib, Seaborn

How to run
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
jupyter notebook Customer_churn_prediction.ipynb
python day2_additions.py
