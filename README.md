# Customer Churn Prediction

A Machine Learning project to analyze customer behavior and predict the likelihood of customer churn using the Telco Customer Churn dataset. This project covers the end-to-end data science workflow, from data preprocessing and exploratory data analysis to model development, evaluation, and deployment through an interactive Streamlit dashboard.

## Project Overview

Customer churn is an important challenge for subscription-based businesses. Understanding customer behavior and identifying customers who are more likely to leave can help businesses develop better customer retention strategies.

This project aims to:
- Explore customer characteristics and patterns associated with churn.
- Identify key features related to customer churn.
- Develop and evaluate machine learning classification models.
- Predict customer churn probability using a trained model.
- Provide business recommendations to support customer retention strategies.

## Dashboard Preview

![Customer Churn Prediction Dashboard](assets/feature_performance.png)


## Dataset

The project uses the **Telco Customer Churn dataset**, which contains customer information from a telecommunications company.

The dataset includes:
- Customer demographics
- Account information
- Services subscribed to
- Contract details
- Payment methods
- Customer tenure
- Churn status

**Target variable:** `Churn`

- `0`: No Churn
- `1`: Churn

## Project Workflow

### 1. Data Preprocessing
- Data cleaning and handling missing values.
- Numerical feature imputation using median values.
- Categorical feature imputation using the most frequent values.
- Categorical encoding using One-Hot Encoding.
- Numerical feature scaling using StandardScaler.
- Feature preparation for model development.

### 2. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand customer characteristics and identify patterns associated with churn.

Key observations include:
- Customer tenure has a negative association with churn.
- Fiber optic internet service shows a positive association with churn.
- Customers with longer-term contracts tend to have lower churn rates.
- Electronic check payment is positively associated with churn.

### 3. Model Development

Several classification algorithms were evaluated:

- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost

Logistic Regression achieved the best baseline ROC-AUC score and was selected for further optimization.

### 4. Model Optimization

Logistic Regression was optimized using GridSearchCV with Stratified 5-Fold Cross-Validation.

The optimization process included:
- Hyperparameter tuning.
- Evaluation using ROC-AUC.
- Classification threshold optimization using out-of-fold predictions.

**Best Hyperparameters:**

| Parameter | Value |
|---|---|
| C | 10 |
| Penalty | L2 |
| Solver | liblinear |
| Class Weight | None |
| Classification Threshold | 0.32 |

### 5. Model Evaluation

The final model was evaluated on the held-out test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 75.94% |
| Precision (Churn) | 53.45% |
| Recall (Churn) | 72.46% |
| F1-Score (Churn) | 61.52% |
| ROC-AUC | 83.82% |

The selected classification threshold of 0.32 prioritizes identifying a larger proportion of customers who may churn.

The model achieves a churn recall of 72.46%, indicating that it identifies approximately 72% of actual churn cases in the test dataset.

### 6. Feature Insights

Logistic Regression coefficients were analyzed to understand how customer characteristics relate to the model's churn predictions.

Some notable features include:

| Feature | Coefficient Direction |
|---|---|
| Internet Service - Fiber Optic | Positive |
| Monthly Charges | Negative |
| Contract - Two Year | Negative |
| Tenure | Negative |
| Payment Method - Electronic Check | Positive |
| Paperless Billing | Positive |

Positive coefficients indicate higher predicted churn odds, while negative coefficients indicate lower predicted churn odds, relative to the model's reference categories and other included features.

These relationships should not be interpreted as causal effects.

## Streamlit Dashboard

An interactive dashboard was developed using Streamlit to make the model and its findings accessible.

The dashboard consists of six main sections:

| Page | Description |
|---|---|
| Overview | Displays overall customer statistics and churn distribution. |
| Customer Analysis | Explores churn patterns across customer characteristics. |
| Model Performance | Presents model evaluation metrics and confusion matrix. |
| Feature Insights | Visualizes Logistic Regression coefficients. |
| Prediction | Predicts individual customer churn probability and risk level. |
| Business Recommendations | Provides retention strategies based on observed patterns and model insights. |

## Business Recommendations

Based on the analysis, several potential customer retention strategies were identified:

1. **Contract Retention:** Encourage customers to consider longer-term contracts through suitable retention programs.
2. **Early Customer Retention:** Strengthen onboarding and customer support during the early subscription period.
3. **Service Experience:** Evaluate customer experience and service quality among fiber optic users.
4. **Payment Experience:** Review potential payment difficulties among electronic check users.
5. **Prediction-Assisted Retention:** Use model predictions to help prioritize customer retention activities.

These recommendations are exploratory and should be evaluated using actual customer feedback and retention outcomes.

## Tech Stack

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Optimization:** GridSearchCV, StratifiedKFold
- **Model Persistence:** Joblib
- **Dashboard:** Streamlit

## Project Structure

```text
customer-churn/
│
├── data/
│   ├── raw.csv (Telco Dataset)
│   └── cleaned_processed.csv (final dataset)
│
├── models/
│   ├── churn_model_final.pkl
│   └── preprocessing.pkl
│
├── notebooks/
│   └── eda.ipynb
|   └── feature_engineering.ipynb
|   └── ml_training.ipynb
│
├── assets/
│   └── dashboard-overview.png
│
├── app.py
└── README.md
```

## Limitations

- The model is developed using a single public telecommunications dataset.
- Model performance may vary when applied to different customer populations.
- The selected classification threshold reflects a trade-off between precision and recall.
- Model predictions represent estimated churn probabilities, not guaranteed customer behavior.
- Feature relationships do not establish causal effects.

## Future Improvements

- Perform additional feature engineering and model optimization.
- Explore more robust validation strategies.
- Improve model interpretability using SHAP.
- Integrate real-time customer data.
- Evaluate retention strategies using actual business outcomes.
- Deploy the dashboard to a cloud platform.

## Author

**Rifki Ramadhani**
