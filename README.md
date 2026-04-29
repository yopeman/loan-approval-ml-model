# Loan Approval Predictor

## Introduction

This project focuses on building a machine learning model to predict loan approval status based on applicant financial and personal attributes. In the banking and financial sector, accurate loan approval prediction is crucial for risk assessment and decision-making. By leveraging data mining techniques, this project aims to automate the loan approval process, helping financial institutions make consistent, data-driven decisions while minimizing the risk of default.

## Dataset Description

### Data Source
- **Source:** https://www.kaggle.com
- **Dataset Name:** Loan Approval Dataset
- **Access Method:** Public dataset available on Kaggle's data science platform

### Dataset Statistics
- **Number of Instances:** 4,269 loan applications
- **Number of Attributes:** 13 (12 features + 1 target variable)

### Dataset Schema

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| loan_id | Integer | Unique identifier for each loan application |
| no_of_dependents | Integer | Number of dependents the applicant has (0-5) |
| education | Categorical | Education level: "Graduate" or "Not Graduate" |
| self_employed | Categorical | Self-employment status: "Yes" or "No" |
| income_annum | Integer | Annual income of the applicant (in currency units) |
| loan_amount | Integer | Amount of loan requested (in currency units) |
| loan_term | Integer | Duration of the loan in months (2-20) |
| cibil_score | Integer | Credit score ranging from 300 to 900 |
| residential_assets_value | Integer | Value of residential assets owned |
| commercial_assets_value | Integer | Value of commercial assets owned |
| luxury_assets_value | Integer | Value of luxury assets owned |
| bank_asset_value | Integer | Total value of assets in bank |
| loan_status | Categorical (Target) | Loan decision: "Approved" or "Rejected" |

### Data Characteristics
- **Numeric Features:** Income, loan amount, CIBIL score, various asset values, number of dependents, loan term
- **Categorical Features:** Education level, self-employment status
- **Target Variable:** Binary classification (Approved/Rejected)
- **Data Quality:** No missing values in the dataset
- **CIBIL Score Range:** 300-900 (higher scores indicate better creditworthiness)

## Methodology

### Train-Test Split Ratio
- **Split:** 80% training, 20% testing
- **Justification:** The 80/20 split is a standard industry practice that provides sufficient data for model training (3,415 samples) while maintaining a robust test set (854 samples) for unbiased performance evaluation. This balance ensures the model learns patterns effectively while allowing reliable generalization assessment.

### Dataset Selection Rationale
This dataset was selected for the following reasons:
1. **Real-world Relevance:** Loan approval prediction is a critical business problem in the financial sector
2. **Balanced Complexity:** Contains both numeric and categorical features representing realistic applicant profiles
3. **Clean Data Structure:** Well-organized data with no missing values, allowing focus on algorithm implementation
4. **Interpretable Features:** Each feature has clear business meaning and relevance to loan decision-making
5. **Binary Classification:** Suitable for demonstrating fundamental data mining and machine learning concepts

### Algorithm Selection: Random Forest Classifier

**Why Random Forest?**
- **Robustness:** Ensemble method that reduces overfitting by combining multiple decision trees
- **Feature Importance:** Automatically ranks features by their contribution to predictions
- **Handles Mixed Data Types:** Works well with both numerical and categorical features
- **No Feature Scaling Required:** Though we applied scaling for consistency, Random Forest is less sensitive to feature scales
- **High Accuracy:** Typically achieves strong performance on tabular data

### How Random Forest Works

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees.

**Key Mechanisms:**

1. **Bootstrap Aggregating (Bagging):**
   - Creates multiple subsets of the training data using random sampling with replacement
   - Each decision tree is trained on a different bootstrap sample

2. **Random Feature Selection:**
   - At each split in the tree, only a random subset of features is considered
   - This decorrelates the trees and reduces variance

3. **Voting Mechanism:**
   - Each tree in the forest votes for a class prediction
   - The final prediction is determined by majority voting

**Advantages for This Problem:**
- Resistant to outliers in financial data
- Captures non-linear relationships between features
- Provides feature importance scores for business insights
- Handles interactions between variables (e.g., income + CIBIL score)

## Implementations

### Data Preprocessing

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load and clean data
df = pd.read_csv('./dataset.csv')
df.columns = df.columns.str.strip().str.lower()

# Strip whitespace from categorical values
for col in df.select_dtypes(include=['object', 'str']).columns:
    df[col] = df[col].astype(str).str.strip()

# Handle missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['str']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
```

### Feature Preparation and Splitting

```python
# Remove identifier and separate features/target
df = df.drop('loan_id', axis=1)
y = df["loan_status"]
X = df.drop("loan_status", axis=1)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Model Training

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### Model Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print(f'Accuracy Score: {accuracy_score(y_test, y_pred) * 100}%')
print(classification_report(y_test, y_pred))
```

### Model Persistence

```python
import joblib

joblib.dump(model, './model.joblib')
joblib.dump(scaler, './scaler.joblib')
joblib.dump(label_encoders, './label_encoders.joblib')
```

## Result and Discussion

### Performance Metrics

#### Overall Accuracy
| Metric | Value |
|--------|-------|
| **Accuracy** | **97.54%** |

#### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Rejected (0) | 0.97 | 0.99 | 0.98 | 536 |
| Approved (1) | 0.98 | 0.96 | 0.97 | 318 |
| **Macro Average** | **0.98** | **0.97** | **0.97** | 854 |
| **Weighted Average** | **0.98** | **0.98** | **0.98** | 854 |

### Interpretation of Findings

1. **High Overall Accuracy (97.54%):**
   - The Random Forest model demonstrates excellent predictive capability
   - Only 2.46% of predictions were incorrect out of 854 test cases

2. **Balanced Performance Across Classes:**
   - **Rejected Loans (Class 0):** High precision (0.97) and exceptional recall (0.99)
     - The model correctly identifies 99% of loans that should be rejected
     - Very few false positives (approving loans that should be rejected)
   
   - **Approved Loans (Class 1):** Strong precision (0.98) and good recall (0.96)
     - High precision means few false approvals (rejecting loans that should be approved)
     - 96% recall indicates most worthy applicants are correctly approved

3. **Business Impact Analysis:**
   - **False Positives (Approving bad loans):** Very low risk due to 0.99 recall on rejections
   - **False Negatives (Rejecting good loans):** Minimal at 4% (1 - 0.96 recall)
   - The model strikes an excellent balance between risk management and customer satisfaction

4. **Feature Importance Insights:**
   - CIBIL score (credit score) is likely the most influential feature
   - Asset values and income contribute significantly to predictions
   - Education and self-employment status provide additional discriminative power

### Model Validation

The model was validated using the held-out test set of 854 samples (20% of total data). The consistent high performance across both classes indicates:
- No overfitting to the training data
- Good generalization to unseen loan applications
- Robust feature representations

## Summary

### Key Insights on Experimental Outcomes

1. **Model Effectiveness:** The Random Forest Classifier achieved an exceptional accuracy of 97.54% on loan approval prediction, demonstrating its suitability for this financial classification task.

2. **Risk Management:** The model's high recall (0.99) for rejected loans is particularly valuable for financial institutions, as it minimizes the risk of approving high-risk applications.

3. **Customer Experience:** With 96% recall on approved loans, the model ensures that the majority of creditworthy applicants are correctly identified, maintaining good customer relations.

4. **Feature Engineering Value:** The combination of financial metrics (CIBIL score, income, assets) with demographic factors (education, self-employment) provides a comprehensive view of applicant creditworthiness.

5. **Preprocessing Impact:** Standard scaling and label encoding proved effective for preparing the mixed data types, while the 80/20 split provided reliable performance estimates.

### Conclusion

This project successfully demonstrates the application of data mining techniques to solve a real-world financial prediction problem. The Random Forest model's high accuracy and balanced performance across both approval classes make it a reliable tool for automated loan approval screening. The methodology combining proper data preprocessing, appropriate algorithm selection, and rigorous evaluation provides a robust framework for similar classification tasks in the financial domain.

---

**Project Files:**
- `dataset.csv`: Original dataset (4,269 records)
- `train_model.ipynb`: Complete training pipeline and analysis
- `model.joblib`: Trained Random Forest model
- `scaler.joblib`: Feature scaler for preprocessing
- `label_encoders.joblib`: Encoders for categorical variables

---

**Group Members:**

| Name | ID |
|------|-----|
| Yohanes Debebe | 1508306 |
| Sisay Atnkut | 1507749 |
| Maedot Demelash | 1506864 |
| Walelign Enemayehu | 1404442 |
| Elsaday Mengesha | 1505870 |
| Israel Assefa | 1506604 |
