import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Dataset
df = pd.read_csv("./loan_approval_dataset.csv")
df.columns = df.columns.str.strip().str.lower()

# Strip leading/trailing spaces from string values in all object columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Handle Missing Values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode Categorical Columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Encoders and their classes:")
for col, le in label_encoders.items():
    print(f"  {col}: {list(le.classes_)}")

# Split Features and Target
y = df["loan_status"]
X = df.drop("loan_status", axis=1)

# Split into Train & Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f'\nAccuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
print("\nModel, scaler, and encoders saved successfully!")

# Test predict
def predict_loan(data):
    df_input = pd.DataFrame([data])
    
    # Strip spaces from string columns
    for col in df_input.select_dtypes(include=['object']).columns:
        df_input[col] = df_input[col].astype(str).str.strip()
    
    # Encode categorical columns
    for col in ['education', 'self_employed']:
        if col in df_input.columns and col in label_encoders:
            df_input[col] = label_encoders[col].transform(df_input[col])
    
    df_scaled = scaler.transform(df_input)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0]
    return pred, prob

result, probs = predict_loan({
    'loan_id': 1,
    'no_of_dependents': 2,
    'education': 'Graduate',
    'self_employed': 'Yes',
    'income_annum': 5000000,
    'loan_amount': 15000000,
    'loan_term': 12,
    'cibil_score': 750,
    'residential_assets_value': 5000000,
    'commercial_assets_value': 2000000,
    'luxury_assets_value': 5000000,
    'bank_asset_value': 3000000,
})

print(f"\nPrediction: {result}, Probabilities: {probs}")
