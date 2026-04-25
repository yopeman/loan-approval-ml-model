import pandas as pd
import joblib

# Test the prediction function logic manually
df = pd.DataFrame([{
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
}])

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

print("Loaded encoders:", list(label_encoders.keys()))
print("Before encoding:")
print(df)
print(df.dtypes)

# Strip whitespace from string columns before encoding
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Encode categorical columns using saved label encoders
for col in ['education', 'self_employed']:
    if col in df.columns and col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

print("\nAfter encoding:")
print(df)
print(df.dtypes)

df_scaled = scaler.transform(df)
pred = model.predict(df_scaled)[0]
prob = model.predict_proba(df_scaled)[0]

print("\nPrediction:", pred)
print("Probabilities:", prob)
