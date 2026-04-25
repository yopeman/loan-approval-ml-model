import pandas as pd
import joblib

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

def predict_loan(data):
    df = pd.DataFrame([data])

    for col in df.select_dtypes(include=['object', 'str']).columns:
        df[col] = df[col].astype(str).str.strip()

    for col in ['education', 'self_employed']:
        if col in df.columns and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

    df = scaler.transform(df)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    return pred, prob