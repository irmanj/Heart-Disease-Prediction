import joblib
import pandas as pd

# load model dan nama fitur
model = joblib.load("models/heart_disease_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# data pasien baru
patient = {
    "age": 63,
    "trestbps": 145,
    "chol": 233,
    "thalch": 150,
    "oldpeak": 2.3,

    # dummy variables
    "sex_Male": 1,

    # semua dummy lainnya isi 0 jika tidak dipilih
    "cp_atypical angina": 0,
    "cp_non-anginal": 0,
    "cp_typical angina": 0,

    "fbs_True": 0,
    "restecg_normal": 0,
    "restecg_st-t abnormality": 0,

    "exang_True": 0,

    "slope_flat": 0,
    "slope_upsloping": 0,

    "thal_fixed defect": 0,
    "thal_reversible defect": 0,

    "ca": 0
}

# DataFrame sesuai fitur training
sample = pd.DataFrame(
    [[0.0] * len(feature_names)],
    columns=feature_names
)

# Isi data pasien
for feature, value in patient.items():
    if feature in sample.columns:
        sample.at[0, feature] = float(value)

prediction = model.predict(sample)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("❤️ Heart Disease")
else:
    print("💚 No Heart Disease")