# ❤️ Heart Disease Prediction

Machine Learning project to predict whether a patient has heart disease using the UCI Heart Disease dataset.

## Dataset

- Heart Disease UCI

## Models Compared

- Logistic Regression
- Decision Tree
- Random Forest

## Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 82.07% |
| Decision Tree | 76.63% |
| Random Forest | **84.78%** |

Best Model: **Random Forest**

## Technologies

- Python
- Pandas
- Scikit-learn
- Joblib

## Project Structure

```
Heart Disease Prediction/
│
├── data/
├── models/
├── src/
├── requirements.txt
└── README.md
```

## Run

Train model

```bash
python src/train.py
```

Predict

```bash
python src/predict.py
```

## Project Workflow

```
Dataset
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Save Model
   ↓
Prediction

```

## API Documentation

Run API

```bash
uvicorn src.app:app --reload
```

Open Swagger

```
http://127.0.0.1:8000/docs
```

Example Request

```json
{
  "age": 63,
  "trestbps": 145,
  "chol": 233,
  "thalch": 150,
  "oldpeak": 2.3,
  "sex_Male": 1,
  "ca": 0
}
```

Example Response

```json
{
  "prediction": 1,
  "result": "Heart Disease"
}
```

## Swagger API
![Swagger](/images/swagger.png.png)

## Disclaimer

This project is for educational purposes only and should not be used as medical advice or clinical diagnosis.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)