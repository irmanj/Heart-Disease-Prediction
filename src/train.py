import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("data/heart_disease_uci.csv")

print(df.head())

df.info()

print(df.describe())

# drop kolom yang tidak dibutuhkan
df = df.drop(columns=["id", "dataset"])

# mengubah target menjadi biner
df["num"] = (df["num"] > 0).astype(int)

# mengubah semua kategori menjadi angka
df = pd.get_dummies(df, drop_first=True)

# missing value
df = df.fillna(0)

X = df.drop("num", axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
                                    X, 
                                    y, 
                                    test_size=0.2, 
                                    random_state=42,
                                    stratify=y
                                )

# model = LogisticRegression(max_iter=1000)
# model = DecisionTreeClassifier(random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/heart_disease_model.pkl")

# save feature names
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("Model saved!")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))