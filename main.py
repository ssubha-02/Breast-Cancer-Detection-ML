import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import joblib

# ============= 1. LOAD THE DATASET =============
# data.csv must be in the SAME FOLDER as this main.py
df = pd.read_csv("data.csv")

print("✅ Data loaded successfully!")
print("First 5 rows:")
print(df.head())
print("\nColumns:")
print(df.columns)
print("\nShape:", df.shape)

# ============= 2. CLEAN & PREPARE DATA =============
# Drop useless columns if they exist
for col in ["Unnamed: 32", "id"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Map diagnosis: B = 0 (benign), M = 1 (malignant)
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Separate features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("\n✅ After cleaning:")
print("Features shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# ============= 3. TRAIN-TEST SPLIT =============
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ============= 4. FEATURE SCALING =============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============= 5. DEFINE MODELS =============
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
}

# ============= 6. TRAIN & EVALUATE MODELS =============
for name, model in models.items():
    print("\n====================================")
    print(f"🏁 Model: {name}")
    print("====================================")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Probabilities for AUC
    try:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

# ============= 7. SAVE BEST MODEL (RANDOM FOREST) =============
best_model = models["Random Forest"]
joblib.dump(scaler, "model_scaler.joblib")
joblib.dump(best_model, "model_random_forest.joblib")

print("\n✅ Model and scaler saved as 'model_scaler.joblib' and 'model_random_forest.joblib'")
print("🎉 Breast Cancer Detection pipeline completed!")
