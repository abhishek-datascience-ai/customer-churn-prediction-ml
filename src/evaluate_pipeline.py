import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_pipeline.joblib"


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def main():
    df = load_and_clean_data(DATA_PATH)
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = joblib.load(MODEL_PATH)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


if __name__ == "__main__":
    main()