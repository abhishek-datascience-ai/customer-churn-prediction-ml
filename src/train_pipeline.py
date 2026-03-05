import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)

    print(f"✅ Saved pipeline to: {MODEL_PATH}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")


if __name__ == "__main__":
    main()