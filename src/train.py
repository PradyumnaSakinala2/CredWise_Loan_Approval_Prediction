"""
CredWise — Loan Approval Prediction
===================================

Reproducible, script-form version of notebooks/credit_wise.ipynb.

Pipeline
--------
1. Load data
2. Impute missing values (mean for numeric, mode for categorical)
3. Encode categoricals (label-encode ordinal/binary, one-hot the rest)
4. Train/test split + standard scaling
5. Train and evaluate Logistic Regression, KNN and Gaussian Naive Bayes
6. Repeat step 5 on an engineered feature set (squared Credit_Score / DTI_Ratio)

Usage
-----
    python src/train.py
    python src/train.py --data data/loan_approval_data.csv --test-size 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

ONE_HOT_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]
TARGET = "Loan_Approved"


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the raw CSV and impute missing values."""
    df = pd.read_csv(path)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns

    df[numerical_cols] = SimpleImputer(strategy="mean").fit_transform(df[numerical_cols])
    df[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
        df[categorical_cols]
    )

    return df.drop("Applicant_ID", axis=1)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the binary columns and one-hot encode the nominal ones."""
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df[TARGET] = LabelEncoder().fit_transform(df[TARGET])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[ONE_HOT_COLS])
    encoded_df = pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(ONE_HOT_COLS), index=df.index
    )

    return pd.concat([df.drop(columns=ONE_HOT_COLS), encoded_df], axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the two strongest predictors with squared transforms."""
    df = df.copy()
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    return df.drop(columns=["Credit_Score", "DTI_Ratio"])


def split_and_scale(df: pd.DataFrame, test_size: float, random_state: int):
    """Split off a test set, then fit the scaler on train only."""
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_test),
        y_train,
        y_test,
    )


def evaluate(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """Fit a model and report the headline classification metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scores = {
        "model": name,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    print(f"\n{name}")
    for key in ("precision", "recall", "f1", "accuracy"):
        print(f"  {key.capitalize():<10} {scores[key]:.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    return scores


def run_all_models(X_train, X_test, y_train, y_test, random_state: int) -> pd.DataFrame:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
    }
    results = [
        evaluate(name, model, X_train, y_train, X_test, y_test)
        for name, model in models.items()
    ]
    return pd.DataFrame(results).set_index("model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CredWise loan-approval models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "loan_approval_data.csv",
        help="Path to the loan approval CSV.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = encode(load_and_clean(args.data))
    print(f"Loaded {len(df)} rows, {df.shape[1] - 1} features after encoding.")
    print(f"Class balance:\n{df[TARGET].value_counts(normalize=True).round(3)}")

    print("\n" + "=" * 60)
    print("BASELINE FEATURE SET")
    print("=" * 60)
    baseline = run_all_models(
        *split_and_scale(df, args.test_size, args.random_state), args.random_state
    )

    print("\n" + "=" * 60)
    print("ENGINEERED FEATURE SET (squared Credit_Score / DTI_Ratio)")
    print("=" * 60)
    engineered = run_all_models(
        *split_and_scale(engineer_features(df), args.test_size, args.random_state),
        args.random_state,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = pd.concat(
        {"baseline": baseline, "engineered": engineered}, names=["feature_set"]
    )
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
