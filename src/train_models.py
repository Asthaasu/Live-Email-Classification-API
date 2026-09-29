import os
import sys
import pickle
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier


# Allow imports from the current src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gmail_features import prepare_training_data


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "model")

mlflow.set_experiment("spam_detector_v2")


def evaluate_model(model, X_test, y_test):
    """Return the Phase 5 comparison metrics."""

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def print_results(model_name, metrics):
    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


def main():
    X, y, feature_names, scaler = prepare_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nPHASE 5: MLFLOW MODEL COMPARISON")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # -------------------------------------------------
    # MODEL 1: Balanced Random Forest
    # -------------------------------------------------
    with mlflow.start_run(run_name="rf_balanced"):

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        rf.fit(X_train, y_train)

        rf_metrics = evaluate_model(rf, X_test, y_test)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("class_weight", "balanced")

        for metric_name in ["precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(metric_name, rf_metrics[metric_name])

        mlflow.sklearn.log_model(rf, "model")

        print_results("RANDOM FOREST — BALANCED", rf_metrics)

    # -------------------------------------------------
    # MODEL 2: Class-balanced XGBoost
    # -------------------------------------------------
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    with mlflow.start_run(run_name="xgboost_balanced"):

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
        )

        xgb.fit(X_train, y_train)

        xgb_metrics = evaluate_model(xgb, X_test, y_test)

        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        for metric_name in ["precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(metric_name, xgb_metrics[metric_name])

        mlflow.sklearn.log_model(xgb, "model")

        print_results("XGBOOST — BALANCED", xgb_metrics)

    # Save both models locally for the later winner-selection step
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "rf_balanced.pkl"), "wb") as file:
        pickle.dump(rf, file)

    with open(os.path.join(MODEL_DIR, "xgb_balanced.pkl"), "wb") as file:
        pickle.dump(xgb, file)

    with open(
        os.path.join(MODEL_DIR, "phase5_feature_names.pkl"),
        "wb",
    ) as file:
        pickle.dump(list(feature_names), file)
    with open(
        os.path.join(MODEL_DIR, "scaler.pkl"),
        "wb",
    ) as file:
        pickle.dump(scaler, file)

    print("\nPHASE 5 TRAINING COMPLETE")
    print("Start the MLflow UI with: mlflow ui")


if __name__ == "__main__":
    main()
