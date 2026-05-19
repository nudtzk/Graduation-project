"""Train a shear-wall resistance regression model.

This script uses the cleaned LS-DYNA-derived dataset in ``all_data.csv`` to
train a decision-tree regression model for shear-wall resistance prediction. It
supports feature normalization, k-fold cross-validation, prediction plotting,
and optional SHAP feature-importance plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeRegressor

TARGET_COLUMN = "F_resistance"
DEFAULT_FEATURE_COLUMNS = [
    "width",
    "height",
    "span",
    "covering",
    "num_longitude",
    "diameter_longitude",
    "num_hoop",
    "diameter_hoop",
    "concrete_stress",
    "deflection",
]


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    missing_columns = [
        column for column in [*DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN] if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    features = data[DEFAULT_FEATURE_COLUMNS].copy()
    target = data[TARGET_COLUMN].copy()
    return features, target


def build_model(normalize_features: bool, random_state: int) -> Pipeline:
    steps = []
    if normalize_features:
        steps.append(("normalizer", Normalizer()))
    steps.append(
        (
            "regressor",
            DecisionTreeRegressor(random_state=random_state),
        )
    )
    return Pipeline(steps)


def evaluate_model(model: Pipeline, features: pd.DataFrame, target: pd.Series, random_state: int) -> dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def cross_validate(model: Pipeline, features: pd.DataFrame, target: pd.Series, folds: int) -> np.ndarray:
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    return cross_val_score(model, features, target, cv=cv, scoring="r2")


def save_prediction_plot(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    output_path: Path,
    random_state: int,
) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    plot_frame = pd.DataFrame(
        {
            "deflection": x_test["deflection"],
            "actual": y_test,
            "predicted": predictions,
        }
    ).sort_values("deflection")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(plot_frame["deflection"], plot_frame["actual"], s=16, label="LS-DYNA data")
    plt.scatter(plot_frame["deflection"], plot_frame["predicted"], s=16, label="Predicted resistance")
    plt.title("Shear-Wall Resistance Prediction")
    plt.xlabel("Deflection")
    plt.ylabel("Resistance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_shap_plots(model: Pipeline, features: pd.DataFrame, output_dir: Path) -> None:
    try:
        import shap
    except ImportError:
        print("SHAP is not installed; skipping SHAP plots.")
        return

    transformed_features = features
    regressor = model.named_steps["regressor"]
    if "normalizer" in model.named_steps:
        transformed_features = pd.DataFrame(
            model.named_steps["normalizer"].transform(features),
            columns=features.columns,
        )

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(transformed_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, transformed_features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shear-wall resistance prediction model.")
    parser.add_argument("--data", type=Path, default=Path("all_data.csv"), help="Input dataset path.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory.")
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable feature normalization.")
    parser.add_argument("--shap", action="store_true", help="Generate SHAP summary plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, target = load_dataset(args.data)
    model = build_model(normalize_features=not args.no_normalize, random_state=args.random_state)

    scores = cross_validate(model, features, target, args.folds)
    metrics = evaluate_model(model, features, target, args.random_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_plot(
        model,
        features,
        target,
        args.output_dir / "prediction_comparison.png",
        args.random_state,
    )

    model.fit(features, target)
    if args.shap:
        save_shap_plots(model, features, args.output_dir)

    print("Cross-validation R2 scores:", np.round(scores, 4))
    print(f"Mean CV R2: {scores.mean():.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Prediction plot saved to {args.output_dir / 'prediction_comparison.png'}")


if __name__ == "__main__":
    main()
