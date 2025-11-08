#!/usr/bin/env python3
"""
train_isolation_mlflow_optuna.py
Orquestación MLflow + Optuna para IsolationForest.
Entrada: CSV con columna de etiqueta de anomalía (0/1).
"""

import argparse
import logging
from pathlib import Path
import json
import warnings

import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsolationForestTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols,
        label_col="is_anomaly",
        test_size=0.2,
        random_state=42,
        mlflow_experiment="IsolationForest_Optuna",
        n_trials=50,
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.test_size = test_size
        self.random_state = random_state
        self.mlflow_experiment = mlflow_experiment
        self.n_trials = n_trials

        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' not found in dataframe")

    def preprocess_split(self):
                # Define el número de días para la prueba y los días que deseas para el entrenamiento
        TEST_DAYS = 90
        TRAIN_DAYS_WINDOW = 90  # Días del entrenamiento desde el 1 de diciembre del 2024

        # ---------- 6. División train/test temporal y ventana de entrenamiento ----------

        # 6.1. Calcular la fecha de división (inicio del conjunto de prueba)
        split_date = scaled.index.max() - pd.Timedelta(days=TEST_DAYS)

        # 6.2. Calcular la fecha de inicio del entrenamiento
        train_start_date = split_date - pd.Timedelta(days=TRAIN_DAYS_WINDOW)

        # 6.3. Filtrar el conjunto de entrenamiento:
        # Solo los días dentro de la ventana y sin outliers
        train_df = scaled[
            (scaled.index >= train_start_date) &
            (scaled.index < split_date) &
            (scaled['is_outlier_mahalanobis'] == 'No')
        ].drop(columns=['day_of_week'])

        # 6.4. Filtrar el conjunto de prueba (sin cambios)
        test_df = scaled[
            scaled.index >= split_date
        ].drop(columns=['day_of_week'])

        # 6.5. Separar variables predictoras (X) y variable objetivo (y)
        X_train = train_df.drop(columns=['is_outlier_mahalanobis','mahalanobis_distance','FSx($)', 'Elastic Load Balancing($)', 'Resilience Hub($)','DataSync($)','Secrets Manager($)']).values
        y_train = train_df['is_outlier_mahalanobis'].values

        X_test = test_df.drop(columns=['is_outlier_mahalanobis','mahalanobis_distance','FSx($)', 'Elastic Load Balancing($)', 'Resilience Hub($)','DataSync($)','Secrets Manager($)']).values
        y_test = test_df['is_outlier_mahalanobis'].values

        # 6.6. Obtener nombres de columnas
        cols = train_df.drop(columns=['is_outlier_mahalanobis','mahalanobis_distance','FSx($)', 'Elastic Load Balancing($)', 'Resilience Hub($)','DataSync($)','Secrets Manager($)']).columns.tolist()

        # Simple preprocessing pipeline for numeric features
        preprocess = Pipeline(
            steps=[
            # CAMBIO: Usar strategy="constant" y especificar fill_value=0
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
                 ]
            )

        X_train_proc = preprocess.fit_transform(X_train)
        X_test_proc = preprocess.transform(X_test)

        self.preprocess_pipeline = preprocess
        self.X_train = X_train_proc
        self.X_test = X_test_proc
        self.y_train = y_train.values
        self.y_test = y_test.values
        return

    def _score_predictions(self, y_true, y_pred_label, y_scores=None):
        # y_pred_label: predicted anomaly label (1 anomaly, 0 normal)
        # sklearn IsolationForest uses -1 for outliers by default in predict
        # but here we will ensure predicted labels are 1 (anomaly) and 0 (normal)
        f1 = f1_score(y_true, y_pred_label, zero_division=0)
        prec = precision_score(y_true, y_pred_label, zero_division=0)
        rec = recall_score(y_true, y_pred_label, zero_division=0)
        roc = None
        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                roc = roc_auc_score(y_true, y_scores)
            except Exception:
                roc = None
        return {"f1": f1, "precision": prec, "recall": rec, "roc_auc": roc}

    def objective(self, trial: optuna.trial.Trial):
        # Hyperparameter search space for IsolationForest
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_samples = trial.suggest_float("max_samples", 0.1, 1.0)
        contamination = trial.suggest_float("contamination", 0.0001, 0.5, log=False)
        max_features = trial.suggest_float("max_features", 0.1, 1.0)
        bootstrap = trial.suggest_categorical("bootstrap", [False, True])
        behaviour = None  # kept for compatibility older sklearn

        # Model
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Fit on training set
        model.fit(self.X_train)

        # Predict on validation (test) set
        # predict -> -1 for anomaly, 1 for normal
        preds = model.predict(self.X_test)
        # convert to 1 anomaly, 0 normal
        preds_label = np.where(preds == -1, 1, 0)

        # decision_function: higher -> more normal. invert to have score for anomaly
        try:
            scores = -model.decision_function(self.X_test)
        except Exception:
            scores = None

        metrics = self._score_predictions(self.y_test, preds_label, y_scores=scores)

        # Log trial params and metrics to Optuna (MLflowCallback will also log to MLflow)
        trial.set_user_attr("n_train", int(self.X_train.shape[0]))
        # Objective: maximize F1 for anomalies
        return metrics["f1"]

    def run_optimization(self):
        # Prepare data
        self.preprocess_split()

        # MLflow experiment
        mlflow.set_experiment(self.mlflow_experiment)

        # Setup Optuna study
        study = optuna.create_study(direction="maximize", study_name="isolation_forest_study")

        # Integrate Optuna <-> MLflow
        mlflow_cb = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="f1"
        )

        logger.info("Starting Optuna optimization for IsolationForest")
        study.optimize(self.objective, n_trials=self.n_trials, callbacks=[mlflow_cb])

        logger.info("Optimization finished")
        self.study = study
        return study

    def train_and_log_best(self, run_name="final_model"):
        best_params = self.study.best_params
        # Retrain on full training data (or optionally entire dataset)
        model = IsolationForest(
            n_estimators=best_params.get("n_estimators", 100),
            max_samples=best_params.get("max_samples", 1.0),
            contamination=best_params.get("contamination", 0.1),
            max_features=best_params.get("max_features", 1.0),
            bootstrap=best_params.get("bootstrap", False),
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Fit on full training split used previously
        model.fit(self.X_train)

        preds = model.predict(self.X_test)
        preds_label = np.where(preds == -1, 1, 0)
        try:
            scores = -model.decision_function(self.X_test)
        except Exception:
            scores = None

        metrics = self._score_predictions(self.y_test, preds_label, y_scores=scores)

        # MLflow logging
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(best_params)
            mlflow.log_param("features", self.feature_cols)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            # log model (sklearn flavor)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="isolation_forest_model",
                registered_model_name=None,
            )
            # log preprocessing pipeline separately
            mlflow.sklearn.log_model(
                sk_model=self.preprocess_pipeline,
                artifact_path="preprocessing_pipeline",
            )
            # save a small sample of test set and preds
            sample_df = pd.DataFrame(self.X_test, columns=[f"f_{i}" for i in range(self.X_test.shape[1])])
            sample_df["y_true"] = self.y_test
            sample_df["y_pred"] = preds_label
            sample_csv = "predictions_sample.csv"
            sample_df.to_csv(sample_csv, index=False)
            mlflow.log_artifact(sample_csv)
        logger.info("Final model trained and logged to MLflow")
        return {"best_params": best_params, "metrics": metrics}


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    if p.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Train IsolationForest with Optuna + MLflow")
    parser.add_argument("--input", required=True, help="input CSV/Parquet path")
    parser.add_argument("--features", required=True, help="JSON list of feature column names or path to columns txt")
    parser.add_argument("--label-col", default="is_anomaly", help="label column name (0 normal, 1 anomaly)")
    parser.add_argument("--experiment", default="IsolationForest_Optuna", help="MLflow experiment name")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--test-size", type=float, default=0.2, help="test size fraction")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_dataframe(args.input)

    # parse feature list: accept JSON string or path to a file
    try:
        # try JSON parse
        feature_cols = json.loads(args.features)
        if not isinstance(feature_cols, list):
            raise ValueError()
    except Exception:
        # try to read as file with one column name per line
        fpath = Path(args.features)
        if fpath.exists():
            feature_cols = [ln.strip() for ln in fpath.read_text().splitlines() if ln.strip()]
        else:
            raise ValueError("features must be a JSON list or path to a file with one column per line")

    trainer = IsolationForestTrainer(
        df=df,
        feature_cols=feature_cols,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.random_state,
        mlflow_experiment=args.experiment,
        n_trials=args.n_trials,
    )

    study = trainer.run_optimization()
    result = trainer.train_and_log_best(run_name="final_isolation_forest_run")

    # Print concise result
    logger.info("Best params:")
    logger.info(result["best_params"])
    logger.info("Metrics on test set:")
    logger.info(result["metrics"])


if __name__ == "__main__":
    main()
