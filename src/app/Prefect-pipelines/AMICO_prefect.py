#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.metrics import classification_report

import mlflow
from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from sklearn.preprocessing import StandardScaler

import s3fs 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration with fallback
def setup_mlflow():
    """Setup MLflow with proper error handling and fallback options."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # Test connection
        mlflow.search_experiments()
        logger.info(f"Connected to MLflow at: {mlflow_uri}")
    except Exception as e:
        logger.warning(f"Failed to connect to {mlflow_uri}: {e}")
        logger.info("Falling back to local SQLite database")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    try:
        mlflow.set_experiment("AMICO-experiment-prefect")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise

# Initialize MLflow
setup_mlflow()


@task(name="load_data", description="Carga de la informacion de costos desde un bucket de S3", retries=3, retry_delay_seconds=10)
def read_dataframe(s3_url: str) -> pd.DataFrame:
    """
    Carga de los datos de costos.

    Args:
        s3_url : Url del bucket de S3
        
    Returns:
        Processed DataFrame
    """
    logger = get_run_logger()
    
    logger.info(f"Loading data from: {s3_url}")
    
    try:
        df = pd.read_csv(s3_url)
        logger.info(f"Successfully loaded {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to load data from {s3_url}: {e}")
        raise

    # 2. DEFINIR summary_data con metadatos simples
    summary_data = [
        {"Métrica": "Número de Filas", "Valor": len(df)},
        {"Métrica": "Número de Columnas", "Valor": len(df.columns)},
        {"Métrica": "Tamaño (KB)", "Valor": f"{df.memory_usage(index=False, deep=True).sum() / 1024:.2f}"}
    ]

    create_table_artifact(
        key=f"data-summary-amico",
        table=summary_data,
        description=f"Data summary for AMICO"
    )

    return df

@task(name="create_features", description="Create feature Amico")
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature matrix from DataFrame.

    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (feature matrix, DictVectorizer)
    """
    logger = get_run_logger()
    
    df_local = df.copy() 

    # eliminar fila de Service total y columna de total costs
    df_local = df_local[df_local['Service'] != 'Service total']
    df_local = df_local.drop('Total costs($)', axis=1)

    # convertir columna de fecha en indice
    df_local['Service'] = pd.to_datetime(df_local['Service'])
    df_local = df_local.rename(columns={'Service':'date'}).sort_values('date').set_index('date')

    # forzar numérico y revisar columnas
    df_local = df_local.apply(pd.to_numeric, errors='coerce')
    # num_cols = df_local.columns.tolist() # Esta línea no es necesaria si no la usas

    # ---------- 3. imputación ----------
    # Strategy: cambiar a 0 los valores nulos
    df_imputed = df_local.fillna(0)

    df_bc=df_imputed
    num_cols = df_bc.columns.tolist()
        # ---------- 5. normalización por día de la semana ----------
    df_bc['day_of_week'] = df_bc.index.day_name()
    scaled = df_bc.copy()
    features = [c for c in num_cols]  # lista de features reales

    for day in scaled['day_of_week'].unique():
        mask = scaled['day_of_week'] == day
        if mask.sum() < 2:
            # si no hay suficientes ejemplos para el día, omitir
            continue
        scaler = StandardScaler()
        scaled.loc[mask, features] = scaler.fit_transform(scaled.loc[mask, features])

    # 1. Definir las columnas numéricas (excluimos 'mahalanobis_distance' previa y no numéricas)
    numerical_cols = [
        'Relational Database Service($)',
        'EC2-Instances($)',
        'Elastic File System($)',
        'EC2-Other($)',
        'CloudWatch($)',
        'S3($)',
        'Backup($)',
        'Key Management Service($)'
    ]

    # 3. Extraer los datos numéricos
    X_scaled = scaled[numerical_cols].values

    # 4. Calcular matriz de covarianza y pseudo-inversa
    covariance_matrix = np.cov(X_scaled.T)
    inv_covariance_matrix = np.linalg.pinv(covariance_matrix)

    # 5. Calcular el vector media
    mean_vector = np.mean(X_scaled, axis=0)

    # 6. Definir el umbral manual
    threshold = 4.2
    # 7. Función para calcular la distancia de Mahalanobis
    def mahalanobis_distance(x, mean, inv_cov):
        diff = x - mean
        return np.sqrt(diff.dot(inv_cov).dot(diff.T))

    # 8. Calcular las distancias
    scaled['mahalanobis_distance'] = [
        mahalanobis_distance(row, mean_vector, inv_covariance_matrix) for row in X_scaled
    ]

    # 9. Clasificar los outliers (distancia > 5)
    scaled['is_outlier_mahalanobis'] = np.where(
        scaled['mahalanobis_distance'] > threshold, 'Si', 'No'
    )
    

    logger.info(f"Created {len(scaled)} feature dictionaries")

     # 2. DEFINIR summary_data con metadatos simples
    summary_data = [
        {"Métrica": "Número de Filas", "Valor": len(scaled)},
        {"Métrica": "Número de Columnas", "Valor": len(scaled.columns)},
        {"Métrica": "Tamaño (KB)", "Valor": f"{scaled.memory_usage(index=False, deep=True).sum() / 1024:.2f}"}
    ]

    create_table_artifact(
        key=f"data-summary-amico-scaled",
        table=summary_data,
        description=f"Data summary for AMICO Scaled"
    )
    return scaled

@task(name="create_train_test", description="Create set train and test Amico")
def create_train_test(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create X_train, X_test, y_train, y_test from DataFrame.

    Args:
        df: Input DataFrame (Se asume que tiene el índice de fecha y la columna 'is_outlier_mahalanobis').
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) como arrays de numpy.
    """
    logger = get_run_logger()
    
    # Usamos df que es el argumento de entrada, no una variable 'scaled' o 'self.df'
    df_local = df.copy() 

    #Asegurar que el índice es de tipo datetime
    if not isinstance(df_local.index, pd.DatetimeIndex):
        try:
            df_local.index = pd.to_datetime(df_local.index)
        except Exception as e:
            logger.error(f"Failed to convert index to datetime: {e}")
            raise ValueError("DataFrame index must be convertible to datetime for time series split.")
    # Define el número de días para la prueba y los días que deseas para el entrenamiento
    TEST_DAYS = 90
    TRAIN_DAYS_WINDOW = 90  # Días del entrenamiento antes de la fecha de corte

    # ---------- 6. División train/test temporal y ventana de entrenamiento ----------

    # 6.1. Calcular la fecha de división (inicio del conjunto de prueba)
    # Se utiliza df_local.index.max() ya que la columna Service fue renombrada a 'date' y se hizo index
    split_date = df_local.index.max() - pd.Timedelta(days=TEST_DAYS)

    # 6.2. Calcular la fecha de inicio del entrenamiento
    train_start_date = split_date - pd.Timedelta(days=TRAIN_DAYS_WINDOW)
    
    # 6.3. Filtrar el conjunto de entrenamiento (dentro de la ventana y sin outliers)
    train_df = df_local[
        (df_local.index >= train_start_date) &
        (df_local.index < split_date) &
        (df_local['is_outlier_mahalanobis'] == 'No')
    ].drop(columns=['day_of_week'])

    # 6.4. Filtrar el conjunto de prueba (sin cambios)
    test_df = df_local[
        df_local.index >= split_date
    ].drop(columns=['day_of_week'])

    # Columnas a excluir para obtener el conjunto de features (X)
    cols_to_drop = [
        'is_outlier_mahalanobis', 
        'mahalanobis_distance',
        'FSx($)', 
        'Elastic Load Balancing($)', 
        'Resilience Hub($)',
        'DataSync($)',
        'Secrets Manager($)'
    ]

    # 6.5. Separar variables predictoras (X) y variable objetivo (y)
    X_train = train_df.drop(columns=cols_to_drop).values
    y_train = train_df['is_outlier_mahalanobis'].values
    
    X_test = test_df.drop(columns=cols_to_drop).values
    y_test = test_df['is_outlier_mahalanobis'].values
    
    logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")

    # 2. DEFINIR summary_data con métricas de Train/Test
    # ✅ Métrica modificada para reflejar el tamaño de los sets
    summary_data = [
        {"Métrica": "Día de Corte", "Valor": split_date.strftime('%Y-%m-%d')},
        {"Métrica": "Filas X_train", "Valor": X_train.shape[0]},
        {"Métrica": "Columnas X_train", "Valor": X_train.shape[1]},
        {"Métrica": "Filas X_test", "Valor": X_test.shape[0]},
        {"Métrica": "Columnas X_test", "Valor": X_test.shape[1]},
    ]

    create_table_artifact(
        key="data-summary-amico-train-test",
        table=summary_data,
        description="Data summary for AMICO Train/Test Split"
    )

    return X_train, X_test, y_train, y_test

@task(name="train_iso_forest", description="Train IsolationForest with MLflow tracking")
def train_iso_forest(X_train, y_train, X_test, y_test) -> str:

    logger = get_run_logger()
    models_folder = Path("models")
    models_folder.mkdir(exist_ok=True)

    logger.info(f"Entrenando IsolationForest con {X_train.shape[0]} samples")

    # FIX: nested=True
    with mlflow.start_run(nested=True) as run:

        params = {
            "n_estimators": 10,
            "max_samples": 0.29775849710689867,
            "contamination": 0.0717176502157413,
            "max_features": 0.5310605708595397,
            "random_state": 42,
            "bootstrap": False
        }

        mlflow.log_params(params)

        iso_forest = IsolationForest(
            **params,
            n_jobs=-1
        ).fit(X_train)

        mapeo_etiquetas = {"No": 0, "Si": 1}
        y_test_numerica = pd.Series(y_test).map(mapeo_etiquetas).astype(int)

        prediction = iso_forest.predict(X_test)
        pred = np.where(prediction == -1, 1, 0)

        report = classification_report(y_test_numerica, pred, output_dict=True)
        f1 = report["1"]["f1-score"]
        precision = report["1"]["precision"]
        recall = report["1"]["recall"]
        accuracy = report["accuracy"]

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_outliers", precision)
        mlflow.log_metric("recall_outliers", recall)
        mlflow.log_metric("f1_outliers", f1)

        model_path = "models/iso_forest.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(iso_forest, f)

        try:
            mlflow.log_artifact(model_path, artifact_path="iso_forest")
            logger.info("Modelo IsolationForest loggeado en MLflow")
        except Exception as e:
            logger.warning(f"No se pudo loggear a MLflow: {e}")

        metrics_table = [
            ["Accuracy", f"{accuracy:.4f}"],
            ["Precision (outlier=1)", f"{precision:.4f}"],
            ["Recall (outlier=1)", f"{recall:.4f}"],
            ["F1 (outlier=1)", f"{f1:.4f}"],
            ["MLflow Run ID", run.info.run_id],
        ]

        create_table_artifact(
            key="iso-forest-performance",
            table=metrics_table,
            description="Métricas de IsolationForest"
        )

        md = f"""
        # IsolationForest Training Summary

        ## Performance
        - Accuracy: {accuracy:.4f}
        - Precision (outliers): {precision:.4f}
        - Recall (outliers): {recall:.4f}
        - F1 (outliers): {f1:.4f}
        - MLflow Run ID: {run.info.run_id}
        """

        create_markdown_artifact(
            key="iso-forest-summary",
            markdown=md,
            description="Resumen entrenamiento IsolationForest"
        )

        return run.info.run_id

@flow(name="AMICO Prediction Pipeline", description="End-to-end ML pipeline for AWS cost prediction")
def amico_prediction_flow(s3_url: str) -> str:
    """
    Main flow AMICO prediction.

    Args:
        s3_url : Url del bucket de S3

    Returns:
        MLflow run ID
    """

    with mlflow.start_run() as active_run:
        run_id = active_run.info.run_id  # <--- run_id AHORA ESTÁ DEFINIDO aquí
        
        #1. Load training data
        df_train = read_dataframe(s3_url=s3_url)

        #2. Create features
        df_scaled = create_features(df_train)
        #3. Create features
        X_train, X_test, y_train, y_test=create_train_test(df_scaled)
        
        # 4. Entrenar IsolationForest con tracking
        iso_forest_run_id = train_iso_forest(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        # Create final pipeline artifact
        pipeline_summary = f"""
        # Pipeline Execution Summary

        ## Data
        - **Training S3Url**: {s3_url}
        
        ## Results
        - **MLflow Run ID**: {run_id}
        - **MLflow Experiment**: nyc-taxi-experiment-prefect

        ## Next Steps
        1. Review model performance in MLflow UI: http://localhost:5000
        2. Compare with previous runs
        3. Consider model deployment if performance is satisfactory
        """

        create_markdown_artifact(
            key="pipeline-summary",
            markdown=pipeline_summary,
            description="Complete pipeline execution summary"
        )

        return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict cost aws whit Prefect.')
    parser.add_argument('--s3_url', type=str, help='Url donde esta almacenada la informacion de los costos')
    parser.add_argument('--mlflow-uri', type=str, help='MLflow tracking URI (overrides environment variable)')
    args = parser.parse_args()

    # Override MLflow URI if provided
    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
        setup_mlflow()

    try:
        # Run the flow
        run_id = amico_prediction_flow(s3_url=args.s3_url)
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 MLflow run_id: {run_id}")
        print(f"🔗 View results at: {mlflow.get_tracking_uri()}")

        # Save run ID for reference
        with open("prefect_run_id.txt", "w") as f:
            f.write(run_id)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

