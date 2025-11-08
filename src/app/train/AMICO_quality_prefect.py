#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import mlflow
from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from sklearn.ensemble import RandomForestClassifier

# Importa tus módulos locales
from etl import GetData 
from feature_engineer import FeatureEngineer # Se asume la existencia de este módulo
from train_with_mlflow_optuna import TrainMlflowOptuna

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuración de Parámetros Globales (Ajusta según tu nuevo dataset de Costos) ---
# Usamos un conjunto de características de ejemplo, debes ajustarlas a tus datos de costos
DEFAULT_NUMERIC_FEATURES = ['Relational Database Service($)', 'EC2-Instances($)', 'S3($)', 'CloudWatch($)']
DEFAULT_TARGET_COLUMN = 'Backup($)' # Usaremos 'Backup($)' como objetivo de predicción (ejemplo)

# MLflow configuration with fallback
def setup_mlflow():
    """Setup MLflow with proper error handling and fallback options."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlops_costs.db")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.search_experiments()
        logger.info(f"Connected to MLflow at: {mlflow_uri}")
    except Exception as e:
        logger.warning(f"Failed to connect to {mlflow_uri}: {e}. Falling back to local SQLite.")
        mlflow.set_tracking_uri("sqlite:///mlops_costs.db")
    
    try:
        # Nuevo nombre de experimento enfocado en costos
        mlflow.set_experiment("aws-costs-prediction-prefect")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise


@task(name="load_data", description="Carga del set de datos de Costos AWS (Local/S3)", retries=3, retry_delay_seconds=10)
def read_dataframe() -> pd.DataFrame:
    """
    Carga del set de datos de Costos desde la ruta local o S3 configurada en etl.py.

    Returns:
        Dataframe de costos cargado.
    """
    logger = get_run_logger()

    # Usamos la clase GetData para cargar el DataFrame
    # Nota: Si usas S3, puedes inicializar con GetData(s3_url="...")
    get_data = GetData() 
    
    # Usamos el método corregido para cargar el DataFrame
    try:
        df = get_data.download_createdf() 
        logger.info(f"Carga satisfactoria. {len(df)} registros cargados.")
    except Exception as e:
        logger.error(f"Error al cargar los datos desde CSV: {e}")
        raise

    # Creación de artefacto con resumen de datos
    summary_data = [
        ["Total Records", len(df)],
        ["Target Column", DEFAULT_TARGET_COLUMN],
        ["Numeric Features", len(DEFAULT_NUMERIC_FEATURES)]
    ]

    create_table_artifact(
        key=f"data-summary",
        table=summary_data,
        description=f"Data summary for AWS Costs dataset (n={len(df)})"
    )

    return df


@task(name="create_features", description="Aplicar Feature Engineering (Normalización, Outliers, Box-Cox, etc.)")
def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list, list, str]:
    """
    Aplica el feature engineering, incluyendo normalización, Box-Cox y tratamiento de outliers.

    Args:
        df: Input DataFrame

    Returns:
        Tuple de (DataFrame con ingeniería aplicada, features numéricas, features categóricas, columna objetivo)
    """
    logger = get_run_logger()

    # --- Definición de Características ---
    numeric_features = DEFAULT_NUMERIC_FEATURES
    categorical_features = [] # Asumiendo que no hay categóricas en los datos de costos
    target_column = DEFAULT_TARGET_COLUMN

    logger.info(f"Iniciando Feature Engineering en {numeric_features}...")

    # Instanciar y aplicar Feature Engineer (Asumiendo que esta clase maneja la normalización/Box-Cox)
    # Debes implementar la clase FeatureEngineer en feature_engineer.py
    feature_engineer = FeatureEngineer(df)
    
    # Este método DEBE implementar y aplicar Normalización, Box-Cox, y Outlier treatment
    df_transformed = feature_engineer.fit_transform_data(numeric_features, target_column)

    logger.info(f"Feature Engineering completado. DataFrame transformado con {df_transformed.shape[1]} columnas.")

    # Guardar los objetos de transformación (scalers, lambdas, etc.) como artefactos de MLflow aquí

    return df_transformed, numeric_features, categorical_features, target_column


@task(name="train_model", description="Entrenar modelo con Optuna y MLflow")
def train_model(df, numeric_features, categorical_features, target_column) -> str:
    """
    Entrena RandomForestClassifier o un modelo de regresión si el objetivo es continuo.
    
    Nota: RandomForestClassifier se mantiene por compatibilidad, pero si 'Backup($)' es un valor continuo,
    deberías cambiarlo por un modelo de Regresión (ej. RandomForestRegressor).

    Returns:
        MLflow run ID
    """
    logger = get_run_logger()
    
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)
    
    # Parámetros Optuna (Manteniendo los de RandomForestClassifier de tu archivo original)
    param_distributions = {
        'n_estimators': ('int', 50, 200),
        'max_depth': ('int', 5, 30),
        'min_samples_split': ('int', 2, 10),
        'min_samples_leaf': ('int', 1, 5),
        'max_features': ('categorical', ['sqrt', 'log2', None])
    }
    
    # Inicializar el Trainer
    # Nota: Si el problema es de Regresión, el 'optimization_metric' debe ser 'rmse' o 'mae'
    trainer = TrainMlflowOptuna(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_column,
        model_class=RandomForestClassifier, # Cuidado si 'Backup($)' es continuo!
        test_size=0.3,
        n_trials=30,
        optimization_metric='f1', 
        param_distributions=param_distributions,
        model_params={'random_state': 42, 'n_jobs': -1},
        mlflow_setup = mlflow
    )

    best_pipeline, run_id, study = trainer.train()

    # ... (Resto del logging de artefactos de Prefect, no se modifican)
    
    logger.info(f"Training complete! MLflow Run ID: {run_id}")

    return run_id


@flow(name="AWS Costs Prediction Pipeline", description="End-to-end ML pipeline for AWS Cost Prediction")
def aws_costs_prediction_flow() -> str:
    """
    Flujo principal para la predicción de costos AWS.

    Returns:
        MLflow run ID
    """
    # 1. Cargar datos
    df = read_dataframe()

    # 2. Ingeniería de Características (Donde ocurre Normalización, Box-Cox, Outliers)
    df_transformed, numeric_features, categorical_features, target_column = create_features(df)

    # 3. Entrenar modelo
    run_id = train_model(df_transformed, numeric_features, categorical_features, target_column)

    # 4. Crear artefacto de resumen final
    pipeline_summary = f"""
    # Pipeline Execution Summary - AWS Costs

    ## Data Source
    - **Source**: Local CSV or S3 (via etl.py)
    - **Target**: {target_column}

    ## Results
    - **MLflow Run ID**: {run_id}
    - **MLflow Experiment**: aws-costs-prediction-prefect
    """

    create_markdown_artifact(
        key="pipeline-summary",
        markdown=pipeline_summary,
        description="Complete pipeline execution summary"
    )

    return run_id


if __name__ == "__main__":
    setup_mlflow()

    try:
        run_id = aws_costs_prediction_flow()
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 MLflow run_id: {run_id}")
        print(f"🔗 View results at: {mlflow.get_tracking_uri()}")

        with open("prefect_run_id.txt", "w") as f:
            f.write(run_id)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # sys.exit(1) # Reemplaza raise con sys.exit(1) si estás en un script de orquestación
        raise