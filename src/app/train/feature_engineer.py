# Carga de librerias
# ------------------
# Librerias de uso general
import holidays
from google.cloud import storage

# Manejo de datos
import numpy as np
import pandas as pd

import kagglehub
import shutil
import os

import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import f1_score, make_scorer

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM  # Importación para OC-SVM
from sklearn.neighbors import LocalOutlierFactor # Importación para LOF

from sklearn.feature_extraction import DictVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.metrics import classification_report
from math import sqrt
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import chi2

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Estadística y series temporales
from statsmodels.tsa.seasonal import seasonal_decompose

# Se importan las funciones
from sklearn.metrics import  mean_absolute_error


class FeatureEngineer:
    def __init__(self, df):
        self.df = df
    
    def create_features(self):
        # eliminar fila de Service total y columna de total costs
        df = df[df['Service'] != 'Service total']
        df = df.drop('Total costs($)', axis=1)

        # convertir columna de fecha en indice
        df['Service'] = pd.to_datetime(df['Service'])
        df = df.rename(columns={'Service':'date'}).sort_values('date').set_index('date')

        # forzar numérico y revisar columnas
        df = df.apply(pd.to_numeric, errors='coerce')
        num_cols = df.columns.tolist()

        # ---------- 3. imputación ----------
        # Strategy: cambiar a 0 los valores nulos
        df_imputed = df.fillna(0)

        return df_imputed
    
    def create_features_bc(self):

        # ---------- Box-Cox selectivo para normalizar las varibles ----------
        # Aplicar Box-Cox solo a columnas muy sesgadas (skew > 1)
        skewed_cols = skewness[skewness > 1].index.tolist()
        df_bc = df_imputed.copy()

        for c in skewed_cols:
            # Box-Cox exige valores > 0. si hay ceros o negativos shift pequeño
            min_val = df_bc[c].min()
            shift = 0.0 if min_val > 0 else abs(min_val) + 1e-6
            try:
                transformed, lam = boxcox(df_bc[c] + shift)
                df_bc[c] = transformed
                print(f"Box-Cox aplicado a {c}, lambda={lam:.4f}")
            except Exception as e:
                # fallback log1p si falla
                df_bc[c] = np.log1p(df_bc[c] + shift)
                print(f"Box-Cox falló en {c}, aplicado log1p")

        return df_bc    
    
    def create_features_day(self):
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

        return datos_aws_day 
    
    def create_features_etiquetaol(self):
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
        return scaled 