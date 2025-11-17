from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
# Manejo de datos
import numpy as np
import pandas as pd


class Train:
    def __init__(self, df):
        self.df = df
        
    def train_test_split(self):
        # Define el número de días para la prueba y los días que deseas para el entrenamiento
        TEST_DAYS = 90
        TRAIN_DAYS_WINDOW = 90  # Días del entrenamiento desde el 1 de diciembre del 2024

        # ---------- 6. División train/test temporal y ventana de entrenamiento ----------

        # 6.1. Calcular la fecha de división (inicio del conjunto de prueba)
        split_date = self.df.index.max() - pd.Timedelta(days=TEST_DAYS)

        # 6.2. Calcular la fecha de inicio del entrenamiento
        train_start_date = split_date - pd.Timedelta(days=TRAIN_DAYS_WINDOW)
        # Solo los días dentro de la ventana y sin outliers
        train_df = self.df[
            (self.df.index >= train_start_date) &
            (self.df.index < split_date) &
            (self.df['is_outlier_mahalanobis'] == 'No')
        ].drop(columns=['day_of_week'])

        # 6.4. Filtrar el conjunto de prueba (sin cambios)
        test_df = self.df[
            self.df.index >= split_date
        ].drop(columns=['day_of_week'])

        # 6.5. Separar variables predictoras (X) y variable objetivo (y)
        X_train = train_df.drop(columns=['is_outlier_mahalanobis','mahalanobis_distance','FSx($)', 'Elastic Load Balancing($)', 'Resilience Hub($)','DataSync($)','Secrets Manager($)']).values
        y_train = train_df['is_outlier_mahalanobis'].values

        X_test = test_df.drop(columns=['is_outlier_mahalanobis','mahalanobis_distance','FSx($)', 'Elastic Load Balancing($)', 'Resilience Hub($)','DataSync($)','Secrets Manager($)']).values
        y_test = test_df['is_outlier_mahalanobis'].values

        # 6.6. Obtener nombres de columnas
        cols = train_df.drop(columns=['is_outlier_mahalanobis','mahalanobis_distance','FSx($)', 'Elastic Load Balancing($)', 'Resilience Hub($)','DataSync($)','Secrets Manager($)']).columns.tolist()

        return X_train, X_test, y_train, y_test