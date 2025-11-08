# Definicion de funciones generales empleadas en el proyecto
import pandas as pd
import numpy as np

def reemplazar_nulos(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    if columna in df.columns:
        df[columna] = df[columna].fillna(0)
    else:
        print(f"Column '{columna}' not found in the DataFrame.")
    return df
