#!/usr/bin/env python3
"""
Script para extraer datos de Costos desde AWS S3.
"""

import os
import pandas as pd

# Importar 's3fs' es crucial para que pandas pueda leer desde S3.
import s3fs 


class GetData:
    def __init__(self, s3_url="s3://amico-udem/DataModels/costs.csv"):
        self.s3_url = s3_url
    
    def download_createdf(self):
        # ---------- 0. configuración ----------
        # Este método es para carga local (como backup o alternativa)
        CSV_PATH = "/Users/angeleduardogamarrarios/Repositorio_UDEM/MLops_AMICO/data/costs.csv" 
        # ---------- 1. carga  ----------
        df = pd.read_csv(CSV_PATH)
        return df # CORREGIDO: Retornar el DataFrame, no None


    def download_data(self):
        print(f"La carga se realizará directamente desde S3: {self.s3_url}")
        print("Asegúrate de que tus credenciales de AWS estén configuradas "
              "(ej. variables de entorno AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, o ~/.aws/credentials).")

        return True

    def create_dataset(self):
        print("Iniciando la lectura del dataset desde S3...")
        
        try:
            df = pd.read_csv(self.s3_url)
            # CORREGIDO: Usar 'df' en lugar de 'df_costs'
            print(f"Lectura exitosa. DataFrame cargado con {len(df)} filas.") 
            return df # CORREGIDO: Retornar 'df'

        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo en la URL de S3: {self.s3_url}")
            return None
        except Exception as e:
            print(f"Ocurrió un error al leer desde S3. Revisa credenciales o permisos.")
            print(f"Detalle del error: {e}")
            return None