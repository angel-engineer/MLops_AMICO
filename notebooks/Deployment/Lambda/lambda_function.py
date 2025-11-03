import json
import boto3
import pickle
import io
import pandas as pd


# Inicializar cliente S3
s3 = boto3.client('s3')

# Cargar modelo al inicio (solo una vez por ejecución fría)
BUCKET_NAME = "amico-udem"
MODEL_PATH = "ModelRepository/amico.pkl"
MODEL_KEY = 'amico.pkl'

def load_model():
    """
    Carga el modelo desde S3 o desde el archivo local (si ya está copiado)
    """
    s3.download_file(BUCKET_NAME, MODEL_KEY, MODEL_PATH)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


def lambda_handler(event, context):
    """
    Función principal ejecutada por AWS Lambda
    """
    try:
        # Obtener información del evento S3
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print(f"Archivo recibido: s3://{bucket}/{key}")

        # Descargar archivo CSV en memoria
        response = s3.get_object(Bucket=bucket, Key=key)
        csv_data = response['Body'].read()

        # Cargar DataFrame
        df = pd.read_csv(io.BytesIO(csv_data))
        print(f"DataFrame cargado con {len(df)} filas y {len(df.columns)} columnas")

        # Cargar modelo
        model = load_model()

        # Procesar el modelo
        predictions = model.predict(df)

        # Puedes guardar las predicciones en un archivo o enviarlas a otro servicio
        output_df = df.copy()
        output_df['prediction'] = predictions

        # Guardar resultado en S3
        output_buffer = io.StringIO()
        output_df.to_csv(output_buffer, index=False)
        result_key = f"results/{os.path.basename(key).replace('.csv', '_predictions.csv')}"
        s3.put_object(Bucket=bucket, Key=result_key, Body=output_buffer.getvalue())

        print(f"Resultados guardados en s3://{bucket}/{result_key}")

        return {
            'statusCode': 200,
            'body': json.dumps(f"Archivo procesado correctamente: {result_key}")
        }

    except Exception as e:
        print(f"Error procesando archivo: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }
