### Paso 2: Iniciar Prefect (en otra terminal)


**🖥️ Terminal 1 - Prefect:**

```shell
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
cd /Users/angeleduardogamarrarios/Repositorio_UDEM/MLops_AMICO/src/app/Prefect-pipelines
uv run prefect server start
```

*Deja esta terminal corriendo - verás logs de Prefect aquí*

* [ ]  Paso 3: Ejecutar el pipeline (en nueva terminal)

**🖥️ Terminal 2 - Pipeline:**

```shell
cd /Users/angeleduardogamarrarios/Repositorio_UDEM/MLops_AMICO/src/app/Prefect-pipelines

# Configurar Prefect: para añadir variable de enterno
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Ejecutar con datos con la Url de S3
uv run python AMICO_prefect.py --s3_url "s3://amico-udem/DataModels/costs.csv"
```
