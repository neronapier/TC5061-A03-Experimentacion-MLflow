# TC5061 - Actividad 3: experimentación con MLflow

Jordan Piero Borda Colque (A01840539). Curso TC5061 Operaciones de Aprendizaje Automático (MLOps).

Script de entrenamiento parametrizado para el modelo base de la Actividad 2: una regresión logística que
predice si una persona gana más de 50K USD al año (dataset UCI Adult). Cada ejecución se registra en
MLflow Tracking.

## Estructura

| Archivo | Contenido |
|---------|-----------|
| `train.py` | Carga, limpieza, entrenamiento, evaluación y registro en MLflow |
| `scripts/run_experimentos.sh` | Las 10 corridas del experimento |
| `notebooks/eda_adult.ipynb` | Solo exploración de datos |
| `requirements.txt` | Dependencias con versión fija |
| `doc/` | Informe en LaTeX y PDF |
| `evidencia/` | Capturas de MLflow y salida de las corridas |

## Instalación

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El dataset se descarga de UCI la primera vez que se ejecuta `train.py` y se guarda en `data/`.

## Ejecución

Con los valores por defecto:

```bash
python train.py
```

Cambiando hiperparámetros:

```bash
python train.py --C 0.01 --l1-ratio 1 --class-weight balanced --seed 42 --run-name mi_corrida
```

Todas las corridas del experimento:

```bash
./scripts/run_experimentos.sh
```

Ver los resultados en la interfaz de MLflow (http://127.0.0.1:5000):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Servidor de MLflow del curso (VPS)

Las corridas entregadas se registraron en el MLflow del VPS (Docker: MLflow 3.16.1 + MinIO para artefactos).
Interfaz pública de solo lectura: https://161.132.54.170 (experimento `TC5061_A03_AdultIncome_LogReg`).

Para registrar en un servidor basta con indicar su URI:

```bash
MLFLOW_TRACKING_URI=http://<servidor>:5000 python train.py
```

En el VPS las corridas se ejecutan dentro del contenedor `runner`:

```bash
cd /opt/tc5061
docker compose --profile run run --rm -T -w /work/actividad3 -e PY=python runner \
    bash scripts/run_experimentos.sh
```

## Argumentos

| Argumento | Defecto | Descripción |
|-----------|---------|-------------|
| `--C` | 1.0 | Inverso de la regularización (menor = más regularización) |
| `--l1-ratio` | 0.0 | 0 = L2, 1 = L1, intermedio = Elastic Net |
| `--class-weight` | none | `none` o `balanced` |
| `--max-iter` | 1000 | Iteraciones máximas del optimizador |
| `--min-frequency` | 50 | Casos mínimos para que una categoría tenga columna propia |
| `--test-size` | 0.2 | Proporción de prueba |
| `--cv-folds` | 5 | Folds de validación cruzada (0 = omitir) |
| `--seed` | 42 | Semilla aleatoria |
| `--tracking-uri` | `sqlite:///mlflow.db` | También se toma de `MLFLOW_TRACKING_URI` |
| `--experiment-name` | `TC5061_A03_AdultIncome_LogReg` | Experimento de MLflow |
| `--run-name` | generado | Nombre de la corrida |

`python train.py --help` muestra la misma ayuda.

## Qué se registra en MLflow

- Parámetros: todos los argumentos, el solver y el tamaño de entrenamiento y prueba.
- Métricas: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC y PR-AUC en prueba; ROC-AUC de
  validación cruzada y de entrenamiento; número de coeficientes distintos de cero.
- Artefactos (`evaluacion/`): matriz de confusión y curva ROC, reporte de clasificación y `metrics.json`.
- Modelo: pipeline completo de scikit-learn con `mlflow.sklearn.log_model`, con firma y ejemplo de entrada.

## Reproducibilidad

- `--seed` fija `random`, `numpy` y `PYTHONHASHSEED`, y se pasa como `random_state` a la división
  entrenamiento/prueba, a la validación cruzada y al modelo.
- Dos ejecuciones con los mismos argumentos dan las mismas métricas (corridas `01_baseline_l2_C1` y
  `10_baseline_repeticion`).
- Las versiones de las librerías están fijadas en `requirements.txt` (numpy 2.3.5 porque la CPU del VPS no
  soporta versiones más nuevas).
- Las métricas fueron idénticas en el VPS y en un entorno local nuevo instalado desde `requirements.txt`.
