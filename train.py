"""Entrenamiento del modelo base (UCI Adult) con registro en MLflow Tracking.

Uso:
    python train.py                                  # valores por defecto
    python train.py --C 0.1 --l1-ratio 1 --class-weight balanced --run-name lasso_balanced
    python train.py --help                           # lista de argumentos
"""

import argparse
import io
import json
import logging
import os
import random
import tempfile
import urllib.request
import warnings
import zipfile
from pathlib import Path

os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             average_precision_score, balanced_accuracy_score,
                             classification_report, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logging.getLogger("mlflow").setLevel(logging.WARNING)

URL_UCI = "https://archive.ics.uci.edu/static/public/2/adult.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
COLUMNAS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "income"]
NUM_SIMPLES = ["age", "education_num", "hours_per_week"]
NUM_SESGADAS = ["capital_gain", "capital_loss"]
CATEGORICAS = ["workclass", "marital_status", "occupation", "relationship", "race", "sex",
               "native_country"]
COLS_DESCARTE = ["education", "fnlwgt", "income"]


def parse_args(argv=None):
    """Lee los hiperparámetros y la configuración desde la línea de comandos."""
    p = argparse.ArgumentParser(
        description="Entrena una regresión logística sobre UCI Adult y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    m = p.add_argument_group("hiperparámetros del modelo")
    m.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularización (menor = más regularización).")
    m.add_argument("--l1-ratio", type=float, default=0.0,
                   help="Mezcla de regularización: 0 = L2 (Ridge), 1 = L1 (Lasso), intermedio = Elastic Net.")
    m.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="Peso de las clases; 'balanced' compensa el desbalance 76/24.")
    m.add_argument("--max-iter", type=int, default=1000, help="Iteraciones máximas del optimizador.")
    m.add_argument("--min-frequency", type=int, default=50,
                   help="Casos mínimos para que una categoría tenga columna propia en el one-hot.")
    e = p.add_argument_group("datos y reproducibilidad")
    e.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba.")
    e.add_argument("--cv-folds", type=int, default=5, help="Folds de validación cruzada (0 = omitir).")
    e.add_argument("--seed", type=int, default=42, help="Semilla aleatoria global.")
    t = p.add_argument_group("MLflow")
    t.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
                   help="URI del servidor o base de MLflow.")
    t.add_argument("--experiment-name", default="TC5061_A03_AdultIncome_LogReg",
                   help="Experimento donde se agrupan las corridas.")
    t.add_argument("--run-name", default=None, help="Nombre de la corrida (por defecto se genera).")
    args = p.parse_args(argv)
    if not 0.0 <= args.l1_ratio <= 1.0:
        p.error("--l1-ratio debe estar entre 0 y 1")
    if args.C <= 0:
        p.error("--C debe ser mayor que 0")
    return args


def fijar_semilla(seed):
    """Fija la semilla en Python y NumPy; scikit-learn la recibe por random_state."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def cargar_datos(data_dir=DATA_DIR):
    """Descarga UCI Adult si hace falta y une adult.data con adult.test."""
    data_dir.mkdir(exist_ok=True)
    if not (data_dir / "adult.data").exists() or not (data_dir / "adult.test").exists():
        with urllib.request.urlopen(URL_UCI, timeout=60) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(data_dir)
    train = pd.read_csv(data_dir / "adult.data", names=COLUMNAS, skipinitialspace=True)
    test = pd.read_csv(data_dir / "adult.test", names=COLUMNAS, skipinitialspace=True, skiprows=1)
    return pd.concat([train, test], ignore_index=True)


def limpiar(df):
    """Aplica las decisiones de limpieza de la Actividad 2 y separa X, y."""
    df = df.copy()
    texto = df.select_dtypes(exclude="number").columns
    df[texto] = df[texto].apply(lambda s: s.str.strip())
    df["income"] = df["income"].str.rstrip(".")
    df["target"] = (df["income"] == ">50K").astype("int8")
    df[CATEGORICAS] = df[CATEGORICAS].replace("?", "Desconocido")
    df = df.drop_duplicates(subset=COLUMNAS).reset_index(drop=True)
    df = df.drop(columns=COLS_DESCARTE)
    X = df.drop(columns="target")
    X[NUM_SIMPLES + NUM_SESGADAS] = X[NUM_SIMPLES + NUM_SESGADAS].astype("float64")
    return X, df["target"]


def construir_modelo(args):
    """Pipeline de preprocesamiento + regresión logística con los hiperparámetros dados."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_SIMPLES),
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scaler", StandardScaler())]), NUM_SESGADAS),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=args.min_frequency,
                              sparse_output=False), CATEGORICAS),
    ])
    clf = LogisticRegression(
        C=args.C, l1_ratio=args.l1_ratio,
        solver="lbfgs" if args.l1_ratio == 0 else "saga",
        class_weight=None if args.class_weight == "none" else "balanced",
        max_iter=args.max_iter, random_state=args.seed)
    return Pipeline([("prep", prep), ("clf", clf)])


def evaluar(modelo, X, y):
    """Métricas de clasificación binaria (clase positiva: >50K)."""
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def guardar_artefactos(modelo, X_test, y_test, metricas, carpeta):
    """Escribe matriz de confusión, curva ROC, reporte y métricas en la carpeta dada."""
    carpeta = Path(carpeta)
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["<=50K", ">50K"],
                                            cmap="Blues", colorbar=False, ax=axes[0], values_format=",")
    axes[0].set_title("Matriz de confusión (prueba)")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1], name="LogReg")
    axes[1].plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    axes[1].set_title("Curva ROC (prueba)")
    fig.tight_layout()
    fig.savefig(carpeta / "evaluacion_test.png", dpi=120)
    plt.close(fig)

    reporte = classification_report(y_test, pred, target_names=["<=50K", ">50K"], digits=4)
    (carpeta / "classification_report.txt").write_text(reporte)
    (carpeta / "metrics.json").write_text(json.dumps(metricas, indent=2))
    return carpeta


def nombre_corrida(args):
    """Nombre legible a partir de los hiperparámetros."""
    return f"C{args.C:g}_l1r{args.l1_ratio:g}_cw-{args.class_weight}_mf{args.min_frequency}_s{args.seed}"


def main(argv=None):
    args = parse_args(argv)
    fijar_semilla(args.seed)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    X, y = limpiar(cargar_datos())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    modelo = construir_modelo(args)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name or nombre_corrida(args)) as run:
        mlflow.set_tags({"autor": "Jordan Piero Borda Colque (A01840539)", "curso": "TC5061",
                         "actividad": "A03", "modelo": "LogisticRegression"})

        # Parámetros
        for nombre, valor in vars(args).items():
            if nombre not in ("tracking_uri", "experiment_name", "run_name"):
                mlflow.log_param(nombre, valor)
        mlflow.log_param("solver", modelo.named_steps["clf"].solver)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        # Validación cruzada sobre entrenamiento
        if args.cv_folds > 1:
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
            cv_auc = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")
            mlflow.log_metric("cv_roc_auc_mean", cv_auc.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_auc.std())

        # Entrenamiento y métricas de prueba
        modelo.fit(X_train, y_train)
        metricas = evaluar(modelo, X_test, y_test)
        for nombre, valor in metricas.items():
            mlflow.log_metric(f"test_{nombre}", valor)
        mlflow.log_metric("train_roc_auc", evaluar(modelo, X_train, y_train)["roc_auc"])
        n_coef = int((modelo.named_steps["clf"].coef_ != 0).sum())
        mlflow.log_metric("n_coef_no_cero", n_coef)

        # Artefactos
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = guardar_artefactos(modelo, X_test, y_test, metricas, tmp)
            for archivo in sorted(carpeta.iterdir()):
                mlflow.log_artifact(str(archivo), artifact_path="evaluacion")

        # Modelo
        firma = infer_signature(X_train, modelo.predict(X_train))
        mlflow.sklearn.log_model(sk_model=modelo, name="modelo", signature=firma,
                                 input_example=X_train.head(3))

    print(f"Corrida: {run.info.run_name} ({run.info.run_id})")
    print(" | ".join(f"{k}={v:.4f}" for k, v in metricas.items()))
    return metricas


if __name__ == "__main__":
    main()
