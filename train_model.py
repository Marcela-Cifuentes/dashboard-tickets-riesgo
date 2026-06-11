"""
train_model.py — Entrenamiento del modelo mejorado (v2 corregido)
==================================================================
Ejecutar:  python train_model.py

CAUSA DEL ERROR ANTERIOR:
  HistGradientBoostingClassifier (scikit-learn ≥ 1.4) no acepta matrices
  sparse. TfidfVectorizer produce matrices sparse. La solución es interponer
  TruncatedSVD (LSA) que: (a) convierte a densa, (b) reduce dimensionalidad
  de 15.000 → 200 componentes, (c) captura semántica latente (sinónimos).

ARQUITECTURA FINAL:
  Texto  → TF-IDF (sparse) → TruncatedSVD/LSA (densa, 200 dims)
  ↓ concatenar con NumPy
  Categóricas → OrdinalEncoder (densa)
  Numéricas   → ya densas
  ↓
  HistGradientBoostingClassifier (requiere entrada densa) ✅

VENTAJAS SOBRE LOGREG ORIGINAL:
  - Captura relaciones no lineales y entre features
  - SVD/LSA agrupa términos similares (vpn ≈ red, contraseña ≈ acceso)
  - Features temporales y de historial del agente
  - early stopping automático anti-overfitting
  - class_weight='balanced' para el desbalanceo (15% positivos)
  - Split temporal (no aleatorio) → sin data leakage
  - Búsqueda de hiperparámetros con RandomizedSearchCV

Genera:
  modelo_logreg.pkl     (pipeline completo: preprocesador + clasificador)
  vectorizer.pkl        (TF-IDF, guardado por compatibilidad)
  encoder.pkl           (OrdinalEncoder, guardado por compatibilidad)
  modelo_metadata.json  (métricas reales del conjunto de test)
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import pipeline as pl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════

BASES_ENTRENAMIENTO = list(pl.URLS_BASES.keys())
TEST_SIZE_RATIO     = 0.20
RANDOM_STATE        = 42
N_ITER_SEARCH       = 15      # reducido para velocidad; sube a 30 si tienes tiempo
CV_FOLDS            = 5
SVD_COMPONENTES     = 150     # dimensiones LSA (texto denso)

COLS_TEXT = "TEXTO_LIMPIO"
COLS_CAT  = ["PRIORIDAD", "GRUPO", "ORIGEN"]
COLS_NUM  = [
    "HORA_CREACION", "DIA_SEMANA", "MES_CREACION",
    "ES_FIN_SEMANA", "ES_HORA_PICO",
    "SCORE_SENT_FILL", "FLAG_URGENCIA", "FLAG_CONFLICTO",
    "HIST_AGENTE_AVG_DIAS",
]


# ═══════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════

def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = df.copy()

    # Temporales
    fe["HORA_CREACION"] = fe["CREACION"].dt.hour.fillna(12).astype(float)
    fe["DIA_SEMANA"]    = fe["CREACION"].dt.dayofweek.fillna(0).astype(float)
    fe["MES_CREACION"]  = fe["CREACION"].dt.month.fillna(1).astype(float)
    fe["ES_FIN_SEMANA"] = (fe["DIA_SEMANA"] >= 5).astype(float)
    fe["ES_HORA_PICO"]  = fe["HORA_CREACION"].between(8, 12).astype(float)

    # NLP scores
    if "SCORE_SENTIMIENTO" in fe.columns:
        fe["SCORE_SENT_FILL"] = fe["SCORE_SENTIMIENTO"].fillna(0.0).astype(float)
    else:
        fe["SCORE_SENT_FILL"] = 0.0

    # .astype(str) necesario: estas columnas pueden ser dtype 'category'
    urg_col  = fe["URGENCIA"].astype(str)  if "URGENCIA"  in fe.columns else pd.Series("Normal",  index=fe.index)
    conf_col = fe["CONFLICTO"].astype(str) if "CONFLICTO" in fe.columns else pd.Series("Normal",  index=fe.index)
    fe["FLAG_URGENCIA"]  = (urg_col  == "🔥 Alta urgencia").astype(float)
    fe["FLAG_CONFLICTO"] = (conf_col == "⚠️ Conflictivo").astype(float)

    # Historial agente (sin fuga: expanding mean del pasado)
    if "AGENTE" in fe.columns:
        fe = fe.sort_values("CREACION").copy()
        fe["HIST_AGENTE_AVG_DIAS"] = (
            fe.groupby("AGENTE", observed=True)["DIAS"]
            .expanding().mean().shift(1)
            .reset_index(level=0, drop=True)
            .fillna(fe["DIAS"].mean())
            .astype(float)
        )
    else:
        fe["HIST_AGENTE_AVG_DIAS"] = float(fe["DIAS"].mean())

    return fe


# ═══════════════════════════════════════════════
# PREPROCESADOR MANUAL (evita sparse en HistGBM)
# ═══════════════════════════════════════════════

# PreprocesadorDenso y ModeloPipeline definidas en pipeline.py
# para que joblib.load funcione correctamente en cualquier entorno.
from pipeline import (PreprocesadorDenso, ModeloPipeline,
                      COLS_MODEL_TEXT as COLS_TEXT,
                      COLS_MODEL_CAT  as COLS_CAT,
                      COLS_MODEL_NUM  as COLS_NUM,
                      SVD_COMPONENTES)


# ═══════════════════════════════════════════════
# CARGA Y SPLIT
# ═══════════════════════════════════════════════

def cargar_y_preparar() -> tuple[pd.DataFrame, pd.Series]:
    frames = []
    for base in BASES_ENTRENAMIENTO:
        log.info("Cargando base: %s", base)
        try:
            df = pl.procesar_base(base)
            df = construir_features(df)
            frames.append(df)
        except Exception as e:
            log.error("Error cargando %s: %s", base, e)

    if not frames:
        raise RuntimeError("No se pudo cargar ninguna base.")

    datos = (pd.concat(frames, ignore_index=True)
               .sort_values("CREACION")
               .reset_index(drop=True))

    y = datos["RIESGO_OPERATIVO"].astype(int)

    # Garantizar todas las columnas necesarias
    for col in COLS_NUM:
        if col not in datos.columns:
            datos[col] = 0.0
    if COLS_TEXT not in datos.columns:
        datos[COLS_TEXT] = datos.get("TEXTO_COMPLETO", pd.Series("")).fillna("")

    return datos, y


def split_temporal(datos, y):
    n       = len(datos)
    n_train = int(n * (1 - TEST_SIZE_RATIO))
    log.info("Train: %d | Test: %d | Balance train: %.1f%% positivos",
             n_train, n - n_train, y.iloc[:n_train].mean() * 100)
    return (datos.iloc[:n_train], datos.iloc[n_train:],
            y.iloc[:n_train],     y.iloc[n_train:])


# ═══════════════════════════════════════════════
# BÚSQUEDA DE HIPERPARÁMETROS (manual, no sklearn CV)
# porque el preprocesador es custom y RandomizedSearchCV
# necesita estimadores sklearn estándar.
# Usamos validación cruzada manual con StratifiedKFold.
# ═══════════════════════════════════════════════

PARAM_GRID = [
    {"lr": 0.05,  "max_iter": 300, "max_depth": 4,    "min_leaf": 20},
    {"lr": 0.10,  "max_iter": 300, "max_depth": 5,    "min_leaf": 20},
    {"lr": 0.05,  "max_iter": 500, "max_depth": 5,    "min_leaf": 30},
    {"lr": 0.10,  "max_iter": 200, "max_depth": 4,    "min_leaf": 10},
    {"lr": 0.15,  "max_iter": 200, "max_depth": 3,    "min_leaf": 20},
    {"lr": 0.05,  "max_iter": 300, "max_depth": 6,    "min_leaf": 30},
    {"lr": 0.10,  "max_iter": 400, "max_depth": None, "min_leaf": 20},
    {"lr": 0.08,  "max_iter": 300, "max_depth": 4,    "min_leaf": 15},
]


def cv_score(params: dict, X_tr: pd.DataFrame, y_tr: pd.Series) -> float:
    """AUC medio en K-fold para una combinación de hiperparámetros."""
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)
    scores = []

    for fold_tr, fold_val in skf.split(X_tr, y_tr):
        prep = PreprocesadorDenso()
        clf  = HistGradientBoostingClassifier(
            learning_rate       = params["lr"],
            max_iter            = params["max_iter"],
            max_depth           = params["max_depth"],
            min_samples_leaf    = params["min_leaf"],
            early_stopping      = True,
            validation_fraction = 0.1,
            n_iter_no_change    = 10,
            scoring             = "roc_auc",
            class_weight        = "balanced",
            random_state        = RANDOM_STATE,
        )
        pipe = ModeloPipeline(prep, clf)
        pipe.fit(X_tr.iloc[fold_tr], y_tr.iloc[fold_tr])

        y_score = pipe.predict_proba(X_tr.iloc[fold_val])[:, 1]
        scores.append(roc_auc_score(y_tr.iloc[fold_val], y_score))

    return float(np.mean(scores))


# ═══════════════════════════════════════════════
# ENTRENAMIENTO PRINCIPAL
# ═══════════════════════════════════════════════

def entrenar() -> None:
    log.info("═══ Inicio del entrenamiento ═══")

    datos, y = cargar_y_preparar()
    X_train, X_test, y_train, y_test = split_temporal(datos, y)

    # — Búsqueda de hiperparámetros —
    log.info("Evaluando %d combinaciones con %d-fold CV...",
             len(PARAM_GRID), CV_FOLDS)

    mejor_auc    = -1.0
    mejores_params = PARAM_GRID[0]

    for i, params in enumerate(PARAM_GRID):
        auc_cv = cv_score(params, X_train, y_train)
        log.info("  [%d/%d] lr=%.2f depth=%s leaf=%d → AUC-CV=%.4f",
                 i + 1, len(PARAM_GRID),
                 params["lr"], params["max_depth"], params["min_leaf"], auc_cv)
        if auc_cv > mejor_auc:
            mejor_auc    = auc_cv
            mejores_params = params

    log.info("Mejores params: %s  (AUC-CV=%.4f)", mejores_params, mejor_auc)

    # — Entrenar modelo final con todos los datos de train —
    log.info("Entrenando modelo final...")
    prep_final = PreprocesadorDenso()
    clf_final  = HistGradientBoostingClassifier(
        learning_rate       = mejores_params["lr"],
        max_iter            = mejores_params["max_iter"],
        max_depth           = mejores_params["max_depth"],
        min_samples_leaf    = mejores_params["min_leaf"],
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 15,
        scoring             = "roc_auc",
        class_weight        = "balanced",
        random_state        = RANDOM_STATE,
    )
    modelo_final = ModeloPipeline(prep_final, clf_final)
    modelo_final.fit(X_train, y_train)

    # — Evaluación en test (datos nunca vistos) —
    log.info("Evaluando en conjunto de test...")
    y_score = modelo_final.predict_proba(X_test)[:, 1]
    y_pred  = (y_score > 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc  = average_precision_score(y_test, y_score)
    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    log.info("AUC-ROC test : %.4f", roc_auc)
    log.info("AUC-PR  test : %.4f", pr_auc)
    log.info("F1  (riesgo) : %.4f", reporte.get("1", {}).get("f1-score", 0))
    log.info("Precisión    : %.4f", reporte.get("1", {}).get("precision", 0))
    log.info("Recall       : %.4f", reporte.get("1", {}).get("recall", 0))

    # — Guardar artefactos —
    joblib.dump(modelo_final,              "modelo_logreg.pkl",   compress=3)
    joblib.dump(prep_final.tfidf,          "vectorizer.pkl",      compress=3)
    joblib.dump(prep_final.encoder,        "encoder.pkl",         compress=3)

    metadata = {
        "algoritmo":    "HistGradientBoostingClassifier + TF-IDF + SVD/LSA",
        "version":      "2.0",
        "svd_dims":     SVD_COMPONENTES,
        "features_num": COLS_NUM,
        "features_cat": COLS_CAT,
        "mejores_params": {k: str(v) for k, v in mejores_params.items()},
        "metricas_cv":  {"auc_roc_cv": round(mejor_auc, 4)},
        "metricas_test": {
            "auc_roc":          round(roc_auc, 4),
            "auc_pr":           round(pr_auc, 4),
            "f1_clase_riesgo":  round(reporte.get("1",{}).get("f1-score",0),   4),
            "precision_riesgo": round(reporte.get("1",{}).get("precision",0),  4),
            "recall_riesgo":    round(reporte.get("1",{}).get("recall",0),     4),
        },
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
    }
    Path("modelo_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Archivos generados:")
    log.info("  modelo_logreg.pkl  → pipeline completo (preprocesador + GBM)")
    log.info("  vectorizer.pkl     → TF-IDF (compatibilidad)")
    log.info("  encoder.pkl        → OrdinalEncoder (compatibilidad)")
    log.info("  modelo_metadata.json")
    log.info("═══ Entrenamiento completo ═══")


if __name__ == "__main__":
    entrenar()
