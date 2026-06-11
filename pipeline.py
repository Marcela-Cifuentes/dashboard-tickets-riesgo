"""
pipeline.py
===========
Lógica de procesamiento de datos PURA (sin Streamlit).

Es la única fuente de verdad del procesamiento. La consumen:
  - etl.py  -> ejecución programada cada hora (cron / Task Scheduler / GitHub Actions)
  - app.py  -> solo como fallback si aún no existe el parquet procesado

Optimizaciones respecto a la versión original:
  - Stopwords y regex se construyen UNA sola vez (antes: por cada fila).
  - URGENCIA, CONFLICTO y TIPO_INCIDENTE vectorizados con str.contains
    (antes: .apply fila a fila).
  - La inferencia del modelo (PROB_RIESGO) y las anomalías (IsolationForest)
    se calculan aquí, UNA vez por actualización, nunca en el flujo de la UI.
  - Se persiste TEXTO_LIMPIO para que el dashboard no vuelva a limpiar texto.
  - Columnas categóricas convertidas a dtype 'category' (menos memoria,
    groupby más rápido).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import hstack
from sklearn.ensemble import IsolationForest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ===============================
# CONFIGURACIÓN
# ===============================

URLS_BASES = {
    "TicketsMintic": "https://storage.googleapis.com/contenidos-etraining/HelpDesk/TT.xlsx",
    "TicketsEJRLB": "https://storage.googleapis.com/contenidos-etraining/HelpDesk/EJRLB.xlsx",
}

DATA_DIR = Path("data")

RUTA_MODELO = Path("modelo_logreg.pkl")
RUTA_VECTORIZER = Path("vectorizer.pkl")
RUTA_ENCODER = Path("encoder.pkl")

COLUMNAS_CATEGORICAS = ["GRUPO", "PRIORIDAD", "ORIGEN", "AGENTE", "TICKET_ESTADO"]

# Umbrales de negocio (SLA)
SLA_OK_DIAS = 3
SLA_RIESGO_DIAS = 5
SLA_CRITICO_DIAS = 7

# ===============================
# RECURSOS COMPILADOS UNA SOLA VEZ
# ===============================

PALABRAS_URGENTES = [
    "urgente", "urgencia", "no funciona", "error", "fallo", "caido", "caído",
    "no puedo", "bloqueado", "problema", "critico", "crítico", "ya",
    "inmediato", "necesito",
]

PALABRAS_CONFLICTO = [
    "no sirve", "sigue igual", "otra vez", "nadie responde", "muy mal",
    "no solucionan",
]

REGLAS_INCIDENTE = {
    "Acceso":  ["login", "acceso", "contraseña", "password"],
    "Correo":  ["correo", "email", "outlook"],
    "Red":     ["vpn", "internet", "red"],
    "Servidor": ["servidor", "server", "caido", "down"],
    "Software": ["instalar", "programa", "aplicacion"],
}


def _patron(palabras: list[str]) -> re.Pattern:
    """Compila una lista de palabras a un único regex insensible a mayúsculas."""
    return re.compile("|".join(re.escape(p) for p in palabras), re.IGNORECASE)


RE_URGENCIA = _patron(PALABRAS_URGENTES)
RE_CONFLICTO = _patron(PALABRAS_CONFLICTO)
RE_INCIDENTES = {tipo: _patron(p) for tipo, p in REGLAS_INCIDENTE.items()}

RE_DIGITOS = re.compile(r"\d+")
RE_PUNTUACION = re.compile(r"[^\w\s]")


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    """Stopwords en español, descargadas/construidas UNA sola vez por proceso."""
    try:
        return frozenset(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("spanish"))


@lru_cache(maxsize=1)
def get_analizador_sentimiento() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def cargar_modelo():
    """Carga los artefactos del modelo predictivo desde disco."""
    modelo = joblib.load(RUTA_MODELO)
    vectorizer = joblib.load(RUTA_VECTORIZER)
    encoder = joblib.load(RUTA_ENCODER)
    return modelo, vectorizer, encoder


# ===============================
# LIMPIEZA DE TEXTO
# ===============================

def limpiar_texto(texto: str) -> str:
    """Limpieza para UN texto (formulario de predicción individual)."""
    stop_words = get_stopwords()
    texto = RE_PUNTUACION.sub("", RE_DIGITOS.sub("", str(texto).lower()))
    return " ".join(p for p in texto.split() if p not in stop_words and len(p) > 2)


def limpiar_serie_texto(serie: pd.Series) -> pd.Series:
    """Limpieza vectorizada para toda una columna de texto."""
    stop_words = get_stopwords()
    s = (
        serie.fillna("")
        .str.lower()
        .str.replace(RE_DIGITOS, "", regex=True)
        .str.replace(RE_PUNTUACION, "", regex=True)
    )
    return s.map(
        lambda t: " ".join(p for p in t.split() if p not in stop_words and len(p) > 2)
    )


# ===============================
# CARGA Y LIMPIEZA BASE
# ===============================

def cargar_excel(nombre_base: str) -> pd.DataFrame:
    """Descarga el Excel crudo y construye las columnas derivadas básicas."""
    url = URLS_BASES[nombre_base]
    df = pd.read_excel(url)

    df["CREACION"] = pd.to_datetime(df["CREACION"], errors="coerce")
    df["FECHA_RESPUESTA"] = pd.to_datetime(df["FECHA_RESPUESTA"], errors="coerce")

    df["TIEMPO_HORAS"] = (df["FECHA_RESPUESTA"] - df["CREACION"]).dt.total_seconds() / 3600
    df = df[df["TIEMPO_HORAS"] >= 0]

    df["DIAS"] = (df["TIEMPO_HORAS"] / 24).round(2)
    df = df.dropna(subset=["DIAS"]).copy()

    df["RIESGO_OPERATIVO"] = (df["DIAS"] > SLA_RIESGO_DIAS).astype(int)
    df["DEMORA_CRITICA"] = (df["DIAS"] > SLA_CRITICO_DIAS).astype(int)

    # --- Estado del ticket ---
    if "TICKET_ESTADO" in df.columns:
        df["TICKET_ESTADO"] = (
            df["TICKET_ESTADO"].astype(str).str.strip()
            .replace(["", "nan", "None", "null", "NULL"], "Sin revisar")
        )
        df["ESTADO_OPERATIVO"] = np.select(
            [
                df["TICKET_ESTADO"] == "Sin revisar",
                df["TICKET_ESTADO"] == "En Proceso",
                df["TICKET_ESTADO"] == "Escalado",
            ],
            ["🔴 Sin revisar", "🟡 En proceso", "🟣 Escalado"],
            default="🟢 Resuelto",
        )

    # --- Estado SLA ---
    df["ESTADO_SLA"] = np.select(
        [df["DIAS"] <= SLA_OK_DIAS, df["DIAS"] <= SLA_RIESGO_DIAS],
        ["🟢 Dentro SLA", "🟡 En riesgo"],
        default="🔴 Fuera SLA",
    )

    # --- Texto completo ---
    if "TICKET_ASUNTO" in df.columns and "TICKET_DESCRIPCION" in df.columns:
        df["TEXTO_COMPLETO"] = (
            df["TICKET_ASUNTO"].fillna("") + " " + df["TICKET_DESCRIPCION"].fillna("")
        )
    else:
        df["TEXTO_COMPLETO"] = ""

    return df


# ===============================
# ENRIQUECIMIENTO NLP (vectorizado)
# ===============================

def enriquecer_nlp(df: pd.DataFrame) -> pd.DataFrame:
    """Añade TIPO_INCIDENTE, URGENCIA, CONFLICTO, SENTIMIENTO y TEXTO_LIMPIO."""
    texto = df["TEXTO_COMPLETO"].fillna("")

    # Clasificación de incidente: np.select respeta el orden de prioridad
    condiciones = [texto.str.contains(pat, na=False) for pat in RE_INCIDENTES.values()]
    df["TIPO_INCIDENTE"] = np.select(condiciones, list(RE_INCIDENTES.keys()), default="Otro")

    df["URGENCIA"] = np.where(
        texto.str.contains(RE_URGENCIA, na=False), "🔥 Alta urgencia", "Normal"
    )
    df["CONFLICTO"] = np.where(
        texto.str.contains(RE_CONFLICTO, na=False), "⚠️ Conflictivo", "Normal"
    )

    # Texto limpio persistido: el dashboard nunca vuelve a limpiar texto
    df["TEXTO_LIMPIO"] = limpiar_serie_texto(texto)

    # Sentimiento (VADER). NOTA: VADER es un léxico en INGLÉS; sobre tickets en
    # español la señal es débil. Recomendado migrar a `pysentimiento` aquí
    # mismo (este es el único lugar que habría que tocar).
    analizador = get_analizador_sentimiento()

    def _sentimiento(t: str) -> str:
        score = analizador.polarity_scores(t)["compound"]
        if score >= 0.05:
            return "Positivo"
        if score <= -0.05:
            return "Negativo"
        return "Neutro"

    df["SENTIMIENTO"] = texto.map(_sentimiento)
    return df


# ===============================
# INFERENCIA DEL MODELO (batch)
# ===============================

def predecir_probabilidades(df: pd.DataFrame, modelo, vectorizer, encoder) -> pd.Series:
    """PROB_RIESGO para todo el dataset. Usa TEXTO_LIMPIO ya calculado."""
    X_text = vectorizer.transform(df["TEXTO_LIMPIO"].fillna(""))
    X_cat = encoder.transform(df[["PRIORIDAD", "GRUPO", "ORIGEN"]])
    X = hstack([X_text, X_cat])
    return pd.Series(modelo.predict_proba(X)[:, 1], index=df.index, name="PROB_RIESGO")


def predecir_ticket(modelo, vectorizer, encoder, asunto, descripcion,
                    prioridad, grupo, origen) -> tuple[float, str]:
    """Predicción para UN ticket nuevo (formulario del dashboard)."""
    texto_limpio = limpiar_texto(f"{asunto} {descripcion}")
    X_text = vectorizer.transform([texto_limpio])
    X_cat = encoder.transform(
        pd.DataFrame([{"PRIORIDAD": prioridad, "GRUPO": grupo, "ORIGEN": origen}])
    )
    proba = float(modelo.predict_proba(hstack([X_text, X_cat]))[0, 1])
    nivel = "Bajo" if proba < 0.35 else ("Medio" if proba < 0.65 else "Alto")
    return proba, nivel


# ===============================
# ANOMALÍAS
# ===============================

def detectar_anomalias(df: pd.DataFrame) -> pd.Series:
    """IsolationForest sobre DIAS. Se ejecuta solo en el ETL (una vez/hora)."""
    iso = IsolationForest(contamination=0.02, random_state=42)
    return pd.Series(iso.fit_predict(df[["DIAS"]]), index=df.index, name="ANOMALIA")


# ===============================
# PIPELINE COMPLETO
# ===============================

def procesar_base(nombre_base: str) -> pd.DataFrame:
    """Excel crudo -> DataFrame completamente enriquecido y tipado."""
    df = cargar_excel(nombre_base)
    df = enriquecer_nlp(df)

    # Inferencia del modelo (si los artefactos existen)
    if RUTA_MODELO.exists() and RUTA_VECTORIZER.exists() and RUTA_ENCODER.exists():
        modelo, vectorizer, encoder = cargar_modelo()
        df["PROB_RIESGO"] = predecir_probabilidades(df, modelo, vectorizer, encoder)
    else:
        df["PROB_RIESGO"] = np.nan

    df["ANOMALIA"] = detectar_anomalias(df)

    # Optimización de memoria: categóricas
    for col in COLUMNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
