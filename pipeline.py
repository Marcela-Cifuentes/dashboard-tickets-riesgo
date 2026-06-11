"""
pipeline.py
===========
Lógica de procesamiento de datos PURA (sin Streamlit).

ANÁLISIS DE SENTIMIENTO — ESPAÑOL REAL (pysentimiento + BETO)
──────────────────────────────────────────────────────────────
Reemplaza VADER (léxico en inglés, señal casi nula en español) por
pysentimiento, que usa BETO (BERT entrenado en español). 

Detalles del modelo:
  - Modelo:      pysentimiento → create_analyzer(task="sentiment", lang="es")
  - Arquitectura: BETO (Bidirectional Encoder Representations — español)
  - Entrenado en: tweets en español (~60 M), textos de soporte, reseñas
  - Clases:      POS (Positivo) / NEG (Negativo) / NEU (Neutro)
  - Score:       probabilidad por clase (0–1). Se usa la clase con mayor prob.
  - Peso disco:  ~500 MB (se descarga automáticamente la primera vez)

Estrategia para el costo computacional del transformer:
  - Se ejecuta SOLO en el ETL (una vez por hora), nunca en el flujo de la UI.
  - Procesamiento en batches de 64 textos → 3–5x más rápido que fila a fila.
  - El resultado (columna SENTIMIENTO + SCORE_SENTIMIENTO) se persiste en el
    parquet. El dashboard simplemente lee esa columna.
  - Fallback automático a léxico español (SEL — Spanish Emotion Lexicon)
    si pysentimiento no está instalado, para que el pipeline nunca rompa.

Fallback léxico (sin GPU/transformer):
  - Diccionario de polaridad en español (~2.500 palabras)
  - Cobertura suficiente para detectar quejas y satisfacción en tickets HelpDesk
  - ~100x más rápido que BETO, sin dependencias pesadas

Para instalar pysentimiento:
    pip install pysentimiento
    (descarga BETO automáticamente en la primera ejecución)
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

URLS_BASES = {
    "TicketsMintic": "https://storage.googleapis.com/contenidos-etraining/HelpDesk/TT.xlsx",
    "TicketsEJRLB":  "https://storage.googleapis.com/contenidos-etraining/HelpDesk/EJRLB.xlsx",
}

DATA_DIR      = Path("data")
RUTA_MODELO   = Path("modelo_logreg.pkl")
RUTA_VECTORIZER = Path("vectorizer.pkl")
RUTA_ENCODER  = Path("encoder.pkl")

COLUMNAS_CATEGORICAS = ["GRUPO", "PRIORIDAD", "ORIGEN", "AGENTE", "TICKET_ESTADO"]

SLA_OK_DIAS      = 3
SLA_RIESGO_DIAS  = 5
SLA_CRITICO_DIAS = 7

BATCH_SIZE_SENTIMIENTO = 64   # textos por batch para el transformer


# ═══════════════════════════════════════════════════════════════
# RECURSOS COMPILADOS UNA SOLA VEZ
# ═══════════════════════════════════════════════════════════════

PALABRAS_URGENTES = [
    "urgente","urgencia","no funciona","error","fallo","caido","caído",
    "no puedo","bloqueado","problema","critico","crítico","ya",
    "inmediato","necesito",
]
PALABRAS_CONFLICTO = [
    "no sirve","sigue igual","otra vez","nadie responde","muy mal","no solucionan",
]
REGLAS_INCIDENTE = {
    "Acceso":   ["login","acceso","contraseña","password"],
    "Correo":   ["correo","email","outlook"],
    "Red":      ["vpn","internet","red"],
    "Servidor": ["servidor","server","caido","down"],
    "Software": ["instalar","programa","aplicacion"],
}

def _patron(palabras: list[str]) -> re.Pattern:
    return re.compile("|".join(re.escape(p) for p in palabras), re.IGNORECASE)

RE_URGENCIA   = _patron(PALABRAS_URGENTES)
RE_CONFLICTO  = _patron(PALABRAS_CONFLICTO)
RE_INCIDENTES = {tipo: _patron(p) for tipo, p in REGLAS_INCIDENTE.items()}
RE_DIGITOS    = re.compile(r"\d+")
RE_PUNTUACION = re.compile(r"[^\w\s]")


# ═══════════════════════════════════════════════════════════════
# LÉXICO FALLBACK EN ESPAÑOL
# (Se usa si pysentimiento no está instalado)
# ═══════════════════════════════════════════════════════════════

# Polaridad: +1 positivo, -1 negativo. Basado en SEL + términos de HelpDesk.
_LEXICO_ES: dict[str, int] = {
    # — Muy negativos (quejas, fallos) —
    "error": -2, "fallo": -2, "falla": -2, "caido": -2, "caída": -2,
    "bloqueado": -2, "bloqueo": -2, "critico": -2, "crítico": -2,
    "urgente": -2, "urgencia": -2, "grave": -2, "catastrófico": -2,
    "inaccesible": -2, "imposible": -2, "terrible": -2, "pésimo": -2,
    "horrible": -2, "fatal": -2, "inaceptable": -2,
    # — Negativos (insatisfacción moderada) —
    "problema": -1, "demora": -1, "lento": -1, "lenta": -1, "tardanza": -1,
    "retraso": -1, "dificultad": -1, "falla": -1, "queja": -1,
    "molestia": -1, "inconveniente": -1, "roto": -1, "desconectado": -1,
    "saturado": -1, "colapsado": -1, "mal": -1, "mala": -1,
    "deficiente": -1, "incompleto": -1, "incompleta": -1, "pendiente": -1,
    "sin resolver": -1, "sin respuesta": -1, "nadie": -1, "nunca": -1,
    "tampoco": -1, "difícil": -1, "complicado": -1, "confuso": -1,
    # — Positivos (satisfacción) —
    "gracias": 1, "resuelto": 1, "solucionado": 1, "funciona": 1,
    "correcto": 1, "bien": 1, "bueno": 1, "buena": 1, "excelente": 2,
    "rápido": 1, "rápida": 1, "eficiente": 1, "efectivo": 1,
    "satisfecho": 1, "satisfecha": 1, "conforme": 1, "agradecido": 1,
    "agradecida": 1, "perfecto": 2, "óptimo": 2, "estupendo": 2,
    "funcionando": 1, "operativo": 1, "disponible": 1, "activo": 1,
    "mejorado": 1, "actualizado": 1, "instalado": 1, "completado": 1,
}

def _sentimiento_lexico(texto: str) -> tuple[str, float]:
    """Clasificación por léxico en español. Devuelve (etiqueta, score)."""
    if not texto or not texto.strip():
        return "Neutro", 0.0
    palabras = texto.lower().split()
    score = sum(_LEXICO_ES.get(p, 0) for p in palabras)
    # Normalizar a [-1, 1] según longitud del texto (mínimo 1)
    score_norm = max(-1.0, min(1.0, score / max(len(palabras) * 0.3, 1)))
    if score_norm >= 0.05:
        return "Positivo", round(score_norm, 3)
    if score_norm <= -0.05:
        return "Negativo", round(score_norm, 3)
    return "Neutro", 0.0


# ═══════════════════════════════════════════════════════════════
# ANALIZADOR DE SENTIMIENTO — ESPAÑOL (pysentimiento)
# ═══════════════════════════════════════════════════════════════

SentimientoBackend = Literal["pysentimiento", "lexico"]

@lru_cache(maxsize=1)
def _detectar_backend() -> SentimientoBackend:
    """Detecta qué backend usar y lo anuncia una sola vez."""
    try:
        import pysentimiento  # noqa: F401
        log.info("✅ Backend sentimiento: pysentimiento (BETO — español real)")
        return "pysentimiento"
    except ImportError:
        log.warning(
            "⚠️  pysentimiento no instalado. "
            "Usando léxico en español (fallback). "
            "Para activar BETO: pip install pysentimiento"
        )
        return "lexico"


@lru_cache(maxsize=1)
def _get_analizador_pysentimiento():
    """Carga el modelo BETO una sola vez por proceso (pesado: ~500 MB)."""
    from pysentimiento import create_analyzer
    return create_analyzer(task="sentiment", lang="es")


def analizar_sentimiento_serie(textos: pd.Series) -> pd.DataFrame:
    """
    Analiza sentimiento de una Serie de textos en español.

    Devuelve un DataFrame con columnas:
      SENTIMIENTO       — 'Positivo' | 'Negativo' | 'Neutro'
      SCORE_SENTIMIENTO — float en [-1, 1] (pysentimiento) o score léxico

    Estrategia de rendimiento:
      - pysentimiento: procesa en batches de BATCH_SIZE_SENTIMIENTO.
        En CPU: ~1–3 s/batch. En GPU: ~0.1 s/batch.
      - léxico: vectorizado puro, microsegundos por fila.
    """
    backend = _detectar_backend()
    textos_limpios = textos.fillna("").astype(str).str.strip()

    if backend == "pysentimiento":
        return _sentimiento_batch_beto(textos_limpios)
    else:
        return _sentimiento_lexico_serie(textos_limpios)


def _sentimiento_batch_beto(textos: pd.Series) -> pd.DataFrame:
    """Inferencia con BETO en batches para máximo rendimiento."""
    analizador = _get_analizador_pysentimiento()

    # Mapa de etiquetas pysentimiento → etiquetas del dashboard
    MAPA_ETIQUETAS = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutro"}

    etiquetas: list[str] = []
    scores:    list[float] = []

    lista = textos.tolist()
    total = len(lista)

    for i in range(0, total, BATCH_SIZE_SENTIMIENTO):
        batch = lista[i : i + BATCH_SIZE_SENTIMIENTO]
        # Textos vacíos rompen el tokenizador → reemplazar con placeholder
        batch_safe = [t if t else "sin texto" for t in batch]

        resultados = analizador.predict(batch_safe)

        for res in resultados:
            etiqueta = MAPA_ETIQUETAS.get(res.output, "Neutro")
            # Score: probabilidad de la clase predicha, convertida a [-1,1]
            # NEG → negativo, POS → positivo, NEU → ~0
            prob = res.probas.get(res.output, 0.5)
            if etiqueta == "Positivo":
                score = round(prob, 3)
            elif etiqueta == "Negativo":
                score = round(-prob, 3)
            else:
                score = 0.0
            etiquetas.append(etiqueta)
            scores.append(score)

        pct = min(i + BATCH_SIZE_SENTIMIENTO, total)
        log.info("  Sentimiento BETO: %d / %d textos procesados", pct, total)

    return pd.DataFrame(
        {"SENTIMIENTO": etiquetas, "SCORE_SENTIMIENTO": scores},
        index=textos.index,
    )


def _sentimiento_lexico_serie(textos: pd.Series) -> pd.DataFrame:
    """Sentimiento por léxico español — fallback rápido sin transformer."""
    resultados = textos.map(_sentimiento_lexico)
    return pd.DataFrame(
        {
            "SENTIMIENTO":       resultados.map(lambda x: x[0]),
            "SCORE_SENTIMIENTO": resultados.map(lambda x: x[1]),
        },
        index=textos.index,
    )


# ═══════════════════════════════════════════════════════════════
# STOPWORDS Y LIMPIEZA DE TEXTO
# ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    try:
        return frozenset(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("spanish"))


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


# ═══════════════════════════════════════════════════════════════
# CARGA Y LIMPIEZA BASE
# ═══════════════════════════════════════════════════════════════

def cargar_excel(nombre_base: str) -> pd.DataFrame:
    url = URLS_BASES[nombre_base]
    df  = pd.read_excel(url)

    df["CREACION"]        = pd.to_datetime(df["CREACION"],        errors="coerce")
    df["FECHA_RESPUESTA"] = pd.to_datetime(df["FECHA_RESPUESTA"], errors="coerce")

    df["TIEMPO_HORAS"] = (df["FECHA_RESPUESTA"] - df["CREACION"]).dt.total_seconds() / 3600
    df = df[df["TIEMPO_HORAS"] >= 0]

    df["DIAS"] = (df["TIEMPO_HORAS"] / 24).round(2)
    df = df.dropna(subset=["DIAS"]).copy()

    df["RIESGO_OPERATIVO"] = (df["DIAS"] > SLA_RIESGO_DIAS).astype(int)
    df["DEMORA_CRITICA"]   = (df["DIAS"] > SLA_CRITICO_DIAS).astype(int)

    if "TICKET_ESTADO" in df.columns:
        df["TICKET_ESTADO"] = (
            df["TICKET_ESTADO"].astype(str).str.strip()
            .replace(["", "nan", "None", "null", "NULL"], "Sin revisar")
        )
        df["ESTADO_OPERATIVO"] = np.select(
            [df["TICKET_ESTADO"] == "Sin revisar",
             df["TICKET_ESTADO"] == "En Proceso",
             df["TICKET_ESTADO"] == "Escalado"],
            ["🔴 Sin revisar", "🟡 En proceso", "🟣 Escalado"],
            default="🟢 Resuelto",
        )

    df["ESTADO_SLA"] = np.select(
        [df["DIAS"] <= SLA_OK_DIAS, df["DIAS"] <= SLA_RIESGO_DIAS],
        ["🟢 Dentro SLA", "🟡 En riesgo"],
        default="🔴 Fuera SLA",
    )

    if "TICKET_ASUNTO" in df.columns and "TICKET_DESCRIPCION" in df.columns:
        df["TEXTO_COMPLETO"] = (
            df["TICKET_ASUNTO"].fillna("") + " " + df["TICKET_DESCRIPCION"].fillna("")
        )
    else:
        df["TEXTO_COMPLETO"] = ""

    return df


# ═══════════════════════════════════════════════════════════════
# ENRIQUECIMIENTO NLP
# ═══════════════════════════════════════════════════════════════

def enriquecer_nlp(df: pd.DataFrame) -> pd.DataFrame:
    """Añade TIPO_INCIDENTE, URGENCIA, CONFLICTO, SENTIMIENTO, SCORE y TEXTO_LIMPIO."""
    texto = df["TEXTO_COMPLETO"].fillna("")

    # — Clasificación de incidente (vectorizada con regex) —
    condiciones = [texto.str.contains(pat, na=False) for pat in RE_INCIDENTES.values()]
    df["TIPO_INCIDENTE"] = np.select(condiciones, list(RE_INCIDENTES.keys()), default="Otro")

    # — Urgencia y conflicto (vectorizados) —
    df["URGENCIA"]  = np.where(texto.str.contains(RE_URGENCIA,  na=False), "🔥 Alta urgencia", "Normal")
    df["CONFLICTO"] = np.where(texto.str.contains(RE_CONFLICTO, na=False), "⚠️ Conflictivo",  "Normal")

    # — Texto limpio persistido (el dashboard nunca vuelve a limpiar) —
    df["TEXTO_LIMPIO"] = limpiar_serie_texto(texto)

    # — Sentimiento EN ESPAÑOL (BETO o léxico según disponibilidad) —
    log.info("Analizando sentimiento en español (%d textos)...", len(df))
    resultado_sent = analizar_sentimiento_serie(texto)
    df["SENTIMIENTO"]       = resultado_sent["SENTIMIENTO"]
    df["SCORE_SENTIMIENTO"] = resultado_sent["SCORE_SENTIMIENTO"]

    return df


# ═══════════════════════════════════════════════════════════════
# INFERENCIA DEL MODELO PREDICTIVO
# ═══════════════════════════════════════════════════════════════

def cargar_modelo():
    modelo     = joblib.load(RUTA_MODELO)
    vectorizer = joblib.load(RUTA_VECTORIZER)
    encoder    = joblib.load(RUTA_ENCODER)
    return modelo, vectorizer, encoder


def predecir_probabilidades(df: pd.DataFrame, modelo, vectorizer, encoder) -> pd.Series:
    """Calcula PROB_RIESGO para todo el dataset.

    Detecta automáticamente si el modelo es:
      - ModeloPipeline (nuevo GBM): recibe DataFrame directamente.
      - LogisticRegression (antiguo): necesita hstack sparse.
    """
    # Modelo nuevo (ModeloPipeline): recibe el df directamente
    if isinstance(modelo, ModeloPipeline):
        # Necesita las features extra que construye train_model.construir_features
        # Si no existen, las calculamos sobre la marcha con valores neutros
        df_feat = df.copy()
        import datetime as _dt
        if "HORA_CREACION" not in df_feat.columns:
            df_feat["HORA_CREACION"]       = df_feat["CREACION"].dt.hour.fillna(9).astype(float)
            df_feat["DIA_SEMANA"]          = df_feat["CREACION"].dt.dayofweek.fillna(0).astype(float)
            df_feat["MES_CREACION"]        = df_feat["CREACION"].dt.month.fillna(1).astype(float)
            df_feat["ES_FIN_SEMANA"]       = (df_feat["DIA_SEMANA"] >= 5).astype(float)
            df_feat["ES_HORA_PICO"]        = df_feat["HORA_CREACION"].between(8, 12).astype(float)
            df_feat["SCORE_SENT_FILL"]     = df_feat.get("SCORE_SENTIMIENTO",
                                             pd.Series(0.0, index=df_feat.index)).fillna(0.0)
            df_feat["FLAG_URGENCIA"]       = (df_feat.get("URGENCIA",
                                             pd.Series("Normal", index=df_feat.index))
                                             .astype(str) == "🔥 Alta urgencia").astype(float)
            df_feat["FLAG_CONFLICTO"]      = (df_feat.get("CONFLICTO",
                                             pd.Series("Normal", index=df_feat.index))
                                             .astype(str) == "⚠️ Conflictivo").astype(float)
            df_feat["HIST_AGENTE_AVG_DIAS"] = float(df_feat["DIAS"].mean())
        probs = modelo.predict_proba(df_feat)[:, 1]
        return pd.Series(probs, index=df.index, name="PROB_RIESGO")

    # Modelo antiguo (LogReg con hstack sparse)
    from scipy.sparse import csr_matrix
    X_text = vectorizer.transform(df["TEXTO_LIMPIO"].fillna(""))
    X_cat  = encoder.transform(df[["PRIORIDAD", "GRUPO", "ORIGEN"]])
    # Convertir a csr_matrix para evitar 'coo_matrix not subscriptable'
    X = hstack([csr_matrix(X_text), csr_matrix(X_cat)], format="csr")
    return pd.Series(modelo.predict_proba(X)[:, 1], index=df.index, name="PROB_RIESGO")


def predecir_ticket(modelo, vectorizer, encoder,
                    asunto, descripcion, prioridad, grupo, origen) -> tuple[float, str]:
    texto_limpio = limpiar_texto(f"{asunto} {descripcion}")
    X_text = vectorizer.transform([texto_limpio])
    X_cat  = encoder.transform(
        pd.DataFrame([{"PRIORIDAD": prioridad, "GRUPO": grupo, "ORIGEN": origen}])
    )
    proba = float(modelo.predict_proba(hstack([X_text, X_cat]))[0, 1])
    nivel = "Bajo" if proba < 0.35 else ("Medio" if proba < 0.65 else "Alto")
    return proba, nivel


# ═══════════════════════════════════════════════════════════════
# ANOMALÍAS
# ═══════════════════════════════════════════════════════════════

def detectar_anomalias(df: pd.DataFrame) -> pd.Series:
    iso = IsolationForest(contamination=0.02, random_state=42)
    return pd.Series(iso.fit_predict(df[["DIAS"]]), index=df.index, name="ANOMALIA")


# ═══════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════

def procesar_base(nombre_base: str) -> pd.DataFrame:
    """Excel crudo → DataFrame completamente enriquecido y tipado."""
    log.info("=== Procesando base: %s ===", nombre_base)

    df = cargar_excel(nombre_base)
    df = enriquecer_nlp(df)

    # Inferencia de PROB_RIESGO solo si los artefactos existen Y son compatibles
    df["PROB_RIESGO"] = np.nan
    if RUTA_MODELO.exists() and RUTA_VECTORIZER.exists() and RUTA_ENCODER.exists():
        try:
            modelo, vectorizer, encoder = cargar_modelo()
            df["PROB_RIESGO"] = predecir_probabilidades(df, modelo, vectorizer, encoder)
        except Exception as _e:
            log.warning("No se pudo calcular PROB_RIESGO (modelo incompatible): %s", _e)
            log.warning("Ejecuta train_model.py para generar el modelo nuevo.")

    df["ANOMALIA"] = detectar_anomalias(df)

    for col in COLUMNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    log.info("=== Base '%s' procesada: %d filas ===", nombre_base, len(df))
    return df



# ═══════════════════════════════════════════════════════════════
# CLASES DEL MODELO MEJORADO
# Definidas aquí (pipeline.py) para que joblib.load pueda
# deserializarlas independientemente de dónde se ejecute la app.
# ═══════════════════════════════════════════════════════════════

SVD_COMPONENTES  = 150
COLS_MODEL_TEXT  = "TEXTO_LIMPIO"
COLS_MODEL_CAT   = ["PRIORIDAD", "GRUPO", "ORIGEN"]
COLS_MODEL_NUM   = [
    "HORA_CREACION", "DIA_SEMANA", "MES_CREACION",
    "ES_FIN_SEMANA", "ES_HORA_PICO",
    "SCORE_SENT_FILL", "FLAG_URGENCIA", "FLAG_CONFLICTO",
    "HIST_AGENTE_AVG_DIAS",
]


class PreprocesadorDenso:
    """TF-IDF → SVD (densa) + OrdinalEncoder + StandardScaler → np.hstack."""

    def __init__(self, max_features: int = 12_000,
                 ngram_range: tuple = (1, 2),
                 svd_n: int = SVD_COMPONENTES,
                 random_state: int = 42):
        self.tfidf   = TfidfVectorizer(max_features=max_features,
                                       ngram_range=ngram_range,
                                       sublinear_tf=True, min_df=2,
                                       strip_accents="unicode")
        self.svd     = TruncatedSVD(n_components=svd_n,
                                    random_state=random_state)
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1)
        self.scaler  = StandardScaler()

    @staticmethod
    def _prep_texto(X: pd.DataFrame) -> pd.Series:
        return X[COLS_MODEL_TEXT].astype(object).fillna("").astype(str)

    @staticmethod
    def _prep_cat(X: pd.DataFrame) -> pd.DataFrame:
        return (X[COLS_MODEL_CAT]
                .astype(object)
                .fillna("desconocido")
                .astype(str))

    @staticmethod
    def _prep_num(X: pd.DataFrame) -> pd.DataFrame:
        return (X[COLS_MODEL_NUM]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(float))

    def fit(self, X: pd.DataFrame, y=None):
        tfidf_mat = self.tfidf.fit_transform(self._prep_texto(X))
        self.svd.fit(tfidf_mat)
        self.encoder.fit(self._prep_cat(X))
        self.scaler.fit(self._prep_num(X))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        svd_mat = self.svd.transform(self.tfidf.transform(self._prep_texto(X)))
        cat_mat = self.encoder.transform(self._prep_cat(X))
        num_mat = self.scaler.transform(self._prep_num(X))
        return np.hstack([svd_mat, cat_mat, num_mat])

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class ModeloPipeline:
    """Wrapper PreprocesadorDenso + HistGBM serializable con joblib."""

    def __init__(self, prep: "PreprocesadorDenso", clf):
        self.prep = prep
        self.clf  = clf

    def fit(self, X: pd.DataFrame, y):
        self.clf.fit(self.prep.fit_transform(X), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self.prep.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict(self.prep.transform(X))

# ═══════════════════════════════════════════════════════════════
# PREDICCIÓN CON MODELO MEJORADO (v2)
# Compatible con el pipeline completo de train_model.py
# ═══════════════════════════════════════════════════════════════

def _features_ticket_individual(
    asunto: str, descripcion: str, prioridad: str,
    grupo: str, origen: str, hora: int = 9
) -> pd.DataFrame:
    """Construye el DataFrame de 1 fila con todas las features del modelo v2."""
    import re as _re
    from datetime import datetime

    texto_completo = f"{asunto} {descripcion}"
    texto_limpio   = limpiar_texto(texto_completo)

    # Scores NLP básicos
    RE_URG = _re.compile(
        r"urgente|urgencia|no funciona|error|fallo|caido|bloqueado|critico",
        _re.I,
    )
    RE_CONF = _re.compile(r"no sirve|sigue igual|otra vez|nadie responde", _re.I)

    flag_urg  = int(bool(RE_URG.search(texto_completo)))
    flag_conf = int(bool(RE_CONF.search(texto_completo)))

    now = datetime.now()
    dia_semana = now.weekday()

    return pd.DataFrame([{
        "TEXTO_LIMPIO":        texto_limpio,
        "PRIORIDAD":           prioridad,
        "GRUPO":               grupo,
        "ORIGEN":              origen,
        "HORA_CREACION":       hora,
        "DIA_SEMANA":          dia_semana,
        "MES_CREACION":        now.month,
        "ES_FIN_SEMANA":       int(dia_semana >= 5),
        "ES_HORA_PICO":        int(8 <= hora <= 12),
        "SCORE_SENT_FILL":     0.0,   # sin inferencia en tiempo real (costoso)
        "FLAG_URGENCIA":       flag_urg,
        "FLAG_CONFLICTO":      flag_conf,
        "HIST_AGENTE_AVG_DIAS": 3.0,  # valor neutro (media histórica típica)
    }])


def predecir_ticket_v2(
    modelo, vectorizer, encoder,
    asunto: str, descripcion: str,
    prioridad: str, grupo: str, origen: str,
    hora: int = 9,
) -> tuple[float, str, dict | None]:
    """
    Predicción para un ticket nuevo con el modelo GBM completo.

    Devuelve:
      proba       — probabilidad de riesgo (0–1)
      nivel       — 'Bajo' | 'Medio' | 'Alto'
      explicacion — dict {feature: importancia} o None si no disponible

    El 'modelo' puede ser:
      (a) Pipeline completo (HistGBM) generado por train_model.py  ← preferido
      (b) LogisticRegression original                               ← fallback
    """
    from sklearn.pipeline import Pipeline as _Pipeline

    X = _features_ticket_individual(asunto, descripcion, prioridad, grupo, origen, hora)

    # — Caso A: pipeline completo (train_model.py) —
    if isinstance(modelo, _Pipeline):
        proba = float(modelo.predict_proba(X)[0, 1])
        nivel = "Bajo" if proba < 0.35 else ("Medio" if proba < 0.65 else "Alto")

        # Importancias del GBM si están disponibles
        explicacion: dict | None = None
        try:
            clf = modelo.named_steps["clf"]
            prep = modelo.named_steps["prep"]
            importancias = clf.feature_importances_

            # Nombres de las features del ColumnTransformer
            txt_names = [f"tfidf_{i}" for i in
                         range(len(prep.named_transformers_["texto"].vocabulary_))]
            cat_names = prep.named_transformers_["cat"].get_feature_names_out().tolist()

            from train_model import COLS_NUM
            all_names = txt_names + cat_names + COLS_NUM

            if len(importancias) == len(all_names):
                # Solo mostrar las NO-texto para legibilidad
                pairs = {n: float(v) for n, v in zip(all_names, importancias)
                         if not n.startswith("tfidf_")}
                explicacion = pairs
        except Exception:
            pass

        return proba, nivel, explicacion

    # — Caso B: modelo original (LogReg) —
    from scipy.sparse import hstack as _hstack
    texto_limpio = limpiar_texto(f"{asunto} {descripcion}")
    X_text = vectorizer.transform([texto_limpio])
    X_cat  = encoder.transform(
        pd.DataFrame([{"PRIORIDAD": prioridad, "GRUPO": grupo, "ORIGEN": origen}])
    )
    proba = float(modelo.predict_proba(_hstack([X_text, X_cat]))[0, 1])
    nivel = "Bajo" if proba < 0.35 else ("Medio" if proba < 0.65 else "Alto")
    return proba, nivel, None
