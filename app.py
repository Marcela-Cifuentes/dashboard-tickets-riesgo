import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="Dashboard Tickets - Riesgo Operativo", layout="wide")

st.markdown("## Sistema de alerta temprana de riesgo operativo")
st.caption("Modelo predictivo basado en análisis histórico de tickets")

# ===============================
# CARGA DE DATOS
# ===============================

@st.cache_data
def cargar_datos():
    df = pd.read_excel("TicketsHD.xlsx")

    df["CREACION"] = pd.to_datetime(df["CREACION"], errors="coerce")
    df["FECHA_RESPUESTA"] = pd.to_datetime(df["FECHA_RESPUESTA"], errors="coerce")

    df["TIEMPO_HORAS"] = (df["FECHA_RESPUESTA"] - df["CREACION"]).dt.total_seconds() / 3600
    df = df[df["TIEMPO_HORAS"] >= 0].dropna(subset=["TIEMPO_HORAS"])

    df["DIAS"] = (df["TIEMPO_HORAS"] / 24).round(2)

    df["RIESGO_OPERATIVO"] = (df["DIAS"] > 5).astype(int)
    df["DEMORA_CRITICA"] = (df["DIAS"] > 7).astype(int)

    df["TEXTO_COMPLETO"] = (
        df["TICKET_ASUNTO"].fillna("") + " " +
        df["TICKET_DESCRIPCION"].fillna("")
    )

    return df


# ===============================
# CARGA MODELO
# ===============================

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_logreg.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    encoder = joblib.load("encoder.pkl")
    return modelo, vectorizer, encoder


def limpiar_texto(texto):
    try:
        _ = stopwords.words("spanish")
    except:
        nltk.download("stopwords")

    stop_words = set(stopwords.words("spanish"))

    texto = texto.lower()
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words and len(p) > 2]

    return " ".join(palabras)


def predecir_riesgo(modelo, vectorizer, encoder, asunto, descripcion, prioridad, grupo, origen, threshold=0.35):
    texto = f"{asunto} {descripcion}"
    texto_limpio = limpiar_texto(texto)

    X_text = vectorizer.transform([texto_limpio])

    X_cat = pd.DataFrame([{
        "PRIORIDAD": prioridad,
        "GRUPO": grupo,
        "ORIGEN": origen
    }])

    X_cat_enc = encoder.transform(X_cat)

    X = hstack([X_text, X_cat_enc])

    proba = float(modelo.predict_proba(X)[0, 1])
    pred = int(proba >= threshold)

    if proba < 0.35:
        nivel = "Bajo"
    elif proba < 0.65:
        nivel = "Medio"
    else:
        nivel = "Alto"

    return proba, pred, nivel


# ===============================
# INTERFAZ
# ===============================

st.title("📊 Dashboard Tickets - Riesgo Operativo")

df = cargar_datos()
modelo, vectorizer, encoder = cargar_modelo()

# ===============================
# FILTROS
# ===============================

st.sidebar.header("Filtros")

grupo_sel = st.sidebar.selectbox("Grupo", ["Todos"] + sorted(df["GRUPO"].dropna().unique()))
prioridad_sel = st.sidebar.selectbox("Prioridad", ["Todos"] + sorted(df["PRIORIDAD"].dropna().unique()))
origen_sel = st.sidebar.selectbox("Origen", ["Todos"] + sorted(df["ORIGEN"].dropna().unique()))

df_filtrado = df.copy()

if grupo_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["GRUPO"] == grupo_sel]

if prioridad_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["PRIORIDAD"] == prioridad_sel]

if origen_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["ORIGEN"] == origen_sel]

# ===============================
# MÉTRICAS
# ===============================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Tickets", len(df_filtrado))
col2.metric("Promedio días", round(df_filtrado["DIAS"].mean(), 2))
col3.metric("% Riesgo >5 días", round(df_filtrado["RIESGO_OPERATIVO"].mean() * 100, 2))
col4.metric("% Demora >7 días", round(df_filtrado["DEMORA_CRITICA"].mean() * 100, 2))

st.divider()

# ===============================
# GRÁFICOS
# ===============================

fig1 = px.histogram(df_filtrado[df_filtrado["DIAS"] <= 30], x="DIAS",
                    nbins=30, title="Distribución de días (≤30)")
fig1.add_vline(x=5, line_dash="dash", line_color="red")
fig1.add_vline(x=7, line_dash="dash", line_color="orange")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(
    df_filtrado.groupby("GRUPO")["RIESGO_OPERATIVO"].mean().reset_index(),
    x="GRUPO",
    y="RIESGO_OPERATIVO",
    title="% Riesgo por Grupo"
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ===============================
# PREDICCIÓN
# ===============================

st.subheader("🔮 Predicción nuevo ticket")

with st.form("form_pred"):
    asunto = st.text_input("Asunto")
    descripcion = st.text_area("Descripción")
    prioridad = st.selectbox("Prioridad", sorted(df["PRIORIDAD"].dropna().unique()))
    grupo = st.selectbox("Grupo", sorted(df["GRUPO"].dropna().unique()))
    origen = st.selectbox("Origen", sorted(df["ORIGEN"].dropna().unique()))
    threshold = st.slider("Umbral", 0.1, 0.9, 0.35)
    submit = st.form_submit_button("Predecir")

if submit:
    proba, pred, nivel = predecir_riesgo(
        modelo, vectorizer, encoder,
        asunto, descripcion, prioridad, grupo, origen, threshold
    )

    st.success(f"Probabilidad riesgo: {round(proba,3)}")
    st.info(f"Nivel: {nivel}")
    
# ===============================
# Evolución de tickets en el tiempo
# ===============================
st.subheader("Evolución de tickets en el tiempo")

df_tiempo = df_filtrado.copy()
df_tiempo["MES"] = df_tiempo["CREACION"].dt.to_period("M").astype(str)

fig3 = px.line(
    df_tiempo.groupby("MES").size().reset_index(name="Tickets"),
    x="MES",
    y="Tickets",
    markers=True,
    title="Tickets por mes"
)

st.plotly_chart(fig3, use_container_width=True)


# ===============================
# Distribución de tickets por prioridad
# ===============================

st.subheader("Distribución por prioridad")

fig4 = px.pie(
    df_filtrado,
    names="PRIORIDAD",
    title="Proporción de tickets por prioridad"
)

st.plotly_chart(fig4, use_container_width=True)




# ===============================
# ETiempo de respuesta por grupo (boxplot)o
# # ===============================

st.subheader("Tiempo de resolución por grupo")

fig5 = px.box(
    df_filtrado,
    x="GRUPO",
    y="DIAS",
    title="Distribución de días de resolución por grupo"
)

st.plotly_chart(fig5, use_container_width=True)




# ===============================
# Riesgo operativo por prioridad
# ===============================

st.subheader("Riesgo operativo por prioridad")

fig6 = px.bar(
    df_filtrado.groupby("PRIORIDAD")["RIESGO_OPERATIVO"]
    .mean()
    .reset_index(),
    x="PRIORIDAD",
    y="RIESGO_OPERATIVO",
    title="Probabilidad de riesgo por prioridad"
)

st.plotly_chart(fig6, use_container_width=True)





# ===============================
# Tickets por origen
# ===============================

st.subheader("Origen de los tickets")

fig7 = px.bar(
    df_filtrado.groupby("ORIGEN").size().reset_index(name="Tickets"),
    x="ORIGEN",
    y="Tickets",
    title="Tickets por origen"
)

st.plotly_chart(fig7, use_container_width=True)



# ===============================
# Top 10 asuntos más frecuentes
# ===============================
st.subheader("Top 10 asuntos más frecuentes")

top_asuntos = (
    df_filtrado["TICKET_ASUNTO"]
    .value_counts()
    .head(10)
    .reset_index()
)

top_asuntos.columns = ["Asunto", "Cantidad"]

fig8 = px.bar(
    top_asuntos,
    x="Cantidad",
    y="Asunto",
    orientation="h",
    title="Top 10 asuntos"
)

st.plotly_chart(fig8, use_container_width=True)
    st.info(f"Nivel: {nivel}")

