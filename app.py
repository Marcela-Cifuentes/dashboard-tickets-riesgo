import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Sistema Inteligente de Tickets", layout="wide")

st.title("📊 Sistema Inteligente de Monitoreo de Service Desk")
st.caption("Analítica predictiva y monitoreo de riesgo operativo")

# refresco automático cada minuto
st_autorefresh(interval=60000, key="refresh")


# ===============================
# CARGA DATOS
# ===============================

@st.cache_data
def cargar_datos():

    df = pd.read_excel("TicketsHD.xlsx")

    df["CREACION"] = pd.to_datetime(df["CREACION"], errors="coerce")
    df["FECHA_RESPUESTA"] = pd.to_datetime(df["FECHA_RESPUESTA"], errors="coerce")

    df["TIEMPO_HORAS"] = (df["FECHA_RESPUESTA"] - df["CREACION"]).dt.total_seconds() / 3600

    df = df[df["TIEMPO_HORAS"] >= 0]

    df["DIAS"] = (df["TIEMPO_HORAS"] / 24).round(2)

    df["RIESGO_OPERATIVO"] = (df["DIAS"] > 5).astype(int)
    df["DEMORA_CRITICA"] = (df["DIAS"] > 7).astype(int)

    df["ESTADO_SLA"] = np.where(
        df["DIAS"] <= 3, "🟢 Dentro SLA",
        np.where(df["DIAS"] <= 5, "🟡 En riesgo", "🔴 Fuera SLA")
    )

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


# ===============================
# LIMPIEZA TEXTO
# ===============================

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


# ===============================
# CLASIFICADOR INCIDENTES
# ===============================

def clasificar_incidente(texto):

    texto = texto.lower()

    reglas = {
        "Acceso": ["login","acceso","contraseña","password"],
        "Correo": ["correo","email","outlook"],
        "Red": ["vpn","internet","red"],
        "Servidor": ["servidor","server","caido","down"],
        "Software": ["instalar","programa","aplicacion"]
    }

    for tipo, palabras in reglas.items():
        if any(p in texto for p in palabras):
            return tipo

    return "Otro"


# ===============================
# PREDICCION INDIVIDUAL
# ===============================

def predecir_riesgo(modelo, vectorizer, encoder, asunto, descripcion, prioridad, grupo, origen):

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

    if proba < 0.35:
        nivel = "Bajo"
    elif proba < 0.65:
        nivel = "Medio"
    else:
        nivel = "Alto"

    return proba, nivel


# ===============================
# PREDICCION MASIVA
# ===============================

def predecir_dataset(df, modelo, vectorizer, encoder):

    textos = df["TEXTO_COMPLETO"].fillna("").apply(limpiar_texto)

    X_text = vectorizer.transform(textos)

    X_cat = df[["PRIORIDAD","GRUPO","ORIGEN"]]

    X_cat_enc = encoder.transform(X_cat)

    X = hstack([X_text, X_cat_enc])

    probs = modelo.predict_proba(X)[:,1]

    df["PROB_RIESGO"] = probs

    return df


# ===============================
# DETECTOR ANOMALIAS
# ===============================

def detectar_anomalias(df):

    iso = IsolationForest(contamination=0.02, random_state=42)

    df["ANOMALIA"] = iso.fit_predict(df[["DIAS"]])

    return df


# ===============================
# CARGA
# ===============================

df = cargar_datos()

modelo, vectorizer, encoder = cargar_modelo()

df["TIPO_INCIDENTE"] = df["TEXTO_COMPLETO"].apply(clasificar_incidente)


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
# ALERTAS OPERATIVAS
# ===============================

st.sidebar.markdown("## 🚦 Estado Operacional")

tickets_criticos = df_filtrado[df_filtrado["DIAS"] > 7]
tickets_riesgo = df_filtrado[df_filtrado["DIAS"] > 5]

if len(tickets_criticos) > 0:
    st.sidebar.error(f"🔴 {len(tickets_criticos)} tickets fuera SLA")

elif len(tickets_riesgo) > 0:
    st.sidebar.warning(f"🟡 {len(tickets_riesgo)} tickets en riesgo")

else:
    st.sidebar.success("🟢 Operación estable")


# ===============================
# TABS
# ===============================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumen",
    "📈 Operación",
    "⚠️ Riesgo",
    "🔮 Predicción IA"
])


# ===============================
# TAB RESUMEN
# ===============================

with tab1:

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Tickets", len(df_filtrado))
    col2.metric("Promedio días", round(df_filtrado["DIAS"].mean(),2))
    col3.metric("% Riesgo >5 días", round(df_filtrado["RIESGO_OPERATIVO"].mean()*100,2))
    col4.metric("% Demora crítica", round(df_filtrado["DEMORA_CRITICA"].mean()*100,2))

    st.divider()

    fig = px.histogram(df_filtrado[df_filtrado["DIAS"]<=30], x="DIAS", nbins=30)

    st.plotly_chart(fig, use_container_width=True)

    fig_sla = px.pie(df_filtrado, names="ESTADO_SLA")

    st.plotly_chart(fig_sla, use_container_width=True)


# ===============================
# TAB OPERACION
# ===============================

with tab2:

    df_tiempo = df_filtrado.copy()

    df_tiempo["MES"] = df_tiempo["CREACION"].dt.to_period("M").astype(str)

    fig = px.line(
        df_tiempo.groupby("MES").size().reset_index(name="Tickets"),
        x="MES",
        y="Tickets",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.pie(df_filtrado, names="PRIORIDAD")

    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        df_filtrado.groupby("ORIGEN").size().reset_index(name="Tickets"),
        x="ORIGEN",
        y="Tickets"
    )

    st.plotly_chart(fig, use_container_width=True)

    tabla_heat = pd.crosstab(df_filtrado["GRUPO"], df_filtrado["PRIORIDAD"])

    fig_heat = px.imshow(tabla_heat, text_auto=True)

    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Tipos de incidentes detectados")

    fig_inc = px.bar(
        df_filtrado.groupby("TIPO_INCIDENTE").size().reset_index(name="Tickets"),
        x="TIPO_INCIDENTE",
        y="Tickets"
    )

    st.plotly_chart(fig_inc, use_container_width=True)


# ===============================
# TAB RIESGO
# ===============================

with tab3:

    fig = px.bar(
        df_filtrado.groupby("GRUPO")["RIESGO_OPERATIVO"].mean().reset_index(),
        x="GRUPO",
        y="RIESGO_OPERATIVO"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df_filtrado, x="GRUPO", y="DIAS")

    st.plotly_chart(fig, use_container_width=True)

    texto = " ".join(df_filtrado["TEXTO_COMPLETO"].astype(str))

    texto = limpiar_texto(texto)

    palabras = texto.split()

    conteo = Counter(palabras)

    top_palabras = pd.DataFrame(conteo.most_common(20), columns=["Palabra","Frecuencia"])

    fig_words = px.bar(top_palabras, x="Frecuencia", y="Palabra", orientation="h")

    st.plotly_chart(fig_words, use_container_width=True)

    df_pred = predecir_dataset(df_filtrado.copy(), modelo, vectorizer, encoder)

    fig_pred = px.histogram(df_pred, x="PROB_RIESGO", nbins=30)

    st.plotly_chart(fig_pred, use_container_width=True)

    st.subheader("🚨 Tickets críticos")

    criticos = df_filtrado[df_filtrado["DIAS"] > 7]

    st.dataframe(
        criticos[[
            "TICKET_ID",
            "TICKET_ASUNTO",
            "GRUPO",
            "PRIORIDAD",
            "DIAS"
        ]]
    )

    df_anom = detectar_anomalias(df_filtrado.copy())

    fig_anom = px.scatter(df_anom, x="DIAS", y="PRIORIDAD", color="ANOMALIA")

    st.plotly_chart(fig_anom, use_container_width=True)


# ===============================
# TAB PREDICCION
# ===============================

with tab4:

    st.subheader("Predicción de riesgo de nuevo ticket")

    asunto = st.text_input("Asunto")

    descripcion = st.text_area("Descripción")

    prioridad = st.selectbox("Prioridad", sorted(df["PRIORIDAD"].dropna().unique()))

    grupo = st.selectbox("Grupo", sorted(df["GRUPO"].dropna().unique()))

    origen = st.selectbox("Origen", sorted(df["ORIGEN"].dropna().unique()))

    if st.button("Predecir"):

        proba, nivel = predecir_riesgo(
            modelo,
            vectorizer,
            encoder,
            asunto,
            descripcion,
            prioridad,
            grupo,
            origen
        )

        st.success(f"Probabilidad riesgo: {round(proba,3)}")

        st.info(f"Nivel riesgo: {nivel}")
