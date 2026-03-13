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
from sklearn.metrics import roc_curve, auc, confusion_matrix
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Sistema Inteligente de Tickets", layout="wide")

st.title("Sistema Inteligente de Monitoreo HelpDesk")
st.caption("Analítica predictiva y monitoreo de riesgo operativo")

#st_autorefresh(interval=6000000, key="refresh")

SLA_COLORS = {
    "🟢 Dentro SLA": "#2ecc71",
    "🟡 En riesgo": "#f1c40f",
    "🔴 Fuera SLA": "#e74c3c"
}

# ===============================
# SELECCIÓN BASE DATOS
# ===============================
if st.sidebar.button("🔄 Actualizar datos"):
    st.cache_data.clear()
    st.experimental_rerun()
URLS_BASES = {
    "TicketsMintic": "https://storage.googleapis.com/contenidos-etraining/HelpDesk/TT.xlsx",
    "TicketsEJRLB": "https://storage.googleapis.com/contenidos-etraining/HelpDesk/EJRLB.xlsx"
}

st.sidebar.header("Fuente de datos")

base_datos = st.sidebar.selectbox(
    "Seleccionar base",
    list(URLS_BASES.keys())
)

st.sidebar.header("Comparación de bases")

base1 = st.sidebar.selectbox(
    "Base 1",
    list(URLS_BASES.keys())
)

base2 = st.sidebar.selectbox(
    "Base 2",
    list(URLS_BASES.keys()),
    index=1
)


# ===============================
# CARGA DATOS
# ===============================
@st.cache_data(ttl=7200)
def cargar_datos(nombre_base):

    url = URLS_BASES[nombre_base]

    df = pd.read_excel(url)

    df["CREACION"] = pd.to_datetime(df["CREACION"], errors="coerce")
    df["FECHA_RESPUESTA"] = pd.to_datetime(df["FECHA_RESPUESTA"], errors="coerce")
    

    df["TIEMPO_HORAS"] = (
        df["FECHA_RESPUESTA"] - df["CREACION"]
    ).dt.total_seconds() / 3600

    df = df[df["TIEMPO_HORAS"] >= 0]

    df["DIAS"] = (df["TIEMPO_HORAS"] / 24).round(2)

    df = df.dropna(subset=["DIAS"])

    df["RIESGO_OPERATIVO"] = (df["DIAS"] > 5).astype(int)
    df["DEMORA_CRITICA"] = (df["DIAS"] > 7).astype(int)

    # ===============================
    # LIMPIEZA ESTADO TICKET
    # ===============================
    
    if "TICKET_ESTADO" in df.columns:
    
        df["TICKET_ESTADO"] = df["TICKET_ESTADO"].astype(str).str.strip()
    
        df["TICKET_ESTADO"] = df["TICKET_ESTADO"].replace(
            ["", "nan", "None"],
            "Sin revisar"
        )
    
        df["ESTADO_OPERATIVO"] = np.where(
            df["TICKET_ESTADO"] == "Sin revisar",
            "🔴 Sin revisar",
            np.where(
                df["TICKET_ESTADO"] == "En Proceso",
                "🟡 En proceso",
                np.where(
                    df["TICKET_ESTADO"] == "Escalado",
                    "🟣 Escalado",
                    "🟢 Resuelto"
                )
            )
        )

    df["ESTADO_SLA"] = np.where(
        df["DIAS"] <= 3,
        "🟢 Dentro SLA",
        np.where(df["DIAS"] <= 5, "🟡 En riesgo", "🔴 Fuera SLA")
    )

    if "TICKET_ASUNTO" in df.columns and "TICKET_DESCRIPCION" in df.columns:
        df["TEXTO_COMPLETO"] = (
            df["TICKET_ASUNTO"].fillna("") + " " +
            df["TICKET_DESCRIPCION"].fillna("")
        )
    else:
        df["TEXTO_COMPLETO"] = ""

    return df

# ===============================
# MODELO
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
# PREDICCION
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
# ANOMALIAS
# ===============================

def detectar_anomalias(df):

    # eliminar valores nulos
    df_clean = df.dropna(subset=["DIAS"]).copy()

    iso = IsolationForest(
        contamination=0.02,
        random_state=42
    )

    df_clean["ANOMALIA"] = iso.fit_predict(df_clean[["DIAS"]])

    return df_clean

# ===============================
# FILTRADO CACHEADO
# ===============================

@st.cache_data
def filtrar_df(df, grupo, prioridad, origen):

    df_filtrado = df.copy()

    if grupo != "Todos":
        df_filtrado = df_filtrado[df_filtrado["GRUPO"] == grupo]

    if prioridad != "Todos":
        df_filtrado = df_filtrado[df_filtrado["PRIORIDAD"] == prioridad]

    if origen != "Todos":
        df_filtrado = df_filtrado[df_filtrado["ORIGEN"] == origen]

    return df_filtrado
# ===============================
# CARGA
# ===============================

df = cargar_datos(base_datos)

modelo, vectorizer, encoder = cargar_modelo()

df["TIPO_INCIDENTE"] = df["TEXTO_COMPLETO"].apply(clasificar_incidente)

# ===============================
# FILTROS
# ===============================

st.sidebar.header("Filtros")

grupo_sel = st.sidebar.selectbox("Grupo", ["Todos"] + sorted(df["GRUPO"].dropna().unique()))
prioridad_sel = st.sidebar.selectbox("Prioridad", ["Todos"] + sorted(df["PRIORIDAD"].dropna().unique()))
origen_sel = st.sidebar.selectbox("Origen", ["Todos"] + sorted(df["ORIGEN"].dropna().unique()))

df_filtrado = filtrar_df(
    df,
    grupo_sel,
    prioridad_sel,
    origen_sel
)

# ===============================
# ALERTAS
# ===============================

st.sidebar.markdown("##  Estado Operacional")

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resumen",
    "Operación",
    "Riesgo",
    "Modelo",
    "Comparación",
    "Agentes"
    
])

# ===============================
# TAB RESUMEN
# ===============================

with tab1:

    # ===============================
    # KPIs PRINCIPALES
    # ===============================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tickets", len(df_filtrado))
    col2.metric("Promedio días resolución", round(df_filtrado["DIAS"].mean(),2))
    col3.metric("% Riesgo >5 días", round(df_filtrado["RIESGO_OPERATIVO"].mean()*100,2))
    col4.metric("% Demora crítica", round(df_filtrado["DEMORA_CRITICA"].mean()*100,2))

    st.divider()

    # ===============================
    # KPIs OPERATIVOS
    # ===============================

    hoy = pd.Timestamp.today().normalize()

    tickets_hoy = df_filtrado[
        df_filtrado["CREACION"].dt.normalize() == hoy
    ]

    backlog = df_filtrado[
        df_filtrado["TICKET_ESTADO"].isin(["Sin revisar","En Proceso","Escalado"])
    ]

    resueltos = df_filtrado[
        df_filtrado["TICKET_ESTADO"] == "Resuelto"
    ]

    mediana_resolucion = df_filtrado["DIAS"].median()

    ticket_mas_antiguo = backlog["DIAS"].max() if len(backlog) > 0 else 0

    tasa_resolucion = (
        (len(resueltos) / len(df_filtrado)) * 100
        if len(df_filtrado) > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Tickets creados hoy", len(tickets_hoy))
    col2.metric("Backlog actual", len(backlog))
    col3.metric("Mediana resolución (días)", round(mediana_resolucion,2))
    col4.metric("Ticket abierto más antiguo", round(ticket_mas_antiguo,2))

    st.metric("Tasa de resolución", f"{round(tasa_resolucion,2)}%")

    # ===============================
    # SEMÁFORO OPERATIVO
    # ===============================

    if len(backlog) > 200:
        st.error("🔴 Operación saturada")
    elif len(backlog) > 100:
        st.warning("🟡 Operación en riesgo")
    else:
        st.success("🟢 Operación estable")

    st.divider()

    # ===============================
    # GRÁFICOS PRINCIPALES
    # ===============================

    colA, colB = st.columns(2)

    # HISTOGRAMA RESOLUCIÓN
    with colA:

        st.subheader("Distribución de días de resolución")

        fig_hist = px.histogram(
            df_filtrado[df_filtrado["DIAS"] <= 30],
            x="DIAS",
            nbins=30,
            title="Distribución de tiempo de resolución"
        )

        fig_hist.add_vline(x=3, line_dash="dash", line_color="green")
        fig_hist.add_vline(x=5, line_dash="dash", line_color="orange")
        fig_hist.add_vline(x=7, line_dash="dash", line_color="red")

        st.plotly_chart(fig_hist, use_container_width=True)

    # SLA DISTRIBUCIÓN
    with colB:

        st.subheader("Estado SLA")

        fig_sla = px.pie(
            df_filtrado,
            names="ESTADO_SLA",
            color="ESTADO_SLA",
            color_discrete_map=SLA_COLORS,
            title="Distribución SLA"
        )

        st.plotly_chart(fig_sla, use_container_width=True)

    st.divider()

    # ===============================
    # DISTRIBUCIÓN POR GRUPO
    # ===============================

    st.subheader("Distribución de tickets por grupo")

    tickets_grupo = (
        df_filtrado.groupby("GRUPO")
        .size()
        .reset_index(name="Tickets")
        .sort_values("Tickets", ascending=False)
    )

    fig_grupo = px.bar(
        tickets_grupo,
        x="GRUPO",
        y="Tickets",
        title="Volumen de tickets por grupo"
    )

    st.plotly_chart(fig_grupo, use_container_width=True)

    # ===============================
    # TENDENCIA SEMANAL
    # ===============================

    st.subheader("Tendencia semanal de tickets")

    df_tmp = df_filtrado.copy()

    df_tmp["SEMANA"] = df_tmp["CREACION"].dt.to_period("W").astype(str)

    tickets_semana = (
        df_tmp.groupby("SEMANA")
        .size()
        .reset_index(name="Tickets")
    )

    fig_semana = px.line(
        tickets_semana,
        x="SEMANA",
        y="Tickets",
        markers=True,
        title="Evolución semanal de tickets"
    )

    st.plotly_chart(fig_semana, use_container_width=True)

# ===============================
# TAB OPERACIÓN
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

    st.subheader(" Tickets críticos")

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
# TAB MODELO
# ===============================

with tab4:

    st.divider()

    st.subheader(" Predicción de riesgo de nuevo ticket")
    
    asunto = st.text_input("Asunto del ticket")
    
    descripcion = st.text_area("Descripción")
    
    prioridad = st.selectbox(
        "Prioridad",
        sorted(df["PRIORIDAD"].dropna().unique())
    )
    
    grupo = st.selectbox(
        "Grupo",
        sorted(df["GRUPO"].dropna().unique())
    )
    
    origen = st.selectbox(
        "Origen",
        sorted(df["ORIGEN"].dropna().unique())
    )

    if st.button("Predecir riesgo"):

        try:

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
    
            st.success(f"Probabilidad de riesgo: {round(proba,3)}")
            st.info(f"Nivel de riesgo: {nivel}")
    
        except Exception as e:
    
            st.error(f"Error en la predicción: {e}")

    df_pred = predecir_dataset(df_filtrado.copy(), modelo, vectorizer, encoder)

    y_true = df_pred["RIESGO_OPERATIVO"]
    y_score = df_pred["PROB_RIESGO"]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig_roc = px.line(x=fpr, y=tpr, title=f"Curva ROC (AUC={roc_auc:.2f})")

    st.plotly_chart(fig_roc, use_container_width=True)

    y_pred = (df_pred["PROB_RIESGO"] > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    fig_cm = px.imshow(cm, text_auto=True)

    st.plotly_chart(fig_cm, use_container_width=True)
# ===============================
# TAB COMPARACION 
# ===============================

with tab5:

    st.subheader("Comparación entre bases de tickets")

    if base1 == base2:
        st.warning("Selecciona dos bases diferentes para comparar")
        st.stop()
    else:

        # cargar ambas bases
        df1 = cargar_datos(base1)
        df2 = cargar_datos(base2)

        comparacion = pd.DataFrame({

            "Base": [base1, base2],

            "Total Tickets": [
                len(df1),
                len(df2)
            ],

            "Promedio días": [
                round(df1["DIAS"].mean(),2),
                round(df2["DIAS"].mean(),2)
            ],

            "% Riesgo": [
                round(df1["RIESGO_OPERATIVO"].mean()*100,2),
                round(df2["RIESGO_OPERATIVO"].mean()*100,2)
            ],

            "% Demora crítica": [
                round(df1["DEMORA_CRITICA"].mean()*100,2),
                round(df2["DEMORA_CRITICA"].mean()*100,2)
            ]

        })

        st.dataframe(comparacion)

        fig_comp = px.bar(
            comparacion,
            x="Base",
            y=["% Riesgo", "% Demora crítica"],
            barmode="group",
            title="Comparación de riesgo operativo"
        )

        df_comp = pd.concat([
            df1.assign(Base=base1),
            df2.assign(Base=base2)
        ])
        
        fig_sla_comp = px.bar(
            df_comp.groupby(["Base","ESTADO_SLA"]).size().reset_index(name="Tickets"),
            x="Base",
            y="Tickets",
            color="ESTADO_SLA",
            barmode="stack",
            color_discrete_map=SLA_COLORS,
            title="Comparación de SLA entre bases"
        )
        
        st.plotly_chart(fig_sla_comp, use_container_width=True)



# ===============================
# TAB GESTIÓN AGENTES
# ===============================

with tab6:

    st.header("Gestión operativa de agentes")

    if "AGENTE" not in df.columns:
        st.warning("La base no contiene columna AGENTE")
        st.stop()

    df_ag = df.copy()

    # NORMALIZAR ESTADO DEL TICKET
    df_ag["TICKET_ESTADO"] = (
        df_ag["TICKET_ESTADO"]
        .replace([None, np.nan], "")
        .astype(str)
        .str.strip()
    )
    
    df_ag["TICKET_ESTADO"] = df_ag["TICKET_ESTADO"].replace(
        ["", "nan", "None", "null", "NULL"],
        "Sin revisar"
    )

    df_ag["MES"] = pd.to_datetime(df_ag["CREACION"], errors="coerce").dt.to_period("M").astype(str)

    # ===============================
    # FILTROS
    # ===============================

    col1, col2, col3 = st.columns(3)

    with col1:
        mes_sel = st.selectbox(
            "Mes",
            ["Todos"] + sorted(df_ag["MES"].dropna().unique()),
            key="mes_agentes"
        )

    with col2:
        agente_sel = st.selectbox(
            "Agente",
            ["Todos"] + sorted(df_ag["AGENTE"].dropna().unique()),
            key="agente_agentes"
        )

    with col3:
        grupo_sel = st.selectbox(
            "Grupo",
            ["Todos"] + sorted(df_ag["GRUPO"].dropna().unique()),
            key="grupo_agentes"
        )

    if mes_sel != "Todos":
        df_ag = df_ag[df_ag["MES"] == mes_sel]

    if agente_sel != "Todos":
        df_ag = df_ag[df_ag["AGENTE"] == agente_sel]

    if grupo_sel != "Todos":
        df_ag = df_ag[df_ag["GRUPO"] == grupo_sel]

    st.divider()

    # ===============================
    # CARGA DE TRABAJO
    # ===============================

    st.subheader("Carga de trabajo por agente")

    carga = (
        df_ag.groupby("AGENTE")
        .size()
        .reset_index(name="Tickets")
        .sort_values("Tickets", ascending=False)
    )

    fig = px.bar(carga, x="AGENTE", y="Tickets", title="Tickets por agente")

    st.plotly_chart(fig, use_container_width=True, key="carga_agentes")

    # ===============================
    # SLA POR AGENTE
    # ===============================

    st.subheader("Cumplimiento SLA por agente")

    sla = (
        df_ag.groupby("AGENTE")["DIAS"]
        .apply(lambda x: (x<=5).mean()*100)
        .reset_index(name="SLA_%")
    )

    fig = px.bar(sla, x="AGENTE", y="SLA_%")

    st.plotly_chart(fig, use_container_width=True, key="sla_agentes")
    # ===============================
    # RANKING
    # ===============================

    st.subheader("Ranking de desempeño")

    ranking = df_ag.groupby("AGENTE").agg(
        Tickets=("TICKET_ID","count"),
        Promedio_dias=("DIAS","mean"),
        SLA=("DIAS", lambda x: (x<=5).mean()*100)
    ).reset_index()

    ranking = ranking.sort_values("SLA", ascending=False)

    st.dataframe(ranking)

    # ===============================
    # PRODUCTIVIDAD MENSUAL
    # ===============================

    st.subheader("Productividad mensual")

    prod = (
        df_ag.groupby(["MES","AGENTE"])
        .size()
        .reset_index(name="Tickets")
    )

    fig = px.line(prod, x="MES", y="Tickets", color="AGENTE", markers=True)

    st.plotly_chart(fig, use_container_width=True, key="productividad_agentes")

    # ===============================
    # AGENTES SATURADOS
    # ===============================

    st.subheader("Detección de agentes saturados")

    limite = carga["Tickets"].mean()*1.5

    carga["Estado"] = np.where(
        carga["Tickets"]>limite,
        "Sobrecarga",
        "Normal"
    )

    fig = px.bar(carga, x="AGENTE", y="Tickets", color="Estado")

    st.plotly_chart(fig, use_container_width=True, key="saturacion_agentes")

    # ===============================
    # AGENTE → GRUPO
    # ===============================

    st.subheader("Relación agente y grupo")

    agente_grupo = (
        df_ag[["AGENTE","GRUPO"]]
        .drop_duplicates()
        .sort_values(["GRUPO","AGENTE"])
    )

    st.dataframe(agente_grupo)

    # ===============================
    # RANKING AGENTE VS GRUPO
    # ===============================

    st.subheader("Ranking por agente y grupo")

    ranking_gr = (
        df_ag.groupby(["GRUPO","AGENTE"])
        .agg(
            Tickets=("TICKET_ID","count"),
            Promedio_dias=("DIAS","mean")
        )
        .reset_index()
    )

    st.dataframe(ranking_gr)

    # ===============================
    # INCUMPLIMIENTO SLA POR GRUPO
    # ===============================

    st.subheader("Incumplimiento SLA por grupo")

    sla_grupo = (
        df_ag.groupby("GRUPO")["DIAS"]
        .apply(lambda x: (x>5).mean()*100)
        .reset_index(name="Incumplimiento_%")
    )

    fig = px.bar(sla_grupo, x="GRUPO", y="Incumplimiento_%")

    st.plotly_chart(fig, use_container_width=True, key="sla_grupo")

    # ===============================
    # TICKETS ABIERTOS
    # ===============================
    
    st.subheader("Tickets no resueltos")

    df_ag["TICKET_ESTADO"] = (
        df_ag["TICKET_ESTADO"]
        .replace([None, np.nan], "")
        .astype(str)
        .str.strip()
    )
    
    df_ag["TICKET_ESTADO"] = df_ag["TICKET_ESTADO"].replace(
        ["", "nan", "None", "null", "NULL"],
        "Sin revisar"
    )
    st.write("Estados reales en dataset:")
    st.write(df_ag["TICKET_ESTADO"].value_counts(dropna=False))
  
    
    # ===============================
    # CLASIFICACIÓN OPERATIVA
    # ===============================
    
    # crear dataset de tickets abiertos
    abiertos = df_ag[
        df_ag["TICKET_ESTADO"].isin(["Sin revisar", "En Proceso", "Escalado"])
    ].copy()
    
    # clasificar estado operativo
    abiertos["ESTADO_OPERATIVO"] = np.where(
        abiertos["TICKET_ESTADO"] == "Sin revisar",
        "🔴 Sin revisar",
        np.where(
            abiertos["TICKET_ESTADO"] == "En Proceso",
            "🟠 En proceso",
            "🟡 Escalado"
        )
    )
    
    if len(abiertos) == 0:
    
        st.success("No hay tickets pendientes")
    
    else:
    
        # ===============================
        # TABLA POR AGENTE
        # ===============================
    
        tabla = (
            abiertos.groupby(["AGENTE", "GRUPO"])
            .size()
            .reset_index(name="Tickets abiertos")
        )
    
        st.dataframe(tabla)
    
        fig_abiertos = px.bar(
            tabla,
            x="AGENTE",
            y="Tickets abiertos",
            color="GRUPO",
            title="Tickets abiertos por agente"
        )
    
        st.plotly_chart(fig_abiertos, use_container_width=True, key="tickets_abiertos")
    
        # ===============================
        # ESTADO OPERATIVO
        # ===============================
        st.write("Estados en el dataset filtrado:")
        st.write(df_ag["TICKET_ESTADO"].value_counts())
        estado_tabla = (
            abiertos["ESTADO_OPERATIVO"]
            .value_counts()
            .reindex(["🔴 Sin revisar","🟠 En proceso","🟡 Escalado"], fill_value=0)
            .reset_index()
        )
        
        estado_tabla.columns = ["ESTADO_OPERATIVO","Tickets"]
    
        fig_estado = px.pie(
            estado_tabla,
            names="ESTADO_OPERATIVO",
            values="Tickets",
            title="Estado operativo de tickets abiertos"
        )
    
        st.plotly_chart(fig_estado, use_container_width=True, key="estado_operativo")

        
    
        # ===============================
        # BACKLOG POR GRUPO
        # ===============================
        
        st.subheader("Backlog de tickets abiertos por grupo")
        
        backlog = (
            abiertos.groupby(["GRUPO","ESTADO_OPERATIVO"])
            .size()
            .reset_index(name="Tickets")
        )
        
        fig_backlog = px.bar(
            backlog,
            x="GRUPO",
            y="Tickets",
            color="ESTADO_OPERATIVO",
            title="Distribución de tickets abiertos por grupo"
        )
        
        st.plotly_chart(
            fig_backlog,
            use_container_width=True,
            key="backlog_grupo"
        )


    # ===============================
    # TICKETS SIN REVISAR POR GRUPO
    # ===============================
    
    st.subheader("Tickets sin revisar por grupo")
    
    sin_revisar = abiertos[abiertos["TICKET_ESTADO"] == "Sin revisar"]
    
    tabla_sin_revisar = (
        sin_revisar.groupby("GRUPO")
        .size()
        .reset_index(name="Tickets sin revisar")
    )
    
    if len(tabla_sin_revisar) > 0:
    
        fig_sin_revisar = px.bar(
            tabla_sin_revisar,
            x="GRUPO",
            y="Tickets sin revisar",
            title="Backlog de tickets sin revisar por grupo",
            color="Tickets sin revisar"
        )
    
        st.plotly_chart(
            fig_sin_revisar,
            use_container_width=True,
            key="sin_revisar_grupo"
        )
    
    else:
        st.info("No hay tickets sin revisar en los filtros actuales")
    # ===============================
    # MEJORA PRO: KPIs + RANKING + RIESGO SLA (tickets abiertos)
    # ===============================
    
    st.divider()
    st.subheader("Análisis avanzado de tickets abiertos")
    # ===============================
    # DETECCIÓN DE TICKETS ESTANCADOS
    # ===============================
    
    tickets_estancados = pd.DataFrame()
    
    if len(abiertos) > 0:
    
        tickets_estancados = abiertos[
            ((abiertos["TICKET_ESTADO"] == "En Proceso") & (abiertos["DIAS"] > 3)) |
            ((abiertos["TICKET_ESTADO"] == "Escalado") & (abiertos["DIAS"] > 5))
        ]
    
    # ===============================
    # KPIs
    # ===============================
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_abiertos = len(abiertos)
    promedio_dias_abiertos = round(abiertos["DIAS"].mean(), 2) if total_abiertos > 0 else 0
    
    riesgo_abiertos = abiertos[abiertos["DIAS"] > 5]
    criticos_abiertos = abiertos[abiertos["DIAS"] > 7]
    
    pct_riesgo = round((len(riesgo_abiertos) / total_abiertos) * 100, 2) if total_abiertos > 0 else 0
    pct_criticos = round((len(criticos_abiertos) / total_abiertos) * 100, 2) if total_abiertos > 0 else 0
    
    col1.metric("Tickets abiertos", total_abiertos)
    col2.metric("Promedio días abiertos", promedio_dias_abiertos)
    col3.metric("% en riesgo SLA", pct_riesgo)
    col4.metric("% críticos (>7 días)", pct_criticos)
    col5.metric(" Tickets estancados", len(tickets_estancados))
    if len(tickets_estancados) > 0:
        st.error(f" {len(tickets_estancados)} tickets estancados detectados")
    
    # ===============================
    # Ranking agentes con más abiertos
    # ===============================
    
    st.subheader("Ranking de agentes con más tickets abiertos")
    
    ranking_abiertos = (
        abiertos.groupby("AGENTE")
        .size()
        .reset_index(name="Tickets abiertos")
        .sort_values("Tickets abiertos", ascending=False)
    )
    
    st.dataframe(ranking_abiertos, use_container_width=True)
    
    fig_rank = px.bar(
        ranking_abiertos,
        x="AGENTE",
        y="Tickets abiertos",
        title="Carga de tickets abiertos por agente"
    )
    
    st.plotly_chart(fig_rank, use_container_width=True, key="ranking_abiertos_agente")
    
    
    # ===============================
    # Riesgo SLA en tickets abiertos
    # ===============================
    
    st.subheader("Riesgo SLA en tickets abiertos")
    
    abiertos["RIESGO_SLA"] = np.where(
        abiertos["DIAS"] <= 3,
        "🟢 Normal",
        np.where(abiertos["DIAS"] <= 5, "🟡 En riesgo", "🔴 Crítico")
    )
    
    riesgo_tabla = (
        abiertos.groupby("RIESGO_SLA")
        .size()
        .reset_index(name="Tickets")
    )
    
    fig_riesgo = px.pie(
        riesgo_tabla,
        names="RIESGO_SLA",
        values="Tickets",
        title="Estado SLA de tickets abiertos"
    )
    
    st.plotly_chart(fig_riesgo, use_container_width=True, key="riesgo_sla_abiertos")
    
    
    
    # ===============================
    # TICKETS POR AGENTE Y GRUPO
    # ===============================

    st.subheader(" Tickets por agente dentro de cada grupo")

    fig_ag_gr = px.bar(
        ranking_gr,
        x="AGENTE",
        y="Tickets",
        color="GRUPO",
        title="Tickets por agente y grupo"
    )
    
    st.plotly_chart(
        fig_ag_gr,
        use_container_width=True,
        key="tickets_agente_grupo"
    )



    # ===============================
    # HEATMAP EXPERTISE AGENTE vs TIPO INCIDENTE
    # ===============================
    
    st.subheader(" Expertise por agente (Tipo de incidente)")
    
    # validar que existan columnas necesarias
    if "TIPO_INCIDENTE" not in df_ag.columns:
        st.warning("No existe la columna TIPO_INCIDENTE en el dataset.")
    else:
    
        # matriz de frecuencia
        matriz = pd.crosstab(
            df_ag["AGENTE"],
            df_ag["TIPO_INCIDENTE"]
        )
    
        if matriz.shape[0] == 0:
            st.info("No hay datos para construir el heatmap con los filtros actuales.")
        else:
    
            fig_heat = px.imshow(
                matriz,
                text_auto=True,
                aspect="auto",
                title="Distribución de tickets por agente y tipo de incidente"
            )
    
            st.plotly_chart(
                fig_heat,
                use_container_width=True,
                key="heatmap_expertise_agentes"
            )
    
            # ===============================
            # TOP AGENTES POR TIPO INCIDENTE
            # ===============================
    
            st.subheader(" Top agente por tipo de incidente")
    
            top_agentes = (
                df_ag.groupby(["TIPO_INCIDENTE","AGENTE"])
                .size()
                .reset_index(name="Tickets")
                .sort_values(["TIPO_INCIDENTE","Tickets"], ascending=[True,False])
            )
    
            top_agentes = top_agentes.groupby("TIPO_INCIDENTE").head(1)
    
            st.dataframe(
                top_agentes,
                use_container_width=True
            )

    # ===============================
    # RECOMENDACIÓN DE AGENTE
    # ===============================
    
    st.subheader(" Recomendación automática de agente")
    
    if "TIPO_INCIDENTE" not in df_ag.columns:
        st.warning("No existe la columna TIPO_INCIDENTE")
    else:
    
        col1, col2 = st.columns(2)
    
        with col1:
            tipo_ticket = st.selectbox(
                "Tipo de incidente",
                sorted(df_ag["TIPO_INCIDENTE"].dropna().unique()),
                key="tipo_recomendacion"
            )
    
        with col2:
            grupo_ticket = st.selectbox(
                "Grupo",
                sorted(df_ag["GRUPO"].dropna().unique()),
                key="grupo_recomendacion"
            )
    
        # filtrar histórico
        df_hist = df_ag[
            (df_ag["TIPO_INCIDENTE"] == tipo_ticket) &
            (df_ag["GRUPO"] == grupo_ticket)
        ]
    
        if len(df_hist) == 0:
    
            st.warning("No hay histórico suficiente para recomendar agente.")
    
        else:
    
            ranking_agentes = (
                df_hist.groupby("AGENTE")
                .agg(
                    Tickets=("TICKET_ID","count"),
                    Promedio_dias=("DIAS","mean")
                )
                .reset_index()
            )
    
            ranking_agentes = ranking_agentes.sort_values(
                ["Promedio_dias","Tickets"],
                ascending=[True,False]
            )
    
            mejor_agente = ranking_agentes.iloc[0]
    
            st.success(
                f"Agente recomendado: **{mejor_agente['AGENTE']}**"
            )
    
            st.info(
                f"Promedio resolución: {round(mejor_agente['Promedio_dias'],2)} días"
            )
    
            st.dataframe(
                ranking_agentes,
                use_container_width=True
            )



    # ===============================
    # ALERTA TEMPRANA DE INCUMPLIMIENTO SLA
    # ===============================
    
    st.subheader("Alerta temprana de riesgo de incumplimiento SLA")
    
    # Reutiliza el dataset filtrado del tab (df_ag)
    try:
        # Ejecutar predicción sobre el dataset actual del tab
        df_riesgo = predecir_dataset(df_ag.copy(), modelo, vectorizer, encoder)
    
        # Umbral configurable
        colu1, colu2 = st.columns([2,1])
        with colu1:
            umbral = st.slider(
                "Umbral de alerta (probabilidad de riesgo)",
                min_value=0.50, max_value=0.95, value=0.80, step=0.05,
                key="umbral_alerta_sla"
            )
        with colu2:
            st.caption("Se listan tickets con PROB_RIESGO ≥ umbral")
    
        # Tickets con alto riesgo
        alto_riesgo = df_riesgo[df_riesgo["PROB_RIESGO"] >= umbral]
    
        col1, col2, col3 = st.columns(3)
        col1.metric("Tickets analizados", len(df_riesgo))
        col2.metric("Tickets en alto riesgo", len(alto_riesgo))
        col3.metric("% alto riesgo", round((len(alto_riesgo)/max(len(df_riesgo),1))*100,2))
    
        # -------------------------
        # Gráfico distribución riesgo
        # -------------------------
        fig_dist = px.histogram(
            df_riesgo,
            x="PROB_RIESGO",
            nbins=30,
            title="Distribución de probabilidad de riesgo SLA"
        )
    
        fig_dist.add_vline(
            x=umbral,
            line_dash="dash",
            line_color="red",
            annotation_text="Umbral alerta"
        )
    
        st.plotly_chart(
            fig_dist,
            use_container_width=True,
            key="dist_riesgo_sla"
        )
    
        # -------------------------
        # Tabla tickets en riesgo
        # -------------------------
        st.subheader(" Tickets con alta probabilidad de incumplir SLA")
    
        if len(alto_riesgo) == 0:
            st.success("No hay tickets que superen el umbral de alerta.")
        else:
            cols = [
                c for c in [
                    "TICKET_ID","TICKET_ASUNTO","AGENTE","GRUPO",
                    "PRIORIDAD","DIAS","PROB_RIESGO"
                ] if c in alto_riesgo.columns
            ]
    
            st.dataframe(
                alto_riesgo[cols].sort_values("PROB_RIESGO", ascending=False),
                use_container_width=True
            )
    
            # gráfico por agente
            if "AGENTE" in alto_riesgo.columns:
                riesgo_agente = (
                    alto_riesgo.groupby("AGENTE")
                    .size()
                    .reset_index(name="Tickets en riesgo")
                    .sort_values("Tickets en riesgo", ascending=False)
                )
    
                fig_riesgo_ag = px.bar(
                    riesgo_agente,
                    x="AGENTE",
                    y="Tickets en riesgo",
                    title="Tickets en alto riesgo por agente"
                )
    
                st.plotly_chart(
                    fig_riesgo_ag,
                    use_container_width=True,
                    key="riesgo_por_agente"
                )
    
    except Exception as e:
        st.error(f"No se pudo calcular la alerta temprana: {e}")




































