import streamlit as st
import plotly.express as px
import pandas as pd

from data_loader import cargar_datos
from model_utils import cargar_modelo
from visualizations import grafico_histograma, grafico_sla
from config import SLA_COLORS

st.set_page_config(page_title="Sistema Inteligente de Tickets", layout="wide")

st.title("Sistema Inteligente de Monitoreo HelpDesk")

# Sidebar
st.sidebar.header("Fuente de datos")

base_datos = st.sidebar.selectbox(
    "Seleccionar base",
    ["TicketsHD.xlsx","TicketsE.xlsx"]
)

st.sidebar.header("Comparación")

base1 = st.sidebar.selectbox("Base 1",["TicketsHD.xlsx","TicketsE.xlsx"])
base2 = st.sidebar.selectbox("Base 2",["TicketsHD.xlsx","TicketsE.xlsx"],index=1)

# Cargar datos
df = cargar_datos(base_datos)

# Tabs
tab1, tab2 = st.tabs(["Resumen","Comparación"])

# =============================
# RESUMEN
# =============================

with tab1:

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Tickets",len(df))
    col2.metric("Promedio días",round(df["DIAS"].mean(),2))
    col3.metric("% Riesgo",round(df["RIESGO_OPERATIVO"].mean()*100,2))
    col4.metric("% Demora crítica",round(df["DEMORA_CRITICA"].mean()*100,2))

    colA,colB = st.columns(2)

    with colA:
        fig = grafico_histograma(df)
        st.plotly_chart(fig,use_container_width=True)

    with colB:
        fig = grafico_sla(df,SLA_COLORS)
        st.plotly_chart(fig,use_container_width=True)

# =============================
# COMPARACION
# =============================

with tab2:

    st.subheader("Comparación entre bases")

    if base1 == base2:
        st.warning("Selecciona dos bases diferentes")
        st.stop()

    df1 = cargar_datos(base1)
    df2 = cargar_datos(base2)

    comparacion = pd.DataFrame({

        "Base":[base1,base2],

        "Total":[len(df1),len(df2)],

        "Promedio días":[
            round(df1["DIAS"].mean(),2),
            round(df2["DIAS"].mean(),2)
        ],

        "% Riesgo":[
            round(df1["RIESGO_OPERATIVO"].mean()*100,2),
            round(df2["RIESGO_OPERATIVO"].mean()*100,2)
        ]

    })

    st.dataframe(comparacion)

    fig = px.bar(
        comparacion,
        x="Base",
        y="% Riesgo",
        title="Comparación de riesgo"
    )

    st.plotly_chart(fig,use_container_width=True)
