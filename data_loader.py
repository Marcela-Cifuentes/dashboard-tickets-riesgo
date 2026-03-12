import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(ttl=60)
def cargar_datos(archivo):

    df = pd.read_excel(archivo)

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