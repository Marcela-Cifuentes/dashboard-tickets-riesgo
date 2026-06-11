from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

import pipeline
from pipeline import DATA_DIR, SLA_CRITICO_DIAS, SLA_RIESGO_DIAS, URLS_BASES

# ===============================
# CONFIGURACIÓN DE PÁGINA
# ===============================

st.set_page_config(page_title="Sistema Inteligente de Tickets", layout="wide")

st.title("Sistema Inteligente de Monitoreo HelpDesk")
st.caption("Analítica predictiva y monitoreo de riesgo operativo")

SLA_COLORS = {
    "🟢 Dentro SLA": "#2ecc71",
    "🟡 En riesgo": "#f1c40f",
    "🔴 Fuera SLA": "#e74c3c",
}

SENTIMENT_COLORS = {
    "Negativo": "#8C0000",
    "Neutro": "#27B0F5",
    "Positivo": "#008C36",
}

TTL_DATOS = 3600  # 1 hora: frescura alineada con la ejecución del ETL


# ===============================
# CARGA DE DATOS (solo lectura)
# ===============================

@st.cache_data(ttl=TTL_DATOS, show_spinner="Cargando datos procesados...")
def cargar_datos(nombre_base: str) -> pd.DataFrame:
    """Lee el parquet procesado por el ETL. Fallback: ejecuta el pipeline.

    El fallback garantiza que la app funcione aunque el ETL aún no haya
    corrido, pero el camino normal (parquet) es 10-50x más rápido que
    leer y procesar el Excel.
    """
    ruta = DATA_DIR / f"{nombre_base}.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)

    st.toast(
        f"No existe {ruta}. Ejecutando pipeline completo (modo fallback). "
        "Programa etl.py para acelerar la carga.",
        icon="⚠️",
    )
    return pipeline.procesar_base(nombre_base)


@st.cache_data(ttl=300)
def leer_metadata() -> dict | None:
    ruta = DATA_DIR / "metadata.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return None


@st.cache_resource(show_spinner="Cargando modelo predictivo...")
def cargar_modelo():
    """Artefactos del modelo, cargados UNA vez por proceso (no por rerun)."""
    return pipeline.cargar_modelo()


@st.cache_data(ttl=TTL_DATOS)
def opciones_filtros(nombre_base: str) -> dict:
    """Valores únicos para los widgets de filtro, calculados una vez."""
    df = cargar_datos(nombre_base)
    return {
        "grupos": sorted(df["GRUPO"].dropna().unique().tolist()),
        "prioridades": sorted(df["PRIORIDAD"].dropna().unique().tolist()),
        "origenes": sorted(df["ORIGEN"].dropna().unique().tolist()),
        "fecha_min": df["CREACION"].min().date(),
        "fecha_max": df["CREACION"].max().date(),
        "dias_max": float(np.ceil(df["DIAS"].max())) or 1.0,
    }


@st.cache_data(ttl=TTL_DATOS)
def metricas_modelo(nombre_base: str):
    """ROC y matriz de confusión sobre la base completa, cacheadas.

    Nota metodológica: si el modelo fue entrenado con esta misma base, estas
    métricas están infladas (no hay hold-out). Mantener un set de validación
    separado en el pipeline de entrenamiento.
    """
    df = cargar_datos(nombre_base)
    df_eval = df.dropna(subset=["PROB_RIESGO"])
    if df_eval.empty:
        return None
    y_true = df_eval["RIESGO_OPERATIVO"]
    y_score = df_eval["PROB_RIESGO"]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    cm = confusion_matrix(y_true, (y_score > 0.5).astype(int))
    return fpr, tpr, auc(fpr, tpr), cm


@st.cache_data(ttl=TTL_DATOS)
def top_palabras(nombre_base: str, top_n: int = 20) -> pd.DataFrame:
    """Frecuencia de palabras sobre TEXTO_LIMPIO (ya limpiado en el ETL)."""
    df = cargar_datos(nombre_base)
    conteo = Counter(" ".join(df["TEXTO_LIMPIO"].fillna("")).split())
    return pd.DataFrame(conteo.most_common(top_n), columns=["Palabra", "Frecuencia"])


def palabras_recurrentes(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Versión sobre el subconjunto filtrado (sin re-limpiar texto)."""
    conteo = Counter(" ".join(df["TEXTO_LIMPIO"].fillna("")).split())
    return pd.DataFrame(conteo.most_common(top_n), columns=["Palabra", "Frecuencia"])


# ===============================
# SIDEBAR: FUENTE DE DATOS
# ===============================

st.sidebar.header("Fuente de datos")

base_datos = st.sidebar.selectbox("Seleccionar base", list(URLS_BASES.keys()))

meta = leer_metadata()
if meta:
    st.sidebar.caption(f"🕐 Última actualización ETL: {meta['actualizado_utc'][:16]} UTC")
else:
    st.sidebar.caption("🕐 ETL aún no ejecutado (modo fallback)")

if st.sidebar.button("🔄 Actualizar ahora", use_container_width=True):
    # Invalida SOLO las cachés de datos (no la caché global de toda la app)
    cargar_datos.clear()
    leer_metadata.clear()
    opciones_filtros.clear()
    metricas_modelo.clear()
    top_palabras.clear()
    st.rerun()

df = cargar_datos(base_datos)
ops = opciones_filtros(base_datos)

# ===============================
# SIDEBAR: FILTROS
# ===============================

FILTER_KEYS = [
    "f_grupos", "f_agentes", "f_prioridades", "f_origenes",
    "f_fechas", "f_dias", "f_busqueda",
]


def limpiar_filtros() -> None:
    """Borra las claves de los widgets: cada uno vuelve a su valor por defecto."""
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)


st.sidebar.header("Filtros")
st.sidebar.button("🧹 Limpiar filtros", on_click=limpiar_filtros, use_container_width=True)

grupos_sel = st.sidebar.multiselect(
    "Grupo", ops["grupos"], key="f_grupos",
    placeholder="Todos los grupos",
)

# --- FILTRO DEPENDIENTE: agentes disponibles según grupos elegidos ---
tiene_agente = "AGENTE" in df.columns
agentes_sel: list = []
if tiene_agente:
    base_agentes = df[df["GRUPO"].isin(grupos_sel)] if grupos_sel else df
    agentes_disponibles = sorted(base_agentes["AGENTE"].dropna().unique().tolist())
    # Sanea selecciones que dejaron de ser válidas al cambiar el grupo
    if "f_agentes" in st.session_state:
        st.session_state.f_agentes = [
            a for a in st.session_state.f_agentes if a in agentes_disponibles
        ]
    agentes_sel = st.sidebar.multiselect(
        "Agente", agentes_disponibles, key="f_agentes",
        placeholder="Todos los agentes",
        help="Opciones limitadas a los grupos seleccionados",
    )

prioridades_sel = st.sidebar.multiselect(
    "Prioridad", ops["prioridades"], key="f_prioridades",
    placeholder="Todas las prioridades",
)
origenes_sel = st.sidebar.multiselect(
    "Origen", ops["origenes"], key="f_origenes",
    placeholder="Todos los orígenes",
)

fechas_sel = st.sidebar.date_input(
    "Rango de creación",
    value=(ops["fecha_min"], ops["fecha_max"]),
    min_value=ops["fecha_min"],
    max_value=ops["fecha_max"],
    key="f_fechas",
)

dias_sel = st.sidebar.slider(
    "Días de resolución",
    min_value=0.0,
    max_value=ops["dias_max"],
    value=(0.0, ops["dias_max"]),
    key="f_dias",
)

busqueda = st.sidebar.text_input(
    "🔍 Buscar en asunto/descripción",
    key="f_busqueda",
    placeholder="ej: vpn, contraseña...",
)


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """UNA sola máscara booleana, UNA sola pasada. Devuelve una vista."""
    mask = pd.Series(True, index=df.index)

    if grupos_sel:
        mask &= df["GRUPO"].isin(grupos_sel)
    if agentes_sel and tiene_agente:
        mask &= df["AGENTE"].isin(agentes_sel)
    if prioridades_sel:
        mask &= df["PRIORIDAD"].isin(prioridades_sel)
    if origenes_sel:
        mask &= df["ORIGEN"].isin(origenes_sel)

    if isinstance(fechas_sel, (list, tuple)) and len(fechas_sel) == 2:
        ini, fin = fechas_sel
        mask &= df["CREACION"].dt.date.between(ini, fin)

    if dias_sel and (dias_sel[0] > 0 or dias_sel[1] < ops["dias_max"]):
        mask &= df["DIAS"].between(dias_sel[0], dias_sel[1])

    q = busqueda.strip()
    if q:
        # regex=False -> búsqueda literal, sin compilar expresiones por término
        mask &= df["TEXTO_COMPLETO"].str.contains(q, case=False, na=False, regex=False)

    return df[mask]


df_filtrado = aplicar_filtros(df)

st.sidebar.caption(f"📊 {len(df_filtrado):,} de {len(df):,} tickets")

if df_filtrado.empty:
    st.warning("Ningún ticket cumple los filtros actuales. Ajusta o limpia los filtros.")
    st.stop()

# ===============================
# SIDEBAR: ALERTAS OPERACIONALES
# ===============================

st.sidebar.markdown("## Estado Operacional")

n_criticos = int((df_filtrado["DIAS"] > SLA_CRITICO_DIAS).sum())
n_riesgo = int((df_filtrado["DIAS"] > SLA_RIESGO_DIAS).sum())

if n_criticos > 0:
    st.sidebar.error(f"🔴 {n_criticos} tickets fuera SLA")
elif n_riesgo > 0:
    st.sidebar.warning(f"🟡 {n_riesgo} tickets en riesgo")
else:
    st.sidebar.success("🟢 Operación estable")

# ===============================
# TABS
# ===============================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Resumen",
    "Operación",
    "Riesgo",
    "Modelo",
    "Comparación",
    "Agentes",
    "Experiencia Usuario",
])

# ===============================
# TAB 1 — RESUMEN
# ===============================

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets", f"{len(df_filtrado):,}")
    col2.metric("Promedio días resolución", round(df_filtrado["DIAS"].mean(), 2))
    col3.metric("% Riesgo >5 días", round(df_filtrado["RIESGO_OPERATIVO"].mean() * 100, 2))
    col4.metric("% Demora crítica", round(df_filtrado["DEMORA_CRITICA"].mean() * 100, 2))

    st.divider()

    hoy = pd.Timestamp.today().normalize()
    n_hoy = int((df_filtrado["CREACION"].dt.normalize() == hoy).sum())

    mask_backlog = df_filtrado["TICKET_ESTADO"].isin(["Sin revisar", "En Proceso", "Escalado"])
    backlog = df_filtrado[mask_backlog]
    n_resueltos = int((df_filtrado["TICKET_ESTADO"] == "Resuelto").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickets creados hoy", n_hoy)
    col2.metric("Backlog actual", len(backlog))
    col3.metric("Mediana resolución (días)", round(df_filtrado["DIAS"].median(), 2))
    col4.metric(
        "Ticket abierto más antiguo",
        round(backlog["DIAS"].max(), 2) if len(backlog) else 0,
    )

    st.metric("Tasa de resolución", f"{round(n_resueltos / len(df_filtrado) * 100, 2)}%")

    if len(backlog) > 200:
        st.error("🔴 Operación saturada")
    elif len(backlog) > 100:
        st.warning("🟡 Operación en riesgo")
    else:
        st.success("🟢 Operación estable")

    st.divider()

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Distribución de días de resolución")
        fig_hist = px.histogram(
            df_filtrado[df_filtrado["DIAS"] <= 30], x="DIAS", nbins=30,
            title="Distribución de tiempo de resolución",
        )
        fig_hist.add_vline(x=3, line_dash="dash", line_color="green")
        fig_hist.add_vline(x=5, line_dash="dash", line_color="orange")
        fig_hist.add_vline(x=7, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

    with colB:
        st.subheader("Estado SLA")
        fig_sla = px.pie(
            df_filtrado, names="ESTADO_SLA", color="ESTADO_SLA",
            color_discrete_map=SLA_COLORS, title="Distribución SLA",
        )
        st.plotly_chart(fig_sla, use_container_width=True)

    st.divider()

    st.subheader("Distribución de tickets por grupo")
    tickets_grupo = (
        df_filtrado.groupby("GRUPO", observed=True).size()
        .reset_index(name="Tickets").sort_values("Tickets", ascending=False)
    )
    st.plotly_chart(
        px.bar(tickets_grupo, x="GRUPO", y="Tickets", title="Volumen de tickets por grupo"),
        use_container_width=True,
    )

    st.subheader("Tendencia semanal de tickets")
    tickets_semana = (
        df_filtrado.assign(SEMANA=df_filtrado["CREACION"].dt.to_period("W").astype(str))
        .groupby("SEMANA").size().reset_index(name="Tickets")
    )
    st.plotly_chart(
        px.line(tickets_semana, x="SEMANA", y="Tickets", markers=True,
                title="Evolución semanal de tickets"),
        use_container_width=True,
    )

# ===============================
# TAB 2 — OPERACIÓN
# ===============================

with tab2:
    tickets_mes = (
        df_filtrado.assign(MES=df_filtrado["CREACION"].dt.to_period("M").astype(str))
        .groupby("MES").size().reset_index(name="Tickets")
    )
    st.plotly_chart(
        px.line(tickets_mes, x="MES", y="Tickets", markers=True,
                title="Evolución mensual de tickets"),
        use_container_width=True,
    )

    st.plotly_chart(
        px.pie(df_filtrado, names="PRIORIDAD", title="Distribución por prioridad"),
        use_container_width=True,
    )

    tickets_origen = (
        df_filtrado.groupby("ORIGEN", observed=True).size().reset_index(name="Tickets")
    )
    st.plotly_chart(
        px.bar(tickets_origen, x="ORIGEN", y="Tickets", title="Tickets por origen"),
        use_container_width=True,
    )

    tabla_heat = pd.crosstab(df_filtrado["GRUPO"], df_filtrado["PRIORIDAD"])
    st.plotly_chart(
        px.imshow(tabla_heat, text_auto=True, title="Grupo vs Prioridad"),
        use_container_width=True,
    )

    st.subheader("Tipos de incidentes detectados")
    tickets_inc = (
        df_filtrado.groupby("TIPO_INCIDENTE", observed=True).size()
        .reset_index(name="Tickets")
    )
    st.plotly_chart(
        px.bar(tickets_inc, x="TIPO_INCIDENTE", y="Tickets"),
        use_container_width=True,
    )

# ===============================
# TAB 3 — RIESGO
# ===============================

with tab3:
    riesgo_grupo = (
        df_filtrado.groupby("GRUPO", observed=True)["RIESGO_OPERATIVO"]
        .mean().reset_index()
    )
    st.plotly_chart(
        px.bar(riesgo_grupo, x="GRUPO", y="RIESGO_OPERATIVO",
               title="Tasa de riesgo operativo por grupo"),
        use_container_width=True,
    )

    st.plotly_chart(
        px.box(df_filtrado, x="GRUPO", y="DIAS", title="Distribución de días por grupo"),
        use_container_width=True,
    )

    # Palabras más frecuentes: sobre TEXTO_LIMPIO precalculado (sin re-limpieza)
    st.plotly_chart(
        px.bar(palabras_recurrentes(df_filtrado, 20),
               x="Frecuencia", y="Palabra", orientation="h",
               title="Palabras más frecuentes"),
        use_container_width=True,
    )

    # PROB_RIESGO ya viene como columna del ETL: costo cero aquí
    if df_filtrado["PROB_RIESGO"].notna().any():
        st.plotly_chart(
            px.histogram(df_filtrado, x="PROB_RIESGO", nbins=30,
                         title="Distribución de probabilidad de riesgo"),
            use_container_width=True,
        )
    else:
        st.info("PROB_RIESGO no disponible: faltan los artefactos del modelo en el ETL.")

    st.subheader("Tickets críticos")
    criticos = df_filtrado[df_filtrado["DIAS"] > SLA_CRITICO_DIAS]
    cols_criticos = [c for c in ["TICKET_ID", "TICKET_ASUNTO", "GRUPO", "PRIORIDAD", "DIAS"]
                     if c in criticos.columns]
    st.dataframe(criticos[cols_criticos], use_container_width=True)

    # ANOMALIA ya viene del ETL (antes: IsolationForest entrenado en cada rerun)
    st.plotly_chart(
        px.scatter(df_filtrado, x="DIAS", y="PRIORIDAD", color="ANOMALIA",
                   title="Detección de anomalías en tiempos de resolución"),
        use_container_width=True,
    )

# ===============================
# TAB 4 — MODELO
# ===============================

with tab4:
    st.subheader("Predicción de riesgo de nuevo ticket")

    @st.fragment
    def formulario_prediccion():
        """Fragment: interactuar aquí NO re-ejecuta el resto de la app."""
        asunto = st.text_input("Asunto del ticket")
        descripcion = st.text_area("Descripción")
        prioridad = st.selectbox("Prioridad", ops["prioridades"])
        grupo = st.selectbox("Grupo", ops["grupos"])
        origen = st.selectbox("Origen", ops["origenes"])

        if st.button("Predecir riesgo"):
            try:
                modelo, vectorizer, encoder = cargar_modelo()
                proba, nivel = pipeline.predecir_ticket(
                    modelo, vectorizer, encoder,
                    asunto, descripcion, prioridad, grupo, origen,
                )
                st.success(f"Probabilidad de riesgo: {round(proba, 3)}")
                st.info(f"Nivel de riesgo: {nivel}")
            except FileNotFoundError:
                st.error("No se encontraron los artefactos del modelo (.pkl).")
            except Exception as e:  # noqa: BLE001
                st.error(f"Error en la predicción: {e}")

    formulario_prediccion()

    st.divider()
    st.subheader("Desempeño del modelo (base completa)")

    resultado = metricas_modelo(base_datos)
    if resultado is None:
        st.info("No hay predicciones disponibles para evaluar el modelo.")
    else:
        fpr, tpr, roc_auc, cm = resultado
        colROC, colCM = st.columns(2)
        with colROC:
            st.plotly_chart(
                px.line(x=fpr, y=tpr, title=f"Curva ROC (AUC={roc_auc:.2f})",
                        labels={"x": "FPR", "y": "TPR"}),
                use_container_width=True,
            )
        with colCM:
            st.plotly_chart(
                px.imshow(cm, text_auto=True, title="Matriz de confusión (umbral 0.5)"),
                use_container_width=True,
            )
        st.caption(
            "⚠️ Métricas calculadas sobre la base completa. Si el modelo se "
            "entrenó con estos mismos datos, el AUC está inflado: usa un "
            "conjunto de validación separado en el entrenamiento."
        )

# ===============================
# TAB 5 — COMPARACIÓN (carga diferida)
# ===============================

with tab5:
    st.subheader("Comparación entre bases de tickets")

    colb1, colb2 = st.columns(2)
    base1 = colb1.selectbox("Base 1", list(URLS_BASES.keys()), key="comp_base1")
    base2 = colb2.selectbox("Base 2", list(URLS_BASES.keys()), index=1, key="comp_base2")

    if base1 == base2:
        # Antes: st.stop() aquí dejaba en blanco los tabs 6 y 7
        st.warning("Selecciona dos bases diferentes para comparar.")
    elif st.toggle("Cargar comparación", key="comp_activa",
                   help="Carga diferida: la segunda base solo se lee si activas esto"):
        df1 = cargar_datos(base1)
        df2 = cargar_datos(base2)

        comparacion = pd.DataFrame({
            "Base": [base1, base2],
            "Total Tickets": [len(df1), len(df2)],
            "Promedio días": [round(df1["DIAS"].mean(), 2), round(df2["DIAS"].mean(), 2)],
            "% Riesgo": [round(df1["RIESGO_OPERATIVO"].mean() * 100, 2),
                         round(df2["RIESGO_OPERATIVO"].mean() * 100, 2)],
            "% Demora crítica": [round(df1["DEMORA_CRITICA"].mean() * 100, 2),
                                 round(df2["DEMORA_CRITICA"].mean() * 100, 2)],
        })
        st.dataframe(comparacion, use_container_width=True)

        st.plotly_chart(
            px.bar(comparacion, x="Base", y=["% Riesgo", "% Demora crítica"],
                   barmode="group", title="Comparación de riesgo operativo"),
            use_container_width=True,
        )

        # Comparación SLA sin concatenar los DataFrames completos (ahorro de memoria)
        sla_comp = pd.concat([
            df1.groupby("ESTADO_SLA", observed=True).size()
               .reset_index(name="Tickets").assign(Base=base1),
            df2.groupby("ESTADO_SLA", observed=True).size()
               .reset_index(name="Tickets").assign(Base=base2),
        ])
        st.plotly_chart(
            px.bar(sla_comp, x="Base", y="Tickets", color="ESTADO_SLA",
                   barmode="stack", color_discrete_map=SLA_COLORS,
                   title="Comparación de SLA entre bases"),
            use_container_width=True,
        )

# ===============================
# TAB 6 — GESTIÓN AGENTES
# ===============================

with tab6:
    st.header("Gestión operativa de agentes")

    if not tiene_agente:
        # Antes: st.stop() aquí dejaba en blanco el tab 7
        st.warning("La base no contiene columna AGENTE.")
    else:
        # Parte de los filtros GLOBALES (antes los ignoraba) + refinamiento por mes
        df_ag = df_filtrado.assign(
            MES=df_filtrado["CREACION"].dt.to_period("M").astype(str)
        )

        mes_sel = st.selectbox(
            "Mes (refinamiento local)",
            ["Todos"] + sorted(df_ag["MES"].dropna().unique().tolist()),
            key="mes_agentes",
        )
        if mes_sel != "Todos":
            df_ag = df_ag[df_ag["MES"] == mes_sel]

        if df_ag.empty:
            st.info("No hay datos para el mes seleccionado.")
        else:
            st.divider()

            # --- Carga de trabajo ---
            st.subheader("Carga de trabajo por agente")
            carga = (
                df_ag.groupby("AGENTE", observed=True).size()
                .reset_index(name="Tickets").sort_values("Tickets", ascending=False)
            )
            st.plotly_chart(
                px.bar(carga, x="AGENTE", y="Tickets", title="Tickets por agente"),
                use_container_width=True, key="carga_agentes",
            )

            # --- SLA por agente ---
            st.subheader("Cumplimiento SLA por agente")
            sla = (
                df_ag.groupby("AGENTE", observed=True)["DIAS"]
                .apply(lambda x: (x <= SLA_RIESGO_DIAS).mean() * 100)
                .reset_index(name="SLA_%")
            )
            st.plotly_chart(
                px.bar(sla, x="AGENTE", y="SLA_%"),
                use_container_width=True, key="sla_agentes",
            )

            # --- Ranking ---
            st.subheader("Ranking de desempeño")
            ranking = (
                df_ag.groupby("AGENTE", observed=True)
                .agg(
                    Tickets=("TICKET_ID", "count"),
                    Promedio_dias=("DIAS", "mean"),
                    SLA=("DIAS", lambda x: (x <= SLA_RIESGO_DIAS).mean() * 100),
                )
                .reset_index().sort_values("SLA", ascending=False)
            )
            st.dataframe(ranking, use_container_width=True)

            # --- Productividad mensual ---
            st.subheader("Productividad mensual")
            prod = (
                df_ag.groupby(["MES", "AGENTE"], observed=True).size()
                .reset_index(name="Tickets")
            )
            st.plotly_chart(
                px.line(prod, x="MES", y="Tickets", color="AGENTE", markers=True),
                use_container_width=True, key="productividad_agentes",
            )

            # --- Saturación ---
            st.subheader("Detección de agentes saturados")
            limite = carga["Tickets"].mean() * 1.5
            carga["Estado"] = np.where(carga["Tickets"] > limite, "Sobrecarga", "Normal")
            st.plotly_chart(
                px.bar(carga, x="AGENTE", y="Tickets", color="Estado"),
                use_container_width=True, key="saturacion_agentes",
            )

            # --- Agente ↔ Grupo ---
            st.subheader("Relación agente y grupo")
            st.dataframe(
                df_ag[["AGENTE", "GRUPO"]].drop_duplicates()
                .sort_values(["GRUPO", "AGENTE"]),
                use_container_width=True,
            )

            st.subheader("Ranking por agente y grupo")
            ranking_gr = (
                df_ag.groupby(["GRUPO", "AGENTE"], observed=True)
                .agg(Tickets=("TICKET_ID", "count"), Promedio_dias=("DIAS", "mean"))
                .reset_index()
            )
            st.dataframe(ranking_gr, use_container_width=True)

            # --- Incumplimiento por grupo ---
            st.subheader("Incumplimiento SLA por grupo")
            sla_grupo = (
                df_ag.groupby("GRUPO", observed=True)["DIAS"]
                .apply(lambda x: (x > SLA_RIESGO_DIAS).mean() * 100)
                .reset_index(name="Incumplimiento_%")
            )
            st.plotly_chart(
                px.bar(sla_grupo, x="GRUPO", y="Incumplimiento_%"),
                use_container_width=True, key="sla_grupo",
            )

            # ===============================
            # TICKETS ABIERTOS
            # ===============================
            st.subheader("Tickets no resueltos")

            abiertos = df_ag[
                df_ag["TICKET_ESTADO"].isin(["Sin revisar", "En Proceso", "Escalado"])
            ].copy()

            abiertos["ESTADO_OPERATIVO"] = np.select(
                [
                    abiertos["TICKET_ESTADO"] == "Sin revisar",
                    abiertos["TICKET_ESTADO"] == "En Proceso",
                ],
                ["🔴 Sin revisar", "🟠 En proceso"],
                default="🟡 Escalado",
            )

            with st.expander("🔍 Diagnóstico: estados presentes en el dataset"):
                st.write(df_ag["TICKET_ESTADO"].value_counts(dropna=False))

            if abiertos.empty:
                st.success("No hay tickets pendientes")
            else:
                tabla = (
                    abiertos.groupby(["AGENTE", "GRUPO"], observed=True).size()
                    .reset_index(name="Tickets abiertos")
                )
                st.dataframe(tabla, use_container_width=True)
                st.plotly_chart(
                    px.bar(tabla, x="AGENTE", y="Tickets abiertos", color="GRUPO",
                           title="Tickets abiertos por agente"),
                    use_container_width=True, key="tickets_abiertos",
                )

                estado_tabla = (
                    abiertos["ESTADO_OPERATIVO"].value_counts()
                    .reindex(["🔴 Sin revisar", "🟠 En proceso", "🟡 Escalado"], fill_value=0)
                    .reset_index()
                )
                estado_tabla.columns = ["ESTADO_OPERATIVO", "Tickets"]
                st.plotly_chart(
                    px.pie(estado_tabla, names="ESTADO_OPERATIVO", values="Tickets",
                           title="Estado operativo de tickets abiertos"),
                    use_container_width=True, key="estado_operativo",
                )

                st.subheader("Backlog de tickets abiertos por grupo")
                backlog_gr = (
                    abiertos.groupby(["GRUPO", "ESTADO_OPERATIVO"], observed=True)
                    .size().reset_index(name="Tickets")
                )
                st.plotly_chart(
                    px.bar(backlog_gr, x="GRUPO", y="Tickets", color="ESTADO_OPERATIVO",
                           title="Distribución de tickets abiertos por grupo"),
                    use_container_width=True, key="backlog_grupo",
                )

                st.subheader("Tickets sin revisar por grupo")
                sin_revisar = (
                    abiertos[abiertos["TICKET_ESTADO"] == "Sin revisar"]
                    .groupby("GRUPO", observed=True).size()
                    .reset_index(name="Tickets sin revisar")
                )
                if len(sin_revisar):
                    st.plotly_chart(
                        px.bar(sin_revisar, x="GRUPO", y="Tickets sin revisar",
                               color="Tickets sin revisar",
                               title="Backlog de tickets sin revisar por grupo"),
                        use_container_width=True, key="sin_revisar_grupo",
                    )
                else:
                    st.info("No hay tickets sin revisar en los filtros actuales")

                # ===============================
                # ANÁLISIS AVANZADO DE ABIERTOS
                # ===============================
                st.divider()
                st.subheader("Análisis avanzado de tickets abiertos")

                tickets_estancados = abiertos[
                    ((abiertos["TICKET_ESTADO"] == "En Proceso") & (abiertos["DIAS"] > 3))
                    | ((abiertos["TICKET_ESTADO"] == "Escalado") & (abiertos["DIAS"] > 5))
                ]

                total_abiertos = len(abiertos)
                riesgo_abiertos = int((abiertos["DIAS"] > SLA_RIESGO_DIAS).sum())
                criticos_abiertos = int((abiertos["DIAS"] > SLA_CRITICO_DIAS).sum())

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Tickets abiertos", total_abiertos)
                c2.metric("Promedio días abiertos", round(abiertos["DIAS"].mean(), 2))
                c3.metric("% en riesgo SLA", round(riesgo_abiertos / total_abiertos * 100, 2))
                c4.metric("% críticos (>7 días)", round(criticos_abiertos / total_abiertos * 100, 2))
                c5.metric("Tickets estancados", len(tickets_estancados))

                if len(tickets_estancados):
                    st.error(f"{len(tickets_estancados)} tickets estancados detectados")

                st.subheader("Ranking de agentes con más tickets abiertos")
                ranking_abiertos = (
                    abiertos.groupby("AGENTE", observed=True).size()
                    .reset_index(name="Tickets abiertos")
                    .sort_values("Tickets abiertos", ascending=False)
                )
                st.dataframe(ranking_abiertos, use_container_width=True)
                st.plotly_chart(
                    px.bar(ranking_abiertos, x="AGENTE", y="Tickets abiertos",
                           title="Carga de tickets abiertos por agente"),
                    use_container_width=True, key="ranking_abiertos_agente",
                )

                st.subheader("Riesgo SLA en tickets abiertos")
                abiertos["RIESGO_SLA"] = np.select(
                    [abiertos["DIAS"] <= 3, abiertos["DIAS"] <= 5],
                    ["🟢 Normal", "🟡 En riesgo"],
                    default="🔴 Crítico",
                )
                riesgo_tabla = (
                    abiertos.groupby("RIESGO_SLA", observed=True).size()
                    .reset_index(name="Tickets")
                )
                st.plotly_chart(
                    px.pie(riesgo_tabla, names="RIESGO_SLA", values="Tickets",
                           title="Estado SLA de tickets abiertos"),
                    use_container_width=True, key="riesgo_sla_abiertos",
                )

            # --- Tickets por agente y grupo ---
            st.subheader("Tickets por agente dentro de cada grupo")
            st.plotly_chart(
                px.bar(ranking_gr, x="AGENTE", y="Tickets", color="GRUPO",
                       title="Tickets por agente y grupo"),
                use_container_width=True, key="tickets_agente_grupo",
            )

            # --- Expertise ---
            st.subheader("Expertise por agente (Tipo de incidente)")
            matriz = pd.crosstab(df_ag["AGENTE"], df_ag["TIPO_INCIDENTE"])
            if matriz.shape[0] == 0:
                st.info("No hay datos para construir el heatmap con los filtros actuales.")
            else:
                st.plotly_chart(
                    px.imshow(matriz, text_auto=True, aspect="auto",
                              title="Distribución de tickets por agente y tipo de incidente"),
                    use_container_width=True, key="heatmap_expertise_agentes",
                )

                st.subheader("Top agente por tipo de incidente")
                top_agentes = (
                    df_ag.groupby(["TIPO_INCIDENTE", "AGENTE"], observed=True).size()
                    .reset_index(name="Tickets")
                    .sort_values(["TIPO_INCIDENTE", "Tickets"], ascending=[True, False])
                    .groupby("TIPO_INCIDENTE", observed=True).head(1)
                )
                st.dataframe(top_agentes, use_container_width=True)

            # ===============================
            # RECOMENDACIÓN DE AGENTE (fragment)
            # ===============================
            st.subheader("Recomendación automática de agente")

            @st.fragment
            def recomendacion_agente(df_ag: pd.DataFrame):
                """Cambiar estos selectbox solo re-ejecuta este bloque."""
                col1, col2 = st.columns(2)
                with col1:
                    tipo_ticket = st.selectbox(
                        "Tipo de incidente",
                        sorted(df_ag["TIPO_INCIDENTE"].dropna().unique().tolist()),
                        key="tipo_recomendacion",
                    )
                with col2:
                    grupo_ticket = st.selectbox(
                        "Grupo",
                        sorted(df_ag["GRUPO"].dropna().unique().tolist()),
                        key="grupo_recomendacion",
                    )

                df_hist = df_ag[
                    (df_ag["TIPO_INCIDENTE"] == tipo_ticket)
                    & (df_ag["GRUPO"] == grupo_ticket)
                ]
                if df_hist.empty:
                    st.warning("No hay histórico suficiente para recomendar agente.")
                    return

                ranking_agentes = (
                    df_hist.groupby("AGENTE", observed=True)
                    .agg(Tickets=("TICKET_ID", "count"), Promedio_dias=("DIAS", "mean"))
                    .reset_index()
                    .sort_values(["Promedio_dias", "Tickets"], ascending=[True, False])
                )
                mejor = ranking_agentes.iloc[0]
                st.success(f"Agente recomendado: **{mejor['AGENTE']}**")
                st.info(f"Promedio resolución: {round(mejor['Promedio_dias'], 2)} días")
                st.dataframe(ranking_agentes, use_container_width=True)

            recomendacion_agente(df_ag)

            # ===============================
            # ALERTA TEMPRANA SLA (fragment)
            # ===============================
            st.subheader("Alerta temprana de riesgo de incumplimiento SLA")

            @st.fragment
            def alerta_temprana(df_ag: pd.DataFrame):
                """Mover el slider solo re-ejecuta este bloque (no toda la app).

                PROB_RIESGO ya viene precalculado del ETL: cero inferencia aquí.
                """
                df_riesgo = df_ag.dropna(subset=["PROB_RIESGO"])
                if df_riesgo.empty:
                    st.info("PROB_RIESGO no disponible (ejecuta el ETL con el modelo).")
                    return

                cu1, cu2 = st.columns([2, 1])
                with cu1:
                    umbral = st.slider(
                        "Umbral de alerta (probabilidad de riesgo)",
                        min_value=0.50, max_value=0.95, value=0.80, step=0.05,
                        key="umbral_alerta_sla",
                    )
                with cu2:
                    st.caption("Se listan tickets con PROB_RIESGO ≥ umbral")

                alto_riesgo = df_riesgo[df_riesgo["PROB_RIESGO"] >= umbral]

                c1, c2, c3 = st.columns(3)
                c1.metric("Tickets analizados", len(df_riesgo))
                c2.metric("Tickets en alto riesgo", len(alto_riesgo))
                c3.metric("% alto riesgo",
                          round(len(alto_riesgo) / max(len(df_riesgo), 1) * 100, 2))

                fig_dist = px.histogram(
                    df_riesgo, x="PROB_RIESGO", nbins=30,
                    title="Distribución de probabilidad de riesgo SLA",
                )
                fig_dist.add_vline(x=umbral, line_dash="dash", line_color="red",
                                   annotation_text="Umbral alerta")
                st.plotly_chart(fig_dist, use_container_width=True, key="dist_riesgo_sla")

                st.subheader("Tickets con alta probabilidad de incumplir SLA")
                if alto_riesgo.empty:
                    st.success("No hay tickets que superen el umbral de alerta.")
                    return

                cols = [c for c in ["TICKET_ID", "TICKET_ASUNTO", "AGENTE", "GRUPO",
                                    "PRIORIDAD", "DIAS", "PROB_RIESGO"]
                        if c in alto_riesgo.columns]
                st.dataframe(
                    alto_riesgo[cols].sort_values("PROB_RIESGO", ascending=False),
                    use_container_width=True,
                )

                riesgo_agente = (
                    alto_riesgo.groupby("AGENTE", observed=True).size()
                    .reset_index(name="Tickets en riesgo")
                    .sort_values("Tickets en riesgo", ascending=False)
                )
                st.plotly_chart(
                    px.bar(riesgo_agente, x="AGENTE", y="Tickets en riesgo",
                           title="Tickets en alto riesgo por agente"),
                    use_container_width=True, key="riesgo_por_agente",
                )

            alerta_temprana(df_ag)

# ===============================
# TAB 7 — EXPERIENCIA DEL USUARIO
# ===============================

with tab7:
    st.header("Experiencia del usuario y sentimiento de tickets")
    st.caption(
        "⚠️ Nota: el sentimiento actual usa VADER (léxico en inglés). Sobre "
        "tickets en español la señal es limitada; se recomienda migrar a "
        "`pysentimiento` en pipeline.py."
    )

    st.subheader("Sentimiento general de los tickets")
    st.plotly_chart(
        px.pie(df_filtrado, names="SENTIMIENTO",
               title="Distribución de sentimiento en tickets",
               color="SENTIMIENTO", color_discrete_map=SENTIMENT_COLORS),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Detección de urgencia en tickets")
    st.plotly_chart(
        px.pie(df_filtrado, names="URGENCIA", title="Tickets con lenguaje de urgencia"),
        use_container_width=True,
    )

    st.subheader("Sentimiento por grupo")
    sent_grupo = (
        df_filtrado.groupby(["GRUPO", "SENTIMIENTO"], observed=True).size()
        .reset_index(name="Tickets")
    )
    st.plotly_chart(
        px.bar(sent_grupo, x="GRUPO", y="Tickets", color="SENTIMIENTO",
               barmode="stack", title="Distribución de sentimiento por grupo",
               color_discrete_map=SENTIMENT_COLORS),
        use_container_width=True,
    )

    st.divider()

    if tiene_agente:
        st.subheader("Sentimiento por agente")
        sent_agente = (
            df_filtrado.groupby(["AGENTE", "SENTIMIENTO"], observed=True).size()
            .reset_index(name="Tickets")
        )
        st.plotly_chart(
            px.bar(sent_agente, x="AGENTE", y="Tickets", color="SENTIMIENTO",
                   barmode="stack", title="Distribución de sentimiento por agente",
                   color_discrete_map=SENTIMENT_COLORS),
            use_container_width=True,
        )
        st.divider()

    # --- Tickets negativos ---
    st.subheader("Tickets con sentimiento negativo")
    negativos = df_filtrado[df_filtrado["SENTIMIENTO"] == "Negativo"]
    if negativos.empty:
        st.success("No hay tickets con sentimiento negativo")
    else:
        cols = [c for c in ["TICKET_ID", "TICKET_ASUNTO", "GRUPO", "AGENTE",
                            "PRIORIDAD", "DIAS"] if c in negativos.columns]
        st.dataframe(negativos[cols], use_container_width=True)

    # --- Tickets urgentes ---
    st.subheader("Tickets detectados como urgentes")
    urgentes = df_filtrado[df_filtrado["URGENCIA"] == "🔥 Alta urgencia"]
    if urgentes.empty:
        st.success("No se detectaron tickets urgentes")
    else:
        cols = [c for c in ["TICKET_ID", "TICKET_ASUNTO", "GRUPO", "AGENTE",
                            "PRIORIDAD", "DIAS"] if c in urgentes.columns]
        st.dataframe(urgentes[cols], use_container_width=True)

    # --- Incidentes recurrentes (fragment) ---
    st.subheader("Detección de incidentes recurrentes")

    @st.fragment
    def incidentes_recurrentes(df_filtrado: pd.DataFrame):
        """Cambiar la palabra clave solo re-ejecuta este bloque."""
        recurrentes = palabras_recurrentes(df_filtrado, 15)
        if recurrentes.empty:
            st.info("No hay texto suficiente para detectar incidentes recurrentes.")
            return

        st.plotly_chart(
            px.bar(recurrentes, x="Frecuencia", y="Palabra", orientation="h",
                   title="Problemas más reportados"),
            use_container_width=True,
        )

        st.subheader("Tickets asociados a incidentes recurrentes")
        palabra_sel = st.selectbox(
            "Seleccionar palabra clave", recurrentes["Palabra"], key="palabra_recurrente",
        )
        tickets_rel = df_filtrado[
            df_filtrado["TEXTO_COMPLETO"].str.contains(
                palabra_sel, case=False, na=False, regex=False
            )
        ]
        cols = [c for c in ["TICKET_ID", "TICKET_ASUNTO", "GRUPO", "AGENTE", "PRIORIDAD"]
                if c in tickets_rel.columns]
        st.dataframe(tickets_rel[cols], use_container_width=True)

    incidentes_recurrentes(df_filtrado)
