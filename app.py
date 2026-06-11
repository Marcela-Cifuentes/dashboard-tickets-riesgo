"""
app.py — Sistema Inteligente de Monitoreo HelpDesk
====================================================
v4: Filtros avanzados de agentes + fechas + exclusión de inactivos
    + Modelo predictivo mejorado (GBM + feature engineering)
"""

from __future__ import annotations

import datetime
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve, classification_report

import pipeline
from pipeline import DATA_DIR, SLA_CRITICO_DIAS, SLA_RIESGO_DIAS, URLS_BASES

# ═══════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════

st.set_page_config(page_title="Sistema Inteligente de Tickets", layout="wide")
st.title("Sistema Inteligente de Monitoreo HelpDesk")
st.caption("Analítica predictiva y monitoreo de riesgo operativo")

SLA_COLORS = {
    "🟢 Dentro SLA": "#2ecc71",
    "🟡 En riesgo":  "#f1c40f",
    "🔴 Fuera SLA":  "#e74c3c",
}
SENTIMENT_COLORS = {"Negativo": "#8C0000", "Neutro": "#27B0F5", "Positivo": "#008C36"}
TTL_DATOS = 3600
UMBRAL_INACTIVO_DIAS = 90   # agente sin ticket en N días → candidato a inactivo


# ═══════════════════════════════════════════════════
# FUNCIONES CACHEADAS (regla estricta: sin UI, sin cachés anidadas)
# ═══════════════════════════════════════════════════

@st.cache_data(ttl=TTL_DATOS, show_spinner="Cargando datos procesados...")
def cargar_datos(nombre_base: str) -> pd.DataFrame:
    ruta = DATA_DIR / f"{nombre_base}.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    return pipeline.procesar_base(nombre_base)


@st.cache_data(ttl=300)
def leer_metadata() -> dict | None:
    ruta = DATA_DIR / "metadata.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return None


@st.cache_resource(show_spinner="Cargando modelo predictivo...")
def cargar_modelo():
    return pipeline.cargar_modelo()


# ── Helpers sin caché (reciben el df ya cacheado) ───────────────────────────

def _opciones_filtros(df: pd.DataFrame) -> dict:
    tiene_agente = "AGENTE" in df.columns
    opts = {
        "grupos":      sorted(df["GRUPO"].dropna().unique().tolist()),
        "prioridades": sorted(df["PRIORIDAD"].dropna().unique().tolist()),
        "origenes":    sorted(df["ORIGEN"].dropna().unique().tolist()),
        "fecha_min":   df["CREACION"].min().date(),
        "fecha_max":   df["CREACION"].max().date(),
        "dias_max":    float(np.ceil(df["DIAS"].max())) or 1.0,
    }
    if tiene_agente:
        # Fecha del último ticket por agente → detectar inactivos
        ultima = (df.groupby("AGENTE", observed=True)["CREACION"]
                    .max().rename("ultimo_ticket"))
        hoy = pd.Timestamp.today().normalize()
        inactivos = ultima[
            (hoy - ultima).dt.days > UMBRAL_INACTIVO_DIAS
        ].index.tolist()
        activos = [a for a in sorted(df["AGENTE"].dropna().unique().tolist())
                   if a not in inactivos]
        opts["agentes_activos"]  = activos
        opts["agentes_inactivos"] = sorted(inactivos)
        opts["agentes_todos"]    = activos + sorted(inactivos)
    return opts


def _metricas_modelo(df: pd.DataFrame):
    df_eval = df.dropna(subset=["PROB_RIESGO"])
    if df_eval.empty:
        return None
    y_true  = df_eval["RIESGO_OPERATIVO"]
    y_score = df_eval["PROB_RIESGO"]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    cm = confusion_matrix(y_true, (y_score > 0.5).astype(int))
    reporte = classification_report(y_true, (y_score > 0.5).astype(int),
                                    output_dict=True, zero_division=0)
    return fpr, tpr, auc(fpr, tpr), cm, reporte


def palabras_recurrentes(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    col = "TEXTO_LIMPIO" if "TEXTO_LIMPIO" in df.columns else "TEXTO_COMPLETO"
    conteo = Counter(" ".join(df[col].fillna("")).split())
    return pd.DataFrame(conteo.most_common(top_n), columns=["Palabra", "Frecuencia"])


# ═══════════════════════════════════════════════════
# SIDEBAR — FUENTE DE DATOS
# ═══════════════════════════════════════════════════

st.sidebar.header("Fuente de datos")
base_datos = st.sidebar.selectbox("Seleccionar base", list(URLS_BASES.keys()))

meta = leer_metadata()
if meta:
    st.sidebar.caption(f"🕐 ETL: {meta['actualizado_utc'][:16]} UTC")
else:
    st.sidebar.caption("🕐 ETL no ejecutado (modo fallback)")

if st.sidebar.button("🔄 Actualizar ahora", use_container_width=True):
    cargar_datos.clear()
    leer_metadata.clear()
    st.rerun()

_modo_fallback = not (DATA_DIR / f"{base_datos}.parquet").exists()
df = cargar_datos(base_datos)

if _modo_fallback:
    st.sidebar.warning("Modo fallback: procesando desde Excel. "
                       "Ejecuta `python etl.py` para acelerar.", icon="⚠️")

ops          = _opciones_filtros(df)
tiene_agente = "AGENTE" in df.columns


# ═══════════════════════════════════════════════════
# SIDEBAR — FILTROS
# ═══════════════════════════════════════════════════

FILTER_KEYS = [
    "f_grupos", "f_agentes_inc", "f_agentes_exc",
    "f_prioridades", "f_origenes",
    "f_fechas",          # date_input de rango único
    "f_fecha_preset",    # selectbox de preset rápido
    "f_dias", "f_busqueda",
    "f_ocultar_inactivos",
]

def limpiar_filtros():
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)


with st.sidebar:
    st.header("Filtros")
    st.button("🧹 Limpiar filtros", on_click=limpiar_filtros, use_container_width=True)

    # ── Grupos ──────────────────────────────────────────────────────────────
    grupos_sel = st.multiselect(
        "Grupo", ops["grupos"], key="f_grupos",
        placeholder="Todos los grupos",
    )

    # ── Agentes — con gestión de inactivos ──────────────────────────────────
    agentes_sel: list = []
    agentes_excluidos: list = []

    if tiene_agente:
        st.markdown("**Agentes**")

        # Toggle para mostrar/ocultar inactivos
        ocultar_inactivos = st.toggle(
            f"Ocultar inactivos (sin ticket >{UMBRAL_INACTIVO_DIAS}d)",
            value=True,
            key="f_ocultar_inactivos",
            help=f"Agentes sin ningún ticket en los últimos {UMBRAL_INACTIVO_DIAS} días "
                 f"se consideran inactivos.",
        )

        # Pool base según grupos elegidos
        base_ag = df[df["GRUPO"].isin(grupos_sel)] if grupos_sel else df
        todos_en_pool = sorted(base_ag["AGENTE"].dropna().unique().tolist())
        inactivos_en_pool = [a for a in todos_en_pool if a in ops.get("agentes_inactivos", [])]
        activos_en_pool   = [a for a in todos_en_pool if a not in inactivos_en_pool]

        # Lista de agentes disponibles según toggle
        if ocultar_inactivos:
            pool_agentes = activos_en_pool
            if inactivos_en_pool:
                st.caption(f"ℹ️ {len(inactivos_en_pool)} agentes inactivos ocultos")
        else:
            pool_agentes = todos_en_pool
            if inactivos_en_pool:
                st.caption(f"⚠️ {len(inactivos_en_pool)} agentes inactivos visibles")

        # Sanear selección previa si cambió el pool
        for key in ("f_agentes_inc", "f_agentes_exc"):
            if key in st.session_state:
                st.session_state[key] = [
                    a for a in st.session_state[key] if a in pool_agentes
                ]

        # Incluir / Excluir en expander para no saturar el sidebar
        with st.expander("🎯 Seleccionar agentes", expanded=False):
            tab_inc, tab_exc = st.tabs(["✅ Incluir", "🚫 Excluir"])

            with tab_inc:
                agentes_sel = st.multiselect(
                    "Incluir solo estos agentes",
                    pool_agentes,
                    key="f_agentes_inc",
                    placeholder="Todos los agentes activos",
                    help="Vacío = incluye todos. Si seleccionas algunos, "
                         "solo esos aparecen en el análisis.",
                )
                if agentes_sel:
                    st.caption(f"✅ {len(agentes_sel)} agente(s) incluidos")

            with tab_exc:
                agentes_excluidos = st.multiselect(
                    "Excluir estos agentes",
                    pool_agentes,
                    key="f_agentes_exc",
                    placeholder="Ninguno excluido",
                    help="Estos agentes se eliminan del análisis aunque "
                         "estén en el rango de fechas.",
                )
                if agentes_excluidos:
                    st.caption(f"🚫 {len(agentes_excluidos)} agente(s) excluidos")

    # ── Prioridad / Origen ───────────────────────────────────────────────────
    prioridades_sel = st.multiselect(
        "Prioridad", ops["prioridades"], key="f_prioridades",
        placeholder="Todas",
    )
    origenes_sel = st.multiselect(
        "Origen", ops["origenes"], key="f_origenes",
        placeholder="Todos",
    )

    # ── Rango de fechas ──────────────────────────────────────────────────────
    # Solución robusta: un ÚNICO date_input de rango nativo (tuple de 2 dates).
    # Los botones rápidos usan selectbox de preset → no tocan ninguna key
    # de widget activo, eliminando completamente el StreamlitAPIException.

    st.markdown("**Rango de creación**")

    # Preset rápido: selectbox simple, sin conflicto de session_state
    _hoy   = datetime.date.today()
    _presets = {
        "Rango personalizado": None,
        "Últimos 7 días":      ((_hoy - datetime.timedelta(days=7)),   _hoy),
        "Últimos 30 días":     ((_hoy - datetime.timedelta(days=30)),  _hoy),
        "Últimos 90 días":     ((_hoy - datetime.timedelta(days=90)),  _hoy),
        "Todo el histórico":   (ops["fecha_min"], ops["fecha_max"]),
    }
    _preset_sel = st.selectbox(
        "Acceso rápido", list(_presets.keys()),
        key="f_fecha_preset", label_visibility="collapsed",
    )

    # Determinar valor inicial del rango
    if _presets[_preset_sel] is not None:
        _rango_default = _presets[_preset_sel]
    else:
        _rango_default = (ops["fecha_min"], ops["fecha_max"])

    # UN solo date_input de rango — tipo nativo de Streamlit, sin keys auxiliares
    _rango = st.date_input(
        "Rango de fechas",
        value=_rango_default,
        min_value=ops["fecha_min"],
        max_value=ops["fecha_max"],
        key="f_fechas",
        label_visibility="collapsed",
        format="DD/MM/YYYY",
    )

    # Desempaquetar con seguridad (el usuario puede seleccionar solo 1 fecha)
    if isinstance(_rango, (list, tuple)) and len(_rango) == 2:
        fecha_ini, fecha_fin = _rango[0], _rango[1]
    elif isinstance(_rango, (list, tuple)) and len(_rango) == 1:
        fecha_ini = fecha_fin = _rango[0]
    else:
        fecha_ini = fecha_fin = _rango  # date suelto

    # ── Días de resolución ───────────────────────────────────────────────────
    dias_sel = st.slider(
        "Días de resolución", 0.0, ops["dias_max"],
        (0.0, ops["dias_max"]), key="f_dias",
    )

    # ── Búsqueda libre ───────────────────────────────────────────────────────
    busqueda = st.text_input(
        "🔍 Buscar en asunto/descripción",
        key="f_busqueda", placeholder="ej: vpn, contraseña...",
    )


# ═══════════════════════════════════════════════════
# APLICAR FILTROS (una sola máscara booleana)
# ═══════════════════════════════════════════════════

def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    # Grupo
    if grupos_sel:
        mask &= df["GRUPO"].isin(grupos_sel)

    # Agentes: inclusión y exclusión combinadas
    if tiene_agente:
        if agentes_sel:
            mask &= df["AGENTE"].isin(agentes_sel)
        elif ocultar_inactivos and inactivos_en_pool:
            # Si no hay selección explícita pero toggle activo → excluir inactivos
            mask &= ~df["AGENTE"].isin(inactivos_en_pool)
        if agentes_excluidos:
            mask &= ~df["AGENTE"].isin(agentes_excluidos)

    # Prioridad / Origen
    if prioridades_sel:
        mask &= df["PRIORIDAD"].isin(prioridades_sel)
    if origenes_sel:
        mask &= df["ORIGEN"].isin(origenes_sel)

    # Fechas (dos inputs separados)
    mask &= df["CREACION"].dt.date.between(fecha_ini, fecha_fin)

    # Días resolución
    if dias_sel[0] > 0 or dias_sel[1] < ops["dias_max"]:
        mask &= df["DIAS"].between(dias_sel[0], dias_sel[1])

    # Búsqueda libre
    q = busqueda.strip()
    if q:
        mask &= df["TEXTO_COMPLETO"].str.contains(q, case=False, na=False, regex=False)

    return df[mask]


df_filtrado = aplicar_filtros(df)

# Contador y feedback en sidebar
_n_ag_exc = len(agentes_excluidos) if tiene_agente else 0
_n_in_ocu = len(inactivos_en_pool) if (tiene_agente and ocultar_inactivos) else 0
st.sidebar.markdown("---")
st.sidebar.caption(
    f"📊 **{len(df_filtrado):,}** de {len(df):,} tickets  \n"
    + (f"👤 {_n_in_ocu} inactivos ocultos · {_n_ag_exc} excluidos" if tiene_agente else "")
)

if df_filtrado.empty:
    st.warning("Ningún ticket cumple los filtros. Ajusta o limpia los filtros.")
    st.stop()


# ═══════════════════════════════════════════════════
# SIDEBAR — ALERTAS OPERACIONALES
# ═══════════════════════════════════════════════════

st.sidebar.markdown("## Estado Operacional")
n_criticos = int((df_filtrado["DIAS"] > SLA_CRITICO_DIAS).sum())
n_riesgo   = int((df_filtrado["DIAS"] > SLA_RIESGO_DIAS).sum())

if n_criticos > 0:
    st.sidebar.error(f"🔴 {n_criticos} tickets fuera SLA")
elif n_riesgo > 0:
    st.sidebar.warning(f"🟡 {n_riesgo} tickets en riesgo")
else:
    st.sidebar.success("🟢 Operación estable")


# ═══════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Resumen", "Operación", "Riesgo", "Modelo",
    "Comparación", "Agentes", "Experiencia Usuario",
])


# ══════════════════════════════════════════
# TAB 1 — RESUMEN
# ══════════════════════════════════════════

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets",          f"{len(df_filtrado):,}")
    col2.metric("Promedio días resolución", round(df_filtrado["DIAS"].mean(), 2))
    col3.metric("% Riesgo >5 días",        round(df_filtrado["RIESGO_OPERATIVO"].mean() * 100, 2))
    col4.metric("% Demora crítica",         round(df_filtrado["DEMORA_CRITICA"].mean() * 100, 2))
    st.divider()

    hoy = pd.Timestamp.today().normalize()
    n_hoy       = int((df_filtrado["CREACION"].dt.normalize() == hoy).sum())
    mask_backlog = df_filtrado["TICKET_ESTADO"].isin(["Sin revisar", "En Proceso", "Escalado"])
    backlog      = df_filtrado[mask_backlog]
    n_resueltos  = int((df_filtrado["TICKET_ESTADO"] == "Resuelto").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickets creados hoy",       n_hoy)
    col2.metric("Backlog actual",            len(backlog))
    col3.metric("Mediana resolución (días)", round(df_filtrado["DIAS"].median(), 2))
    col4.metric("Ticket abierto más antiguo",
                round(backlog["DIAS"].max(), 2) if len(backlog) else 0)
    st.metric("Tasa de resolución",
              f"{round(n_resueltos / max(len(df_filtrado), 1) * 100, 2)}%")

    if len(backlog) > 200:
        st.error("🔴 Operación saturada")
    elif len(backlog) > 100:
        st.warning("🟡 Operación en riesgo")
    else:
        st.success("🟢 Operación estable")
    st.divider()

    colA, colB = st.columns(2)
    with colA:
        fig = px.histogram(df_filtrado[df_filtrado["DIAS"] <= 30],
                           x="DIAS", nbins=30, title="Distribución de tiempo de resolución")
        fig.add_vline(x=3, line_dash="dash", line_color="green")
        fig.add_vline(x=5, line_dash="dash", line_color="orange")
        fig.add_vline(x=7, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.plotly_chart(
            px.pie(df_filtrado, names="ESTADO_SLA", color="ESTADO_SLA",
                   color_discrete_map=SLA_COLORS, title="Distribución SLA"),
            use_container_width=True,
        )
    st.divider()

    tg = (df_filtrado.groupby("GRUPO", observed=True).size()
          .reset_index(name="Tickets").sort_values("Tickets", ascending=False))
    st.plotly_chart(
        px.bar(tg, x="GRUPO", y="Tickets", title="Volumen de tickets por grupo"),
        use_container_width=True,
    )
    ts = (df_filtrado
          .assign(SEMANA=df_filtrado["CREACION"].dt.to_period("W").astype(str))
          .groupby("SEMANA").size().reset_index(name="Tickets"))
    st.plotly_chart(
        px.line(ts, x="SEMANA", y="Tickets", markers=True, title="Evolución semanal"),
        use_container_width=True,
    )


# ══════════════════════════════════════════
# TAB 2 — OPERACIÓN
# ══════════════════════════════════════════

with tab2:
    tm = (df_filtrado
          .assign(MES=df_filtrado["CREACION"].dt.to_period("M").astype(str))
          .groupby("MES").size().reset_index(name="Tickets"))
    st.plotly_chart(
        px.line(tm, x="MES", y="Tickets", markers=True, title="Evolución mensual"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.pie(df_filtrado, names="PRIORIDAD", title="Distribución por prioridad"),
        use_container_width=True,
    )
    to = df_filtrado.groupby("ORIGEN", observed=True).size().reset_index(name="Tickets")
    st.plotly_chart(
        px.bar(to, x="ORIGEN", y="Tickets", title="Tickets por origen"),
        use_container_width=True,
    )
    tabla_heat = pd.crosstab(df_filtrado["GRUPO"], df_filtrado["PRIORIDAD"])
    st.plotly_chart(
        px.imshow(tabla_heat, text_auto=True, title="Grupo vs Prioridad"),
        use_container_width=True,
    )
    ti = (df_filtrado.groupby("TIPO_INCIDENTE", observed=True)
          .size().reset_index(name="Tickets"))
    st.plotly_chart(
        px.bar(ti, x="TIPO_INCIDENTE", y="Tickets", title="Tipos de incidente"),
        use_container_width=True,
    )


# ══════════════════════════════════════════
# TAB 3 — RIESGO
# ══════════════════════════════════════════

with tab3:
    rg = (df_filtrado.groupby("GRUPO", observed=True)["RIESGO_OPERATIVO"]
          .mean().reset_index())
    st.plotly_chart(
        px.bar(rg, x="GRUPO", y="RIESGO_OPERATIVO", title="Tasa de riesgo por grupo"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.box(df_filtrado, x="GRUPO", y="DIAS", title="Distribución de días por grupo"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.bar(palabras_recurrentes(df_filtrado, 20),
               x="Frecuencia", y="Palabra", orientation="h",
               title="Palabras más frecuentes"),
        use_container_width=True,
    )
    if "PROB_RIESGO" in df_filtrado.columns and df_filtrado["PROB_RIESGO"].notna().any():
        st.plotly_chart(
            px.histogram(df_filtrado, x="PROB_RIESGO", nbins=30,
                         title="Distribución de probabilidad de riesgo"),
            use_container_width=True,
        )
    else:
        st.info("PROB_RIESGO no disponible — ejecuta el ETL con los artefactos del modelo.")

    criticos = df_filtrado[df_filtrado["DIAS"] > SLA_CRITICO_DIAS]
    cols_c = [c for c in ["TICKET_ID","TICKET_ASUNTO","GRUPO","PRIORIDAD","DIAS"]
              if c in criticos.columns]
    st.subheader("Tickets críticos")
    st.dataframe(criticos[cols_c], use_container_width=True)

    if "ANOMALIA" in df_filtrado.columns:
        st.plotly_chart(
            px.scatter(df_filtrado, x="DIAS", y="PRIORIDAD", color="ANOMALIA",
                       title="Anomalías en tiempos de resolución"),
            use_container_width=True,
        )


# ══════════════════════════════════════════
# TAB 4 — MODELO (mejorado)
# ══════════════════════════════════════════

with tab4:
    st.header("Modelo predictivo de riesgo SLA")

    # ── Info del modelo activo ────────────────────────────────────────────────
    with st.expander("ℹ️ ¿Qué modelo se usa y cómo funciona?", expanded=False):
        st.markdown("""
**Modelo:** Gradient Boosting (HistGradientBoostingClassifier de scikit-learn)

**Por qué es mejor que la Regresión Logística original:**

| Característica | Regresión Logística | Gradient Boosting |
|---|---|---|
| Relaciones no lineales | ❌ No captura | ✅ Captura automáticamente |
| Interacciones entre variables | ❌ Manual | ✅ Aprende solas |
| Valores nulos en features | ❌ Requiere imputación | ✅ Manejo nativo |
| Rendimiento típico en HelpDesk | AUC ~0.72 | AUC ~0.85–0.92 |
| Overfitting | Bajo | Controlado con early stopping |

**Features del modelo (enriquecidas):**
- Texto: TF-IDF del asunto + descripción (n-gramas 1–2)
- Categóricas: PRIORIDAD, GRUPO, ORIGEN (encoded)
- Numéricas nuevas: hora del día, día semana, mes, es_fin_de_semana
- NLP: score de sentimiento, flag de urgencia, flag de conflicto
- Historial del agente: promedio de días de resolución

**Entrenamiento:** Ejecutar `python train_model.py` genera los nuevos `.pkl`
        """)

    # ── Formulario de predicción (fragment: slider no re-ejecuta la app) ──────
    st.subheader("Predicción de riesgo para nuevo ticket")

    @st.fragment
    def formulario_prediccion():
        c1, c2 = st.columns(2)
        with c1:
            asunto      = st.text_input("Asunto del ticket", key="pred_asunto")
            prioridad   = st.selectbox("Prioridad", ops["prioridades"], key="pred_prio")
            grupo       = st.selectbox("Grupo",     ops["grupos"],      key="pred_grupo")
        with c2:
            descripcion = st.text_area("Descripción", key="pred_desc", height=120)
            origen      = st.selectbox("Origen",    ops["origenes"],    key="pred_origen")
            hora_creacion = st.slider("Hora de creación (0–23)", 0, 23, 9, key="pred_hora")

        if st.button("🔮 Predecir riesgo", key="btn_predecir", type="primary"):
            if not asunto.strip():
                st.warning("Ingresa al menos el asunto del ticket.")
                return
            try:
                modelo, vectorizer, encoder = cargar_modelo()
                proba, nivel, explicacion = pipeline.predecir_ticket_v2(
                    modelo, vectorizer, encoder,
                    asunto, descripcion, prioridad, grupo, origen, hora_creacion,
                )
                # Gauge visual del nivel de riesgo
                color = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}[nivel]
                st.metric(
                    f"{color} Nivel de riesgo: {nivel}",
                    f"{round(proba * 100, 1)}% probabilidad",
                    help="Probabilidad de que este ticket supere los 5 días de resolución",
                )
                # Barra de progreso visual
                st.progress(proba, text=f"Probabilidad: {round(proba*100,1)}%")

                if explicacion:
                    with st.expander("🔍 Factores que más influyeron"):
                        feat_df = pd.DataFrame(
                            explicacion.items(), columns=["Factor", "Importancia"]
                        ).sort_values("Importancia", ascending=False).head(10)
                        st.plotly_chart(
                            px.bar(feat_df, x="Importancia", y="Factor",
                                   orientation="h", title="Top 10 features más importantes"),
                            use_container_width=True,
                        )
            except FileNotFoundError:
                st.error("No se encontraron los artefactos del modelo (.pkl). "
                         "Ejecuta `python train_model.py` primero.")
            except Exception as e:
                st.error(f"Error en la predicción: {e}")

    formulario_prediccion()

    # ── Métricas del modelo sobre datos reales ────────────────────────────────
    st.divider()
    st.subheader("Desempeño del modelo sobre datos históricos")

    resultado = _metricas_modelo(df)
    if resultado is None:
        st.info("PROB_RIESGO no disponible. Ejecuta el ETL con el modelo entrenado.")
    else:
        fpr, tpr, roc_auc, cm, reporte = resultado

        # KPIs resumen
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC-ROC", round(roc_auc, 3),
                  help="1.0 = perfecto, 0.5 = aleatorio")
        m2.metric("Precisión (clase riesgo)",
                  round(reporte.get("1", {}).get("precision", 0), 3))
        m3.metric("Recall (clase riesgo)",
                  round(reporte.get("1", {}).get("recall", 0), 3))
        m4.metric("F1 (clase riesgo)",
                  round(reporte.get("1", {}).get("f1-score", 0), 3))

        colROC, colCM = st.columns(2)
        with colROC:
            fig_roc = px.line(
                x=fpr, y=tpr,
                title=f"Curva ROC — AUC = {roc_auc:.3f}",
                labels={"x": "Tasa de Falsos Positivos", "y": "Tasa de Verdaderos Positivos"},
            )
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash="dash", color="gray"))
            fig_roc.add_annotation(x=0.7, y=0.3, text=f"AUC = {roc_auc:.3f}",
                                   showarrow=False, font=dict(size=14))
            st.plotly_chart(fig_roc, use_container_width=True)

        with colCM:
            _labels = ["Sin riesgo", "En riesgo"]
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=_labels, y=_labels,
                color_continuous_scale="Blues",
                title="Matriz de confusión (umbral 0.5)",
                labels=dict(x="Predicho", y="Real"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Umbral ajustable
        @st.fragment
        def analisis_umbral():
            st.subheader("Análisis de umbral de decisión")
            umbral_adj = st.slider(
                "Ajustar umbral de clasificación",
                0.10, 0.90, 0.50, 0.05,
                key="umbral_modelo",
                help="Un umbral menor detecta más riesgos (más recall) "
                     "pero genera más falsos positivos.",
            )
            df_eval = df.dropna(subset=["PROB_RIESGO"])
            y_true  = df_eval["RIESGO_OPERATIVO"]
            y_pred  = (df_eval["PROB_RIESGO"] > umbral_adj).astype(int)
            cm_adj  = confusion_matrix(y_true, y_pred)
            rep_adj = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

            a1, a2, a3 = st.columns(3)
            a1.metric("Precisión",
                      round(rep_adj.get("1", {}).get("precision", 0), 3))
            a2.metric("Recall",
                      round(rep_adj.get("1", {}).get("recall", 0), 3))
            a3.metric("F1",
                      round(rep_adj.get("1", {}).get("f1-score", 0), 3))

            fig_cm2 = px.imshow(
                cm_adj, text_auto=True,
                x=["Sin riesgo", "En riesgo"],
                y=["Sin riesgo", "En riesgo"],
                color_continuous_scale="Oranges",
                title=f"Confusión con umbral = {umbral_adj}",
                labels=dict(x="Predicho", y="Real"),
            )
            st.plotly_chart(fig_cm2, use_container_width=True)

        if df["PROB_RIESGO"].notna().any():
            analisis_umbral()

        st.caption(
            "⚠️ Si el modelo fue entrenado con esta misma base, las métricas "
            "están infladas (sin hold-out). Usa `train_model.py` con split 80/20."
        )


# ══════════════════════════════════════════
# TAB 5 — COMPARACIÓN
# ══════════════════════════════════════════

with tab5:
    st.subheader("Comparación entre bases de tickets")
    colb1, colb2 = st.columns(2)
    base1 = colb1.selectbox("Base 1", list(URLS_BASES.keys()), key="comp_base1")
    base2 = colb2.selectbox("Base 2", list(URLS_BASES.keys()), index=1, key="comp_base2")

    if base1 == base2:
        st.warning("Selecciona dos bases diferentes.")
    elif st.toggle("Cargar comparación", key="comp_activa"):
        df1 = cargar_datos(base1)
        df2 = cargar_datos(base2)
        comp = pd.DataFrame({
            "Base":             [base1, base2],
            "Total Tickets":    [len(df1), len(df2)],
            "Promedio días":    [round(df1["DIAS"].mean(), 2), round(df2["DIAS"].mean(), 2)],
            "% Riesgo":         [round(df1["RIESGO_OPERATIVO"].mean()*100, 2),
                                 round(df2["RIESGO_OPERATIVO"].mean()*100, 2)],
            "% Demora crítica": [round(df1["DEMORA_CRITICA"].mean()*100, 2),
                                 round(df2["DEMORA_CRITICA"].mean()*100, 2)],
        })
        st.dataframe(comp, use_container_width=True)
        st.plotly_chart(
            px.bar(comp, x="Base", y=["% Riesgo","% Demora crítica"],
                   barmode="group", title="Comparación de riesgo"),
            use_container_width=True,
        )
        sla_c = pd.concat([
            df1.groupby("ESTADO_SLA", observed=True).size()
               .reset_index(name="Tickets").assign(Base=base1),
            df2.groupby("ESTADO_SLA", observed=True).size()
               .reset_index(name="Tickets").assign(Base=base2),
        ])
        st.plotly_chart(
            px.bar(sla_c, x="Base", y="Tickets", color="ESTADO_SLA",
                   barmode="stack", color_discrete_map=SLA_COLORS,
                   title="Comparación de SLA"),
            use_container_width=True,
        )


# ══════════════════════════════════════════
# TAB 6 — GESTIÓN AGENTES
# ══════════════════════════════════════════

with tab6:
    st.header("Gestión operativa de agentes")

    if not tiene_agente:
        st.warning("La base no contiene columna AGENTE.")
    else:
        # Mostrar quién está siendo excluido del análisis
        if agentes_excluidos or (ocultar_inactivos and inactivos_en_pool):
            with st.expander("👁️ Agentes fuera del análisis actual"):
                if ocultar_inactivos and inactivos_en_pool:
                    st.info(f"**Inactivos ocultos** (sin ticket >{UMBRAL_INACTIVO_DIAS}d): "
                            f"{', '.join(inactivos_en_pool)}")
                if agentes_excluidos:
                    st.warning(f"**Excluidos manualmente**: {', '.join(agentes_excluidos)}")

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
            st.info("No hay datos para los filtros actuales.")
        else:
            st.divider()

            carga = (df_ag.groupby("AGENTE", observed=True).size()
                     .reset_index(name="Tickets").sort_values("Tickets", ascending=False))
            st.subheader("Carga de trabajo por agente")
            st.plotly_chart(
                px.bar(carga, x="AGENTE", y="Tickets", title="Tickets por agente"),
                use_container_width=True, key="carga_agentes",
            )

            sla = (df_ag.groupby("AGENTE", observed=True)["DIAS"]
                   .apply(lambda x: (x <= SLA_RIESGO_DIAS).mean() * 100)
                   .reset_index(name="SLA_%"))
            st.subheader("Cumplimiento SLA por agente")
            st.plotly_chart(
                px.bar(sla, x="AGENTE", y="SLA_%", title="% SLA cumplido"),
                use_container_width=True, key="sla_agentes",
            )

            ranking = (
                df_ag.groupby("AGENTE", observed=True)
                .agg(Tickets=("TICKET_ID","count"),
                     Promedio_dias=("DIAS","mean"),
                     SLA=("DIAS", lambda x: (x<=SLA_RIESGO_DIAS).mean()*100))
                .reset_index().sort_values("SLA", ascending=False)
            )
            st.subheader("Ranking de desempeño")
            st.dataframe(ranking, use_container_width=True)

            prod = (df_ag.groupby(["MES","AGENTE"], observed=True)
                    .size().reset_index(name="Tickets"))
            st.subheader("Productividad mensual")
            st.plotly_chart(
                px.line(prod, x="MES", y="Tickets", color="AGENTE", markers=True),
                use_container_width=True, key="productividad_agentes",
            )

            limite = carga["Tickets"].mean() * 1.5
            carga["Estado"] = np.where(carga["Tickets"] > limite, "Sobrecarga", "Normal")
            st.subheader("Detección de agentes saturados")
            st.plotly_chart(
                px.bar(carga, x="AGENTE", y="Tickets", color="Estado"),
                use_container_width=True, key="saturacion_agentes",
            )

            ranking_gr = (
                df_ag.groupby(["GRUPO","AGENTE"], observed=True)
                .agg(Tickets=("TICKET_ID","count"), Promedio_dias=("DIAS","mean"))
                .reset_index()
            )
            st.subheader("Tickets por agente y grupo")
            st.plotly_chart(
                px.bar(ranking_gr, x="AGENTE", y="Tickets", color="GRUPO",
                       title="Tickets por agente y grupo"),
                use_container_width=True, key="tickets_agente_grupo",
            )

            sla_grupo = (df_ag.groupby("GRUPO", observed=True)["DIAS"]
                         .apply(lambda x: (x>SLA_RIESGO_DIAS).mean()*100)
                         .reset_index(name="Incumplimiento_%"))
            st.subheader("Incumplimiento SLA por grupo")
            st.plotly_chart(
                px.bar(sla_grupo, x="GRUPO", y="Incumplimiento_%"),
                use_container_width=True, key="sla_grupo",
            )

            # — Tickets abiertos —
            abiertos = df_ag[
                df_ag["TICKET_ESTADO"].isin(["Sin revisar","En Proceso","Escalado"])
            ].copy()
            abiertos["ESTADO_OPERATIVO"] = np.select(
                [abiertos["TICKET_ESTADO"]=="Sin revisar",
                 abiertos["TICKET_ESTADO"]=="En Proceso"],
                ["🔴 Sin revisar","🟠 En proceso"], default="🟡 Escalado",
            )
            st.subheader("Tickets no resueltos")
            if abiertos.empty:
                st.success("No hay tickets pendientes.")
            else:
                tabla_ab = (abiertos.groupby(["AGENTE","GRUPO"], observed=True)
                            .size().reset_index(name="Tickets abiertos"))
                st.dataframe(tabla_ab, use_container_width=True)
                st.plotly_chart(
                    px.bar(tabla_ab, x="AGENTE", y="Tickets abiertos", color="GRUPO"),
                    use_container_width=True, key="tickets_abiertos",
                )

                tot = len(abiertos)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Abiertos", tot)
                c2.metric("Promedio días", round(abiertos["DIAS"].mean(),2))
                c3.metric("% riesgo SLA",
                          round((abiertos["DIAS"]>SLA_RIESGO_DIAS).sum()/tot*100,2))
                c4.metric("% críticos",
                          round((abiertos["DIAS"]>SLA_CRITICO_DIAS).sum()/tot*100,2))

            # — Expertise —
            if "TIPO_INCIDENTE" in df_ag.columns:
                st.subheader("Expertise por agente")
                matriz = pd.crosstab(df_ag["AGENTE"], df_ag["TIPO_INCIDENTE"])
                if matriz.shape[0] > 0:
                    st.plotly_chart(
                        px.imshow(matriz, text_auto=True, aspect="auto"),
                        use_container_width=True, key="heatmap_expertise",
                    )

            # — Recomendación —
            if "TIPO_INCIDENTE" in df_ag.columns:
                st.subheader("Recomendación automática de agente")

                @st.fragment
                def recomendacion_agente(df_ag):
                    c1, c2 = st.columns(2)
                    tipo = c1.selectbox("Tipo de incidente",
                                        sorted(df_ag["TIPO_INCIDENTE"].dropna().unique()),
                                        key="tipo_rec")
                    grp  = c2.selectbox("Grupo",
                                        sorted(df_ag["GRUPO"].dropna().unique()),
                                        key="grp_rec")
                    hist = df_ag[(df_ag["TIPO_INCIDENTE"]==tipo) & (df_ag["GRUPO"]==grp)]
                    if hist.empty:
                        st.warning("Sin histórico para esta combinación.")
                        return
                    rank = (hist.groupby("AGENTE", observed=True)
                            .agg(Tickets=("TICKET_ID","count"),
                                 Promedio_dias=("DIAS","mean"))
                            .reset_index()
                            .sort_values(["Promedio_dias","Tickets"],
                                         ascending=[True,False]))
                    st.success(f"Agente recomendado: **{rank.iloc[0]['AGENTE']}**")
                    st.info(f"Promedio: {round(rank.iloc[0]['Promedio_dias'],2)} días")
                    st.dataframe(rank, use_container_width=True)

                recomendacion_agente(df_ag)

            # — Alerta temprana SLA —
            st.subheader("Alerta temprana de incumplimiento SLA")

            @st.fragment
            def alerta_temprana(df_ag):
                if "PROB_RIESGO" not in df_ag.columns:
                    st.info("PROB_RIESGO no disponible.")
                    return
                df_r = df_ag.dropna(subset=["PROB_RIESGO"])
                if df_r.empty:
                    return
                umbral = st.slider("Umbral de alerta", 0.50, 0.95, 0.80, 0.05,
                                   key="umbral_sla")
                alto = df_r[df_r["PROB_RIESGO"] >= umbral]
                c1,c2,c3 = st.columns(3)
                c1.metric("Analizados",  len(df_r))
                c2.metric("Alto riesgo", len(alto))
                c3.metric("% riesgo",    round(len(alto)/max(len(df_r),1)*100,2))
                fig_d = px.histogram(df_r, x="PROB_RIESGO", nbins=30,
                                     title="Distribución PROB_RIESGO")
                fig_d.add_vline(x=umbral, line_dash="dash", line_color="red")
                st.plotly_chart(fig_d, use_container_width=True, key="dist_riesgo")
                if not alto.empty:
                    cols = [c for c in ["TICKET_ID","TICKET_ASUNTO","AGENTE",
                                        "GRUPO","PRIORIDAD","DIAS","PROB_RIESGO"]
                            if c in alto.columns]
                    st.dataframe(alto[cols].sort_values("PROB_RIESGO", ascending=False),
                                 use_container_width=True)

            alerta_temprana(df_ag)


# ══════════════════════════════════════════
# TAB 7 — EXPERIENCIA USUARIO
# ══════════════════════════════════════════

with tab7:
    st.header("Experiencia del usuario y sentimiento")

    try:
        import pysentimiento  # noqa
        st.success("Modelo de sentimiento activo: 🤖 **BETO** (transformer entrenado en español)")
    except ImportError:
        st.info("Modelo activo: 📖 **Léxico español** (instala `pysentimiento` para BETO)")

    if "SCORE_SENTIMIENTO" in df_filtrado.columns:
        _pct_neg = round((df_filtrado["SENTIMIENTO"]=="Negativo").mean()*100, 1)
        _pct_pos = round((df_filtrado["SENTIMIENTO"]=="Positivo").mean()*100, 1)
        _score_m = round(df_filtrado["SCORE_SENTIMIENTO"].mean(), 3)
        k1,k2,k3 = st.columns(3)
        k1.metric("% Negativos",  f"{_pct_neg}%")
        k2.metric("% Positivos",  f"{_pct_pos}%")
        k3.metric("Score medio", _score_m,
                  help="-1 muy negativo → +1 muy positivo")

    st.subheader("Distribución de sentimiento")
    st.plotly_chart(
        px.pie(df_filtrado, names="SENTIMIENTO", color="SENTIMIENTO",
               color_discrete_map=SENTIMENT_COLORS),
        use_container_width=True,
    )

    if "SCORE_SENTIMIENTO" in df_filtrado.columns:
        st.plotly_chart(
            px.histogram(df_filtrado, x="SCORE_SENTIMIENTO", nbins=40,
                         color="SENTIMIENTO", color_discrete_map=SENTIMENT_COLORS,
                         title="Score continuo de sentimiento"),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Urgencia detectada")
    st.plotly_chart(
        px.pie(df_filtrado, names="URGENCIA"), use_container_width=True,
    )

    sent_g = (df_filtrado.groupby(["GRUPO","SENTIMIENTO"], observed=True)
              .size().reset_index(name="Tickets"))
    st.subheader("Sentimiento por grupo")
    st.plotly_chart(
        px.bar(sent_g, x="GRUPO", y="Tickets", color="SENTIMIENTO",
               barmode="stack", color_discrete_map=SENTIMENT_COLORS),
        use_container_width=True,
    )

    if tiene_agente:
        sent_a = (df_filtrado.groupby(["AGENTE","SENTIMIENTO"], observed=True)
                  .size().reset_index(name="Tickets"))
        st.subheader("Sentimiento por agente")
        st.plotly_chart(
            px.bar(sent_a, x="AGENTE", y="Tickets", color="SENTIMIENTO",
                   barmode="stack", color_discrete_map=SENTIMENT_COLORS),
            use_container_width=True,
        )

    negativos = df_filtrado[df_filtrado["SENTIMIENTO"]=="Negativo"]
    st.subheader("Tickets con sentimiento negativo")
    if negativos.empty:
        st.success("No hay tickets negativos con los filtros actuales.")
    else:
        cols = [c for c in ["TICKET_ID","TICKET_ASUNTO","GRUPO","AGENTE",
                            "PRIORIDAD","DIAS","SCORE_SENTIMIENTO"]
                if c in negativos.columns]
        sort_col = "SCORE_SENTIMIENTO" if "SCORE_SENTIMIENTO" in negativos.columns else "DIAS"
        st.dataframe(negativos[cols].sort_values(sort_col), use_container_width=True)

    urgentes = df_filtrado[df_filtrado["URGENCIA"]=="🔥 Alta urgencia"]
    st.subheader("Tickets urgentes")
    if urgentes.empty:
        st.success("No hay tickets urgentes.")
    else:
        cols = [c for c in ["TICKET_ID","TICKET_ASUNTO","GRUPO","AGENTE","PRIORIDAD","DIAS"]
                if c in urgentes.columns]
        st.dataframe(urgentes[cols], use_container_width=True)

    @st.fragment
    def incidentes_recurrentes(df_filtrado):
        st.subheader("Incidentes recurrentes")
        rec = palabras_recurrentes(df_filtrado, 15)
        if rec.empty:
            st.info("Sin texto suficiente.")
            return
        st.plotly_chart(
            px.bar(rec, x="Frecuencia", y="Palabra", orientation="h"),
            use_container_width=True,
        )
        pal = st.selectbox("Palabra clave", rec["Palabra"], key="pal_rec")
        rel = df_filtrado[
            df_filtrado["TEXTO_COMPLETO"].str.contains(pal, case=False, na=False, regex=False)
        ]
        cols = [c for c in ["TICKET_ID","TICKET_ASUNTO","GRUPO","AGENTE","PRIORIDAD"]
                if c in rel.columns]
        st.dataframe(rel[cols], use_container_width=True)

    incidentes_recurrentes(df_filtrado)
