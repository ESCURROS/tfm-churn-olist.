"""
TFM — Optimización de la retención de clientes (Churn)
Aplicación Streamlit — Máster en Data Science Executive, IMF
Autor: Eduardo Jose Sanz Curros
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Configuración de la página ───────────────────────────────────────────
st.set_page_config(
    page_title="Olist Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fuentes */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Fondo principal */
    .main { background-color: #F7F8FA; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F1923;
        color: white;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }

    /* Tarjetas métricas */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .metric-card.churn { border-left-color: #DC2626; }
    .metric-card.safe  { border-left-color: #16A34A; }
    .metric-card.info  { border-left-color: #D97706; }

    .metric-label { font-size: 12px; color: #6B7280; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-value { font-size: 32px; font-weight: 600; color: #111827; line-height: 1.1; margin-top: 4px; }
    .metric-sub   { font-size: 13px; color: #9CA3AF; margin-top: 4px; }

    /* Badge de predicción */
    .prediction-badge {
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
        display: inline-block;
        margin: 8px 0;
    }
    .badge-churn  { background: #FEE2E2; color: #991B1B; }
    .badge-active { background: #DCFCE7; color: #166534; }

    /* Tabla comparativa */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Headers de sección */
    .section-header {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6B7280;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #E5E7EB;
    }

    /* Ocultar header de Streamlit */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# CARGA DE RECURSOS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def cargar_modelo():
    modelo  = joblib.load("modelo_churn.pkl")
    scaler  = joblib.load("scaler_churn.pkl")
    with open("modelo_metadatos.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, scaler, meta

@st.cache_data
def cargar_datos():
    return pd.read_csv("olist_sample_streamlit.csv")

try:
    modelo, scaler, meta = cargar_modelo()
    df_sample = cargar_datos()
    FEATURES = meta["features"]
    modelo_cargado = True
except Exception as e:
    st.error(f"Error cargando recursos: {e}")
    st.info("Asegúrate de que modelo_churn.pkl, scaler_churn.pkl, modelo_metadatos.json y olist_sample_streamlit.csv están en el mismo directorio que app.py")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Olist Churn Predictor")
    st.markdown("**TFM — Data Science Executive**  \nIMF Smart Education")
    st.markdown("---")

    st.markdown("### 🧭 Navegación")
    pagina = st.radio(
        "",
        ["🏠 Inicio & Metodología",
         "🔮 Predicción Individual",
         "📦 Predicción por Lotes",
         "📈 Rendimiento del Modelo"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Modelo activo")
    st.markdown(f"**{meta['modelo_nombre']}**")

    metricas = meta["metricas_test"]
    st.markdown(f"""
    | Métrica | Valor |
    |---------|-------|
    | AUC-ROC | `{metricas['AUC-ROC']}` |
    | F1-Score | `{metricas['F1-Score']}` |
    | Precisión | `{metricas['Precision']}` |
    | Recall | `{metricas['Recall']}` |
    """)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px; color:#475569; line-height:1.6">
    📅 Cutoff: {meta.get('cutoff_date', 'jun-2018')}<br>
    ⏱ Ventana predicción: {meta.get('pred_window_days', 92)} días<br>
    🗄 Dataset: Olist (Kaggle)<br>
    👤 Eduardo Jose Sanz Curros
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PÁGINA 1: INICIO & METODOLOGÍA
# ══════════════════════════════════════════════════════════════════════════

if pagina == "🏠 Inicio & Metodología":
    st.markdown("# Predicción de Abandono de Clientes")
    st.markdown("#### Plataforma de análisis de churn para el e-commerce Olist")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card churn">
            <div class="metric-label">Tasa de Churn</div>
            <div class="metric-value">99.4%</div>
            <div class="metric-sub">Clientes sin recompra (92 días)</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card safe">
            <div class="metric-label">Clientes analizados</div>
            <div class="metric-value">75,387</div>
            <div class="metric-sub">Con historial hasta jun-2018</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card info">
            <div class="metric-label">AUC-ROC</div>
            <div class="metric-value">{metricas['AUC-ROC']}</div>
            <div class="metric-sub">Logistic Regression</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Período de predicción</div>
            <div class="metric-value">92 días</div>
            <div class="metric-sub">jun-2018 → sep-2018</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("### 🔬 Metodología")
        st.markdown("""
        Este proyecto aplica **aprendizaje automático supervisado** para predecir la probabilidad
        de que un cliente del marketplace brasileño Olist no realice ninguna compra en los próximos 92 días.

        **Pipeline técnico:**

        1. **Adquisición de datos** — 9 tablas relacionales del dataset Olist (Kaggle), 96.478 pedidos entregados (2016–2018)
        2. **Feature Engineering** — Métricas RFM calculadas con **ventana temporal estricta** (datos anteriores al cutoff 01/06/2018) para evitar data leakage
        3. **Balanceo de clases** — SMOTE aplicado exclusivamente sobre el conjunto de entrenamiento
        4. **Modelado** — Comparativa de 4 algoritmos: Logistic Regression, Random Forest, XGBoost y LightGBM
        5. **Evaluación** — Métricas orientadas a churn: AUC-ROC, F1-Score, Precision, Recall + validación cruzada 5-fold
        """)

        st.info("""
        **⚠️ Nota sobre data leakage corregido**  
        La versión inicial del modelo obtenía AUC-ROC = 1.0 debido a que `recencia_dias` se calculaba
        sobre el dataset completo, siendo equivalente a la definición de churn. Este error se corrigió
        aplicando una separación temporal estricta entre el período de observación y el de predicción.
        """)

    with col_b:
        st.markdown("### 📐 Variables del modelo")
        variables_info = {
            "recencia_dias": "Días desde la última compra hasta el cutoff",
            "frecuencia": "Número total de pedidos realizados",
            "valor_total": "Gasto acumulado total (BRL)",
            "avg_review_score": "Puntuación media de reseñas (1-5)",
            "num_categorias": "Categorías distintas compradas",
            "avg_delivery_days": "Días medios hasta la entrega"
        }
        for var, desc in variables_info.items():
            st.markdown(f"""
            <div style="background:white; padding:10px 14px; border-radius:8px;
                        margin-bottom:8px; border:1px solid #E5E7EB;">
                <div style="font-family:monospace; font-size:12px; color:#2563EB; font-weight:500">{var}</div>
                <div style="font-size:12px; color:#6B7280; margin-top:2px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PÁGINA 2: PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════

elif pagina == "🔮 Predicción Individual":
    st.markdown("# Predicción Individual de Churn")
    st.markdown("Introduce los datos de un cliente para obtener su probabilidad de abandono.")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("### 📋 Datos del cliente")

        recencia     = st.slider("Recencia (días desde última compra hasta cutoff)", 0, 700, 200,
                                  help="Cuanto mayor, más tiempo sin comprar antes del período de análisis")
        frecuencia   = st.slider("Frecuencia (número de pedidos)", 1, 20, 1)
        valor_total  = st.number_input("Valor total gastado (BRL)", min_value=0.0, value=150.0, step=10.0)
        review_score = st.slider("Score medio de reseñas", 1.0, 5.0, 4.0, step=0.1)
        num_cats     = st.slider("Número de categorías compradas", 1, 15, 1)
        delivery_days = st.slider("Días medios de entrega", 1, 100, 12)

        predecir = st.button("🔮 Calcular probabilidad de churn", use_container_width=True, type="primary")

    with col_result:
        st.markdown("### 📊 Resultado")

        if predecir:
            input_data = pd.DataFrame([[recencia, frecuencia, valor_total,
                                         review_score, num_cats, delivery_days]],
                                       columns=FEATURES)

            # El modelo es Logistic Regression → necesita escalado
            input_scaled = scaler.transform(input_data)
            prob_churn   = modelo.predict_proba(input_scaled)[0][1]
            prediccion   = modelo.predict(input_scaled)[0]

            # Badge de resultado
            if prediccion == 1:
                st.markdown(f"""
                <div class="prediction-badge badge-churn">
                    ⚠️ RIESGO DE ABANDONO
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-badge badge-active">
                    ✅ CLIENTE ACTIVO
                </div>""", unsafe_allow_html=True)

            # Probabilidad
            st.markdown(f"""
            <div class="metric-card {'churn' if prediccion==1 else 'safe'}">
                <div class="metric-label">Probabilidad de churn</div>
                <div class="metric-value">{prob_churn:.1%}</div>
                <div class="metric-sub">Estimación del modelo en el período de 92 días</div>
            </div>""", unsafe_allow_html=True)

            # Gauge visual con matplotlib
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh([""], [prob_churn], color="#DC2626" if prob_churn > 0.5 else "#16A34A",
                    height=0.4, alpha=0.85)
            ax.barh([""], [1], color="#E5E7EB", height=0.4, zorder=0)
            ax.axvline(0.5, color="#374151", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilidad")
            ax.set_title(f"Probabilidad de churn: {prob_churn:.1%}", fontsize=12, fontweight="bold")
            ax.spines[["top","right","left"]].set_visible(False)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Desglose de inputs
            st.markdown("**Resumen del perfil del cliente:**")
            resumen = pd.DataFrame({
                "Variable": ["Recencia", "Frecuencia", "Valor total", "Review score", "Categorías", "Entrega"],
                "Valor": [f"{recencia} días", f"{frecuencia} pedido(s)", f"R$ {valor_total:.2f}",
                           f"{review_score}/5", f"{num_cats}", f"{delivery_days} días"]
            })
            st.dataframe(resumen, hide_index=True, use_container_width=True)

        else:
            st.markdown("""
            <div style="background:#F3F4F6; border-radius:12px; padding:40px;
                        text-align:center; color:#9CA3AF;">
                <div style="font-size:40px">🔮</div>
                <div style="font-size:14px; margin-top:12px">
                    Ajusta los parámetros y pulsa<br><strong>Calcular probabilidad de churn</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PÁGINA 3: PREDICCIÓN POR LOTES
# ══════════════════════════════════════════════════════════════════════════

elif pagina == "📦 Predicción por Lotes":
    st.markdown("# Predicción por Lotes")
    st.markdown("Sube un CSV con clientes para obtener predicciones masivas.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📁 Cargar datos")
        st.markdown(f"""
        El CSV debe tener exactamente estas columnas (en cualquier orden):
        ```
        {', '.join(FEATURES)}
        ```
        """)

        uploaded = st.file_uploader("Sube tu CSV de clientes", type=["csv"])

        st.markdown("**O usa los datos de muestra del TFM:**")
        usar_muestra = st.button("📊 Cargar datos de muestra (500 clientes)", use_container_width=True)

    with col2:
        st.markdown("### 📊 Resultados")

        df_pred = None
        if usar_muestra:
            df_pred = df_sample.copy()
        elif uploaded:
            try:
                df_pred = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Error leyendo el archivo: {e}")

        if df_pred is not None:
            cols_faltantes = [c for c in FEATURES if c not in df_pred.columns]
            if cols_faltantes:
                st.error(f"Columnas faltantes: {cols_faltantes}")
            else:
                X = df_pred[FEATURES]
                X_scaled = scaler.transform(X)
                probs = modelo.predict_proba(X_scaled)[:, 1]
                preds = modelo.predict(X_scaled)

                df_resultado = df_pred[FEATURES].copy()
                df_resultado["prob_churn"]  = probs.round(4)
                df_resultado["prediccion"]  = ["Churn" if p == 1 else "Activo" for p in preds]
                df_resultado["riesgo"]      = pd.cut(probs,
                                                      bins=[0, 0.4, 0.6, 0.8, 1.0],
                                                      labels=["Bajo", "Medio", "Alto", "Crítico"])

                # KPIs
                n_churn = (preds == 1).sum()
                n_activos = (preds == 0).sum()
                tasa = n_churn / len(preds)

                k1, k2, k3 = st.columns(3)
                k1.metric("Total clientes", f"{len(preds):,}")
                k2.metric("Predicción churn", f"{n_churn:,}", f"{tasa:.1%}")
                k3.metric("Predicción activo", f"{n_activos:,}", f"{1-tasa:.1%}")

                # Tabla
                st.dataframe(
                    df_resultado.sort_values("prob_churn", ascending=False).head(20),
                    hide_index=True, use_container_width=True
                )

                # Descarga
                csv_out = df_resultado.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Descargar predicciones completas (CSV)",
                    data=csv_out,
                    file_name="predicciones_churn_olist.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# ══════════════════════════════════════════════════════════════════════════
# PÁGINA 4: RENDIMIENTO DEL MODELO
# ══════════════════════════════════════════════════════════════════════════

elif pagina == "📈 Rendimiento del Modelo":
    st.markdown("# Rendimiento del Modelo")
    st.markdown("Métricas de evaluación y comparativa de algoritmos.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Comparativa de modelos", "🎯 Métricas clave", "ℹ️ Interpretación"])

    with tab1:
        comparativa = meta.get("comparativa", [])
        if comparativa:
            df_comp = pd.DataFrame(comparativa)

            fig, ax = plt.subplots(figsize=(10, 5))
            metricas_cols = ["AUC-ROC", "F1-Score", "Precision", "Recall"]
            x = np.arange(len(metricas_cols))
            width = 0.18
            colores = ["#2563EB", "#16A34A", "#D97706", "#9333EA"]

            for i, (_, row) in enumerate(df_comp.iterrows()):
                vals = [row.get(m, 0) for m in metricas_cols]
                bars = ax.bar(x + i * width, vals, width, label=row["Modelo"],
                               color=colores[i], alpha=0.85, edgecolor="white")
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2.,
                            bar.get_height() + 0.01,
                            f"{bar.get_height():.3f}", ha="center", fontsize=8)

            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(metricas_cols, fontsize=11)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Valor de la métrica")
            ax.set_title("Comparativa de modelos — Test Set (sin data leakage)", fontsize=13, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="Baseline")
            ax.spines[["top", "right"]].set_visible(False)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("**Tabla comparativa completa:**")
            st.dataframe(df_comp.set_index("Modelo").style.highlight_max(
                axis=0, color="#DCFCE7", subset=["AUC-ROC"]),
                use_container_width=True)

    with tab2:
        metricas_display = meta["metricas_test"]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AUC-ROC</div>
                <div class="metric-value">{metricas_display['AUC-ROC']}</div>
                <div class="metric-sub">Capacidad discriminativa real</div>
            </div>
            <div class="metric-card safe">
                <div class="metric-label">Precisión</div>
                <div class="metric-value">{metricas_display['Precision']}</div>
                <div class="metric-sub">Cuando predice churn, acierta {metricas_display['Precision']:.0%}</div>
            </div>""", unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="metric-card info">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{metricas_display['F1-Score']}</div>
                <div class="metric-sub">Media armónica Precision-Recall</div>
            </div>
            <div class="metric-card churn">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{metricas_display['Recall']}</div>
                <div class="metric-sub">Churners reales detectados</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Validación cruzada 5-fold (Logistic Regression):**")
        cv_data = {
            "Métrica": ["AUC-ROC", "F1-Score", "Precisión", "Recall"],
            "Media":   [0.6064, 0.5662, 0.5649, 0.5675],
            "Desv. Std": [0.0034, 0.0036, 0.0038, 0.0043]
        }
        st.dataframe(pd.DataFrame(cv_data).set_index("Métrica"), use_container_width=True)

    with tab3:
        st.markdown("""
        ### Interpretación de los resultados

        **¿Por qué AUC-ROC de 0.60 es un resultado válido?**

        En este problema, el desbalanceo de clases es extremo: el 99.4% de los clientes
        de Olist son clasificados como churn dentro de la ventana de 92 días.
        Esto refleja una característica estructural del marketplace: la mayoría de compradores
        realizan una única transacción sin repetir.

        Un AUC-ROC de 0.60 indica que el modelo tiene **capacidad discriminativa real**
        (mejor que el azar), siendo el único de los cuatro algoritmos evaluados que logra
        distinguir genuinamente entre las dos clases, en lugar de predecir siempre churn.

        **¿Por qué XGBoost/LightGBM tienen F1 ≈ 1.0 pero AUC ≈ 0.50?**

        Estos modelos aprenden a predecir siempre la clase mayoritaria (churn),
        lo que maximiza el F1-Score en datos tan desbalanceados, pero sus probabilidades
        de predicción no están calibradas para discriminar entre clases.
        El AUC próximo a 0.50 confirma que son esencialmente equivalentes a un clasificador aleatorio.

        **Conclusión de negocio**

        El modelo permite a Olist identificar qué clientes tienen mayor probabilidad de recompra,
        priorizando campañas de retención hacia ese segmento minoritario de alto valor.
        """)

        st.info("""
        **Nota metodológica**: Las features del modelo se calculan sobre el historial de compra anterior
        al 1 de junio de 2018. La predicción indica si el cliente comprará o no en los 92 días siguientes.
        Para aplicar el modelo en producción, recalcular las métricas RFM con el historial actual
        del cliente y el cutoff correspondiente al momento de predicción.
        """)
