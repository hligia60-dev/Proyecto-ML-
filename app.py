# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import plotly.express as px
import os

# 1. CONFIGURACIÓN DE LA PÁGINA (Debe ser la primera línea de Streamlit)
st.set_page_config(
    page_title="Simulador Analítico de supervivencia del Titanic",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. DISEÑO Y COLORES PERSONALIZADOS (CSS)
st.markdown("""
    <style>
    /* Cambiar el fondo principal y el tipo de letra */
    .main { background-color: #f4f6f9; font-family: 'Helvetica Neue', Arial, sans-serif; }
    
    /* Estilo para los títulos principales */
    .main-title { color: #0A2540; font-size: 42px; font-weight: 800; text-align: center; margin-bottom: 5px; }
    .subtitle { color: #627D98; font-size: 18px; text-align: center; margin-bottom: 30px; }
    
    /* Tarjetas contenedoras para los resultados */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #0056b3;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. CARGA DE MODELOS Y DATOS HISTÓRICOS


@st.cache_resource
def cargar_modelos():
    mod_log = joblib.load('modelo_logistica.pkl')
    mod_lin = joblib.load('modelo_lineal.pkl')
    return mod_log, mod_lin


@st.cache_data
def cargar_datos_historicos():
    return pd.read_csv('titanic.csv')


try:
    modelo_logistica, modelo_lineal = cargar_modelos()
    df_historico = cargar_datos_historicos()
except Exception as e:
    st.error("⚠️ Error al cargar los archivos del proyecto. Asegúrate de haber ejecutado todo el cuaderno Jupyter previamente.")
    st.stop()

# 4. ENCABEZADO DE LA APLICACIÓN
st.markdown("<div class='main-title'>🚢 Simulador Analítico de supervivencia del Titanic</div>",
            unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predicción Supervisada mediante CRISP-ML e Interfaces Inmersivas</div>",
            unsafe_allow_html=True)

# 5. MENÚ LATERAL (SIDEBAR) - Entrada de Datos del Usuario con colores de fondo ordenados
st.sidebar.header("🎯 Parámetros del Pasajero")
st.sidebar.write(
    "Modifica los valores para calcular las predicciones en tiempo real.")

clase = st.sidebar.selectbox("Clase del Boleto (Pclass)", [
                             1, 2, 3], index=2, help="1 = Primera clase (Alta), 3 = Tercera clase (Baja)")
genero = st.sidebar.radio("Género / Sexo", ["Femenino", "Masculino"])
edad = st.sidebar.slider("Edad (Años)", 1, 100, 25)
tarifa_manual = st.sidebar.slider("Tarifa Pagada ($ USD)", 0.0, 500.0, 32.0)

# Procesamiento de variables de entrada para los modelos
sexo_num = 1 if genero == "Femenino" else 0
datos_entrada_log = np.array([[clase, sexo_num, edad, tarifa_manual]])

# 6. DISTRIBUCIÓN DE LA PÁGINA EN PESTAÑAS (TABS)
tab1, tab2, tab3 = st.tabs(
    ["🔮 Simulador Inteligente", "📊 Gráficos Estadísticos", "🕶️ Realidad Aumentada"])

# --- PESTAÑA 1: SIMULADOR DE PASAJEROS ---
with tab1:
    st.subheader("Simulación de Escenarios en Tiempo Real")
    st.write("A continuación se muestran las respuestas calculadas simultáneamente por tus modelos de Regresión Logística y Lineal.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Clasificación: ¿Sobreviviría al Naufragio?")
        probabilidad = modelo_logistica.predict_proba(datos_entrada_log)[0][1]
        prediccion = modelo_logistica.predict(datos_entrada_log)[0]

        if prediccion == 1:
            st.balloons()
            st.markdown(f"""
                <div class='metric-card' style='border-left-color: #2ec4b6;'>
                    <h3 style='color: #2ec4b6; margin-top:0;'>🟢 ESTADO: SOBREVIVE</h3>
                    <p>La Regresión Logística estima una probabilidad de supervivencia del <b>{probabilidad:.2%}</b> para este perfil.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='metric-card' style='border-left-color: #e71d36;'>
                    <h3 style='color: #e71d36; margin-top:0;'>🔴 ESTADO: NO SOBREVIVE</h3>
                    <p>La Regresión Logística estima una probabilidad de supervivencia de apenas el <b>{probabilidad:.2%}</b>.</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Estimación: Costo Justo de su Boleto")
        # Predecimos la tarifa justa según la clase, edad, género y si asumimos que sobrevive
        datos_entrada_lin = np.array([[clase, sexo_num, edad, 1]])
        tarifa_predicha = modelo_lineal.predict(datos_entrada_lin)[0]

        # Evitar tarifas negativas por el comportamiento de la regresión lineal simple
        tarifa_final = max(0.0, tarifa_predicha)

        st.markdown(f"""
            <div class='metric-card' style='border-left-color: #ff9f1c;'>
                <h3 style='color: #ff9f1c; margin-top:0;'>💰 TARIFA CALCULADA: ${tarifa_final:.2f} USD</h3>
                <p>La Regresión Lineal proyecta que un boleto con estas características demográficas correspondía a ese valor de mercado.</p>
            </div>
        """, unsafe_allow_html=True)

# --- PESTAÑA 2: GRÁFICOS INTERACTIVOS (Análisis de Datos Históricos) ---
with tab2:
    st.subheader("Exploración Gráfica del Dataset Histórico")
    st.write("Estos gráficos interactivos permiten contrastar tus predicciones con las proporciones reales del evento histórico.")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Gráfico 1: Distribución de supervivencia por género
        df_plot = df_historico.copy()
        df_plot['Survived_Text'] = df_plot['Survived'].map(
            {0: 'Falleció', 1: 'Sobrevivió'})

        fig1 = px.histogram(
            df_plot,
            x="Sex",
            color="Survived_Text",
            barmode="group",
            title="Tasa de Supervivencia Real por Género",
            labels={"Sex": "Género", "Survived_Text": "Resultado"},
            color_discrete_map={'Falleció': '#e71d36', 'Sobrevivió': '#2ec4b6'}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        # Gráfico 2: Relación Edad vs Precio del boleto por clase
        fig2 = px.scatter(
            df_historico,
            x="Age",
            y="Fare",
            color="Pclass",
            title="Relación de Tarifas según la Edad y Clase",
            labels={"Age": "Edad",
                    "Fare": "Precio del Boleto ($)", "Pclass": "Clase"},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- PESTAÑA 3: CONCEPTO DE REALIDAD AUMENTADA ---
with tab3:
    st.subheader("Fusión Tecnológica: Proyección de imagenes")
    st.write("Esta sección simula la integración de interfaces visuales")

    # Intentar buscar la imagen dentro de tu carpeta personalizada
    ruta_imagen = "Realidad Aumentada/tu_imagen_ra 1.png"

    if os.path.exists(ruta_imagen):
        col_img, col_txt = st.columns([1, 1])
        with col_img:
            st.image(ruta_imagen, caption="Prototipo de Visualización Holográfica de Datos",
                     use_container_width=True)
        with col_txt:
            st.info("""
            **💡 Nota de Innovación Técnica:**
            En el diseño final de la Landing Page, los usuarios interactúan con este gemelo digital tridimensional. La aplicación de la Inteligencia Artificial Generativa nos permite prototipar cómo la Realidad Aumentada superpone capas de datos analíticos (probabilidades de evacuación, puntos estructurales de quiebre y flujos de pasajeros) directamente sobre gemelos digitales en entornos inmersivos reales.
            """)
    else:
        st.warning(
            f"⚠️ No se encontró el archivo de imagen en la ruta: `{ruta_imagen}`.")
        st.info("Para solucionarlo, asegúrate de colocar tu imagen dentro de la carpeta 'Realidad Aumentada' y que su nombre sea exactamente `tu_imagen_ra.png`.")
