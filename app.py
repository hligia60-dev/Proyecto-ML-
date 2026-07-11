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
    page_title="Simulador Analítico - Titanic ML",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. DISEÑO Y COLORES PERSONALIZADOS (CSS Avanzado)
st.markdown("""
    <style>
    /* Importar fuente premium */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@600;700;800;900&family=JetBrains+Mono&display=swap');
    
    /* Configuración global de fuentes */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Estilos para títulos principales */
    .main-title {
        font-family: 'Outfit', sans-serif;
        color: #0d1b2a;
        font-size: 44px;
        font-weight: 900;
        text-align: center;
        margin-top: -30px;
        margin-bottom: 2px;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #5c677d;
        font-size: 18px;
        font-weight: 400;
        text-align: center;
        margin-bottom: 40px;
    }
    
    /* Tarjetas de Métricas Premium */
    .metric-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        margin-top: 15px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
    }
    
    /* Bloques de fórmulas */
    .math-block {
        font-family: 'JetBrains Mono', monospace;
        background: #0f172a;
        color: #00f2fe;
        padding: 15px;
        border-radius: 10px;
        font-size: 13px;
        margin-bottom: 20px;
        border-left: 4px solid #4facfe;
    }
    
    /* Estilo del Badge de la Autora en el Sidebar */
    .author-sidebar-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }
    
    .author-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #00f2fe;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .author-name {
        font-family: 'Outfit', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }
    
    .author-desc {
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.5;
    }
    
    /* Pie de página */
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
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
st.markdown("<div class='main-title'>🚢 Simulador Analítico del Titanic</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predicción Inteligente mediante CRISP-ML e Interfaces Visuales de Realidad Aumentada</div>", unsafe_allow_html=True)

# 5. MENÚ LATERAL (SIDEBAR) - Entrada de Datos del Usuario
st.sidebar.header("🎯 Parámetros de Simulación")
st.sidebar.write("Modifica los valores para calcular las predicciones en tiempo real.")

clase = st.sidebar.selectbox(
    "Clase del Boleto (Pclass)", 
    [1, 2, 3], 
    index=2, 
    help="1 = Primera clase (Alta), 2 = Segunda clase (Media), 3 = Tercera clase (Baja)"
)
genero = st.sidebar.radio("Género / Sexo", ["Femenino", "Masculino"])
edad = st.sidebar.slider("Edad (Años)", 1, 100, 25)
tarifa_manual = st.sidebar.slider("Tarifa Pagada ($ USD de la época)", 0.0, 500.0, 32.0)

# Procesamiento de variables de entrada para los modelos
sexo_num = 1 if genero == "Femenino" else 0
datos_entrada_log = np.array([[clase, sexo_num, edad, tarifa_manual]])

# Tarjeta de la Autora en la barra lateral
st.sidebar.markdown("""
    <div class='author-sidebar-card'>
        <div class='author-title'>Directora del Proyecto</div>
        <div class='author-name'>Ligia Elena Herrera</div>
        <div class='author-desc'>
            Científica de datos encargada del modelado analítico supervisado y diseño de entornos de realidad mixta para la preservación de memoria histórica.
        </div>
    </div>
""", unsafe_allow_html=True)

# 6. DISTRIBUCIÓN DE LA PÁGINA EN PESTAÑAS (TABS)
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Simulador Interactivo", 
    "🧠 Modelos de Machine Learning", 
    "📊 Gráficos Estadísticos", 
    "🕶️ Gemelos Digitales (RA)"
])

# --- PESTAÑA 1: SIMULADOR DE PASAJEROS ---
with tab1:
    st.markdown("### Simulación de Escenarios en Tiempo Real")
    st.write("A continuación se muestran los resultados calculados en paralelo por los modelos supervisados de Regresión Logística (Clasificación) y Regresión Lineal (Estimación).")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Clasificación: ¿Sobreviviría al Naufragio?")
        probabilidad = modelo_logistica.predict_proba(datos_entrada_log)[0][1]
        prediccion = modelo_logistica.predict(datos_entrada_log)[0]

        if prediccion == 1:
            st.balloons()
            st.markdown(f"""
                <div class='metric-card' style='border-left: 6px solid #10b981;'>
                    <h3 style='color: #10b981; margin-top:0; font-family: "Outfit", sans-serif;'>🟢 PREDICCIÓN: SOBREVIVE</h3>
                    <p style='font-size: 15px; color: #374151; margin-bottom: 5px;'>La Regresión Logística estima una probabilidad de supervivencia del <b>{probabilidad:.2%}</b> para este perfil de pasajero.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='metric-card' style='border-left: 6px solid #ef4444;'>
                    <h3 style='color: #ef4444; margin-top:0; font-family: "Outfit", sans-serif;'>🔴 PREDICCIÓN: NO SOBREVIVE</h3>
                    <p style='font-size: 15px; color: #374151; margin-bottom: 5px;'>La Regresión Logística estima una probabilidad de supervivencia de apenas el <b>{probabilidad:.2%}</b> para este perfil.</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Estimación: Costo Justo de su Boleto")
        # Predecimos la tarifa justa según la clase, edad, género y si asumimos que sobrevive
        datos_entrada_lin = np.array([[clase, sexo_num, edad, 1]])
        tarifa_predicha = modelo_lineal.predict(datos_entrada_lin)[0]

        # Evitar tarifas negativas por el comportamiento de la regresión lineal
        tarifa_final = max(0.0, tarifa_predicha)

        st.markdown(f"""
            <div class='metric-card' style='border-left: 6px solid #4facfe;'>
                <h3 style='color: #4facfe; margin-top:0; font-family: "Outfit", sans-serif;'>💰 TARIFA ESTIMADA: ${tarifa_final:.2f} USD</h3>
                <p style='font-size: 15px; color: #374151; margin-bottom: 5px;'>La Regresión Lineal proyecta que un boleto con estas características demográficas correspondía a ese valor de mercado histórico.</p>
            </div>
        """, unsafe_allow_html=True)

# --- PESTAÑA 2: DOCUMENTACIÓN DE MODELOS ---
with tab2:
    st.markdown("### Explicación de Modelos de Machine Learning")
    st.write("Detalles matemáticos, características clave y rendimiento de los modelos supervisados integrados en la aplicación:")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("#### 1. Regresión Logística (Supervivencia)")
        st.markdown("""
        **Tipo de Tarea**: Clasificación Binaria
        
        Este modelo calcula la probabilidad de supervivencia mapeando una combinación lineal de las características del pasajero mediante la función logística (sigmoide):
        """)
        
        st.markdown("<div class='math-block'>P(Y=1|X) = 1 / (1 + e^-(Z))<br>donde Z = β₀ + β₁*Pclass + β₂*Sex + β₃*Age + β₄*Fare</div>", unsafe_allow_html=True)
        
        st.markdown("""
        *   **Variables de Entrada**:
            *   `Pclass` (Clase): Indica el nivel socioeconómico de la cubierta del pasajero.
            *   `Sex` (Género): Recodificado (Mujer: 1, Hombre: 0). Refleja la política de 'mujeres y niños primero'.
            *   `Age` (Edad): Edad continua del pasajero.
            *   `Fare` (Tarifa): Costo del boleto pagado.
        *   **Objetivo (`Survived`)**: 1 si el modelo predice supervivencia, 0 de lo contrario.
        *   **Métrica de Precisión (Accuracy)**: **80.45%** de acierto global.
        """)
        
    with col_m2:
        st.markdown("#### 2. Regresión Lineal (Tarifa Justa)")
        st.markdown("""
        **Tipo de Tarea**: Regresión Cuantitativa
        
        Diseñado para estimar el costo económico de un boleto basándose en el perfil demográfico y el estado final de supervivencia del pasajero:
        """)
        
        st.markdown("<div class='math-block'>Fare = β₀ + β₁*Pclass + β₂*Sex + β₃*Age + β₄*Survived</div>", unsafe_allow_html=True)
        
        st.markdown("""
        *   **Variables de Entrada**:
            *   `Pclass` (Clase): Principal condicionante de la tarifa.
            *   `Sex` (Género): Impacto demográfico en la asignación del boleto.
            *   `Age` (Edad): Para modelar tarifas diferenciadas por ciclo de vida.
            *   `Survived` (Supervivencia): Variable indicadora si el pasaje influyó en su supervivencia final.
        *   **Objetivo (`Fare`)**: Valor de mercado continuo proyectado en USD.
        *   **Varianza Explicada (R²)**: **33.87%**, lo cual refleja la alta dispersión y variabilidad de las tarifas históricas.
        """)

# --- PESTAÑA 3: GRÁFICOS INTERACTIVOS ---
with tab3:
    st.markdown("### Exploración Gráfica del Dataset Histórico")
    st.write("Gráficos interactivos para contrastar los parámetros simulados con el dataset histórico real (`titanic.csv`).")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        df_plot = df_historico.copy()
        df_plot['Survived_Text'] = df_plot['Survived'].map({0: 'Falleció', 1: 'Sobrevivió'})

        fig1 = px.histogram(
            df_plot,
            x="Sex",
            color="Survived_Text",
            barmode="group",
            title="Tasa de Supervivencia Real por Género",
            labels={"Sex": "Género", "Survived_Text": "Resultado"},
            color_discrete_map={'Falleció': '#ef4444', 'Sobrevivió': '#10b981'}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        fig2 = px.scatter(
            df_historico,
            x="Age",
            y="Fare",
            color="Pclass",
            title="Relación de Tarifas según la Edad y Clase de Boleto",
            labels={"Age": "Edad", "Fare": "Precio del Boleto ($)", "Pclass": "Clase"},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- PESTAÑA 4: GALERÍA DE GEMELOS DIGITALES (RA) ---
with tab4:
    st.markdown("### Galería de Gemelos Digitales e Inteligencia Visual")
    st.write("Selecciona una de las simulaciones y reconstrucciones virtuales generadas en Realidad Aumentada y Analítica de Inteligencia Visual:")
    
    # Listado estructurado de imágenes y metadatos
    imagenes_galeria = {
        "Estructura Holográfica": {
            "ruta": "Realidad Aumentada/tu_imagen_ra 1.png",
            "badge": "Holograma RA",
            "desc": "Simulación tridimensional interactiva que proyecta la estructura externa y las cubiertas del barco. Ayuda a visualizar las diferentes áreas asignadas a las clases de pasajeros, permitiendo entender espacialmente la distribución física del barco."
        },
        "Simulación de Evacuación": {
            "ruta": "Realidad Aumentada/tu_imagen_ra 2.png",
            "badge": "Modelado e Inmersión",
            "desc": "Modelado inmersivo en realidad aumentada que superpone rutas de escape virtuales y cuellos de botella en los pasillos principales. Muestra cómo la disposición interna influyó directamente en las tasas de supervivencia de las cubiertas inferiores."
        },
        "Puntos de Quiebre Estructural": {
            "ruta": "Realidad Aumentada/tu_imagen_ra 3.png",
            "badge": "Modelado e Inmersión",
            "desc": "Visualización detallada en RA del punto de quiebre estructural del Titanic. Este modelo permite estudiar el impacto físico del iceberg y la posterior inundación de los compartimentos estancos en tiempo real."
        },
        "Gemelo Digital con HUD": {
            "ruta": "Realidad Aumentada/titanic_digital_twin.png",
            "badge": "Holograma RA",
            "desc": "Representación futurista de un gemelo digital del Titanic en un entorno de laboratorio técnico de analistas. Muestra HUDs de datos estadísticos y métricas de Machine Learning proyectadas directamente sobre la silueta tridimensional holográfica del barco."
        },
        "Panel Analítico de Modelos": {
            "ruta": "Realidad Aumentada/titanic_analytics_dashboard.png",
            "badge": "Analítica & Machine Learning",
            "desc": "Panel premium interactivo que consolida la analítica del dataset del Titanic, mostrando gráficos de supervivencia por género y clase social, facilitando la comprensión de los pesos matemáticos dentro de los modelos supervisados."
        },
        "Pipeline Analítico & Variables": {
            "ruta": "Realidad Aumentada/titanic_ml_pipeline.png",
            "badge": "Analítica & Machine Learning",
            "desc": "Visualización conceptual de la arquitectura de datos del proyecto Titanic ML. Ilustra el flujo lógico del procesamiento de datos, entrenamiento de modelos supervisados y el renderizado final de métricas estadísticas."
        }
    }
    
    # Selector dinámico de la galería
    seleccion = st.selectbox("Selecciona una simulación para examinar:", list(imagenes_galeria.keys()))
    
    # Renderizado de la imagen seleccionada con su tarjeta técnica
    item = imagenes_galeria[seleccion]
    if os.path.exists(item["ruta"]):
        col_img, col_txt = st.columns([1.2, 0.8])
        with col_img:
            st.image(item["ruta"], use_container_width=True)
        with col_txt:
            st.markdown(f"### {seleccion}")
            st.markdown(f"**Categoría:** `{item['badge']}`")
            st.info(item["desc"])
    else:
        st.warning(f"⚠️ No se encontró la imagen en la ruta `{item['ruta']}`.")

# 7. PIE DE PÁGINA
st.markdown("""
    <div class='app-footer'>
        Proyecto Titanic ML e Inteligencia Visual | Creado por <strong>Ligia Elena Herrera</strong> © 2026
    </div>
""", unsafe_allow_html=True)
