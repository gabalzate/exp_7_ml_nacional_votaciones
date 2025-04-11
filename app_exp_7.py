import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción Electoral por Ideología",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar el modelo
@st.cache_resource
def load_model():
    try:
        # Intenta cargar el modelo desde la ubicación especificada
        with open('./out/exp7_nacional.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("No se encontró el archivo del modelo. Por favor, asegúrate de que el archivo 'exp7_nacional.pkl' esté en el directorio de la aplicación.")
        return None

# Estilos CSS personalizados
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
    }
    .st-bx {
        background-color: #F8F9FA;
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #E9ECEF;
    }
    .centro-derecha {
        color: #FF7C44;
    }
    .centro-izquierda {
        color: #28A745;
    }
    .derecha {
        color: #DC3545;
    }
    .izquierda {
        color: #0D6EFD;
    }
    .header {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #212529;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
#st.markdown("<h1 style='text-align: center;'>Sistema de Predicción Electoral por Ideología Política</h1>", unsafe_allow_html=True)

# Función para mostrar logo e imagen
def display_header_with_logo():
    col1, col2 = st.columns([1, 2])
    with col1:
        # Logo de Guarumo desde URL
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <img src="https://images.jifo.co/34436825_1542400128284.png" 
                 width="180px">
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<h2 style='margin-top: 5px;'>Sistema de Predicción Electoral</h2>", unsafe_allow_html=True)

# Mostrar el encabezado con logo
display_header_with_logo()

# Línea divisoria para separar el encabezado del contenido
st.markdown("<hr style='margin-top: 0; margin-bottom: 20px;'>", unsafe_allow_html=True)



# Cargar el modelo
model = load_model()

# Definir las ideologías
ideologias = ["Centro derecha", "Centro izquierda", "Derecha", "Izquierda"]
colors = ["#FF7C44", "#4ADE80", "#EF4444", "#3B82F6"]
color_map = dict(zip(ideologias, colors))

# Crear dos columnas para la interfaz
col1, col2 = st.columns([1, 1])

# Columna izquierda - Formulario de entrada
with col1:
    st.markdown("<div class='header'>Parámetros de Entrada</div>", unsafe_allow_html=True)
    
    # Crear un diccionario para almacenar los valores de entrada
    input_data = {}
    
    # Para cada ideología, crear un formulario de entrada
    for i, ideologia in enumerate(ideologias):
        color_class = ideologia.lower().replace(" ", "-")
        st.markdown(f"<div class='st-bx'><div class='{color_class} subheader'>{ideologia}</div>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            google_mensual = st.number_input(
                f"Google Mensual", 
                min_value=0.0, 
                max_value=100.0, 
                value=15.0, 
                step=0.1,
                key=f"gm_{i}"
            )
            google_semanal = st.number_input(
                f"Google Semanal", 
                min_value=0.0, 
                max_value=100.0, 
                value=15.0, 
                step=0.1,
                key=f"gs_{i}"
            )
        
        with col_b:
            encuesta = st.number_input(
                f"Encuesta", 
                min_value=0.0, 
                max_value=100.0, 
                value=15.0, 
                step=0.1,
                key=f"enc_{i}"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Almacenar los valores en el diccionario
        input_data[ideologia] = {
            "GOOGLE_MENSUAL": google_mensual,
            "GOOGLE_SEMANAL": google_semanal,
            "encuesta": encuesta
        }
    
    # Botón para realizar la predicción
    predict_button = st.button("Predecir", type="primary")

# Columna derecha - Resultados
with col2:
    st.markdown("<div class='header'>Resultados de la Predicción</div>", unsafe_allow_html=True)
    
    if predict_button and model is not None:
        # Crear el DataFrame con los datos de entrada
        rows = []
        for ideologia, values in input_data.items():
            row = {
                "IDEOLOGIA": ideologia,
                "GOOGLE_MENSUAL": values["GOOGLE_MENSUAL"],
                "GOOGLE_SEMANAL": values["GOOGLE_SEMANAL"],
                "encuesta": values["encuesta"]
            }
            rows.append(row)
        
        df_input = pd.DataFrame(rows)
        
        # Procesamiento similar al notebook
        # Codificación one-hot para la columna IDEOLOGIA
        df_encoded = pd.get_dummies(df_input, columns=["IDEOLOGIA"], prefix=["IDEOLOGIA"])
        
        # Asegurarse de que todas las columnas esperadas estén presentes
        expected_columns = [
            'GOOGLE_MENSUAL', 'GOOGLE_SEMANAL', 'encuesta',
            'IDEOLOGIA_Centro derecha', 'IDEOLOGIA_Centro izquierda',
            'IDEOLOGIA_Derecha', 'IDEOLOGIA_Izquierda'
        ]
        
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Seleccionar solo las columnas esperadas y en el orden correcto
        X = df_encoded[expected_columns]
        
        # Escalar las variables numéricas
        numerical_columns = ['GOOGLE_MENSUAL', 'GOOGLE_SEMANAL', 'encuesta']
        scaler = StandardScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        
        # Realizar la predicción
        predictions = model.predict(X)
        
        # Asignar las predicciones al DataFrame original
        df_input['resultado'] = predictions
        
        # Crear el gráfico de barras para las predicciones
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('default')
        
        # Ordenar por valor de predicción descendente
        df_sorted = df_input.sort_values(by='resultado', ascending=False)
        
        # Crear barras con colores según la ideología
        bars = plt.barh(
            y=df_sorted['IDEOLOGIA'],
            width=df_sorted['resultado'],
            color=[color_map[i] for i in df_sorted['IDEOLOGIA']]
        )
        
        # Añadir etiquetas a las barras
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                    va='center', ha='left', fontsize=10)
        
        plt.xlabel('Resultado predicho (%)')
        plt.title('Predicción de Resultados por Ideología Política', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Mostrar tabla de resultados
        st.markdown("<div class='st-bx'><div class='subheader'>Detalle de Resultados</div>", unsafe_allow_html=True)
        
        # Formatear los resultados para la tabla
        result_df = df_input.copy()
        result_df['resultado'] = result_df['resultado'].apply(lambda x: f"{x:.2f}%")
        result_df = result_df.rename(columns={
            'IDEOLOGIA': 'Ideología',
            'resultado': 'Resultado (%)',
            'GOOGLE_MENSUAL': 'Google Mensual',
            'GOOGLE_SEMANAL': 'Google Semanal',
            'encuesta': 'Encuesta'
        })
        
        # Identificar la ideología con mayor predicción
        max_ideologia = df_input.loc[df_input['resultado'].idxmax(), 'IDEOLOGIA']
        
        # Añadir columna de mayor predicción
        result_df['Mayor predicción'] = result_df['Ideología'].apply(
            lambda x: "Sí" if x == max_ideologia else "No"
        )
        
        # Mostrar solo columnas relevantes
        st.dataframe(
            result_df[['Ideología', 'Resultado (%)','Mayor predicción']], 
            hide_index=True,
            use_container_width=True
        )
        
        # Mostrar suma total
        total_sum = df_input['resultado'].sum()
        st.markdown(f"<div style='text-align: center; margin-top: 10px;'>Suma total de predicciones: {total_sum:.2f}%</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Mostrar un placeholder cuando no hay predicciones
        st.info("Completa los datos de entrada y haz clic en 'Predecir' para ver los resultados.")
        
        # Opcionalmente, mostrar una imagen de ejemplo
        placeholder_chart = """
        <div style="background-color: #1E2030; padding: 20px; border-radius: 5px; height: 300px; display: flex; justify-content: center; align-items: center;">
            <div style="text-align: center; color: #718096;">
                <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M21 21H4.6C4.03995 21 3.75992 21 3.54601 20.891C3.35785 20.7951 3.20487 20.6422 3.10899 20.454C3 20.2401 3 19.9601 3 19.4V3" stroke="#718096" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M7 15L11 11L15 15L20 10" stroke="#718096" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <p>Los resultados de la predicción se mostrarán aquí</p>
            </div>
        </div>
        """
        st.markdown(placeholder_chart, unsafe_allow_html=True)
        
        # Placeholder para la tabla
        placeholder_table = """
        <div style="background-color: #F8F9FA; padding: 20px; border-radius: 5px; margin-top: 20px; border: 1px solid #E9ECEF;">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #212529;">Detalle de Resultados</div>
            <div style="height: 170px; display: flex; justify-content: center; align-items: center;">
                <div style="text-align: center; color: #6C757D;">
                    <p>La tabla de resultados aparecerá aquí</p>
                </div>
            </div>
        </div>
        """
        st.markdown(placeholder_table, unsafe_allow_html=True)

# Pie de página
st.markdown("""
<div style="text-align: center; margin-top: 30px; font-size: 12px; color: #6C757D;">
    Desarrollado con Streamlit | Modelo XGBoost entrenado con datos electorales
</div>
""", unsafe_allow_html=True)