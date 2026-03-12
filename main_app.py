import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuración de la página
st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide")

st.title("👗 Clasificador Fashion MNIST")
st.markdown("Selecciona los parámetros de tu red neuronal (MLP/DNN) y predice prendas de vestir.")

# 1. Carga de Datos (Cache para evitar recargas lentas)
@st.cache_data
def load_data():
    # Fetch Fashion MNIST desde OpenML
    mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X = mnist.data / 255.0  # Normalización
    y = mnist.target
    target_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    return X, y, target_names

X, y, class_names = load_data()

# Sidebar: Configuración del Modelo
st.sidebar.header("Configuración de la Red")
activation = st.sidebar.selectbox("Función de Activación", ["relu", "tanh", "logistic", "identity"])
n_layers = st.sidebar.slider("Número de Capas Ocultas", 1, 5, 2)
neurons_per_layer = st.sidebar.number_input("Neuronas por capa", min_value=10, max_value=500, value=100)

# Construir la estructura de capas (Tupla de neuronas)
hidden_layers = tuple([neurons_per_layer] * n_layers)

# Botón para entrenar
if st.sidebar.button("Entrenar Modelo"):
    with st.spinner('Entrenando... esto puede tardar un momento.'):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # El MLPClassifier de sklearn actúa como DNN según la profundidad definida
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            max_iter=20, # Reducido para rapidez en demo, aumentar para mejor desempeño
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Guardar en estado de sesión para predicciones individuales
        st.session_state['model'] = model
        st.session_state['accuracy'] = accuracy_score(y_test, y_pred)
        st.session_state['report'] = classification_report(y_test, y_pred, target_names=class_names)

# --- Mostrar Desempeño ---
if 'accuracy' in st.session_state:
    st.subheader("📊 Desempeño de la Red")
    col1, col2 = st.columns(2)
    col1.metric("Precisión (Accuracy)", f"{st.session_state['accuracy']:.2%}")
    col2.text("Reporte de Clasificación:")
    col2.code(st.session_state['report'])

---

# 2. Selección de Imagen y Predicción
st.subheader("🎯 Predicción Individual")
img_idx = st.slider("Selecciona el índice de una imagen del dataset", 0, len(X)-1, 100)

col_img, col_pred = st.columns(2)

# Mostrar imagen seleccionada
fig, ax = plt.subplots()
ax.imshow(X[img_idx].reshape(28, 28), cmap='gray')
ax.axis('off')
col_img.pyplot(fig)
col_img.write(f"Etiqueta real: **{class_names[int(y[img_idx])]}**")

# Realizar predicción
if 'model' in st.session_state:
    prediction = st.session_state['model'].predict([X[img_idx]])
    predicted_label = class_names[int(prediction[0])]
    col_pred.success(f"La red predice que es: **{predicted_label}**")
else:
    col_pred.info("Primero configura y entrena el modelo en la barra lateral.")
