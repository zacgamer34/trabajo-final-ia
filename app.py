import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


class RedDesercion(nn.Module):
    # Misma arquitectura que en el entrenamiento (si cambia allá, debe cambiar aquí)
    def __init__(self, input_dim, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.red(x)


@st.cache_resource          # carga una sola vez, no en cada clic
def cargar_artefactos():
    info = joblib.load("trabajo final/feature_info.pkl")
    scaler = joblib.load("trabajo final/scaler.pkl")
    ckpt = torch.load("trabajo final/modelo_desercion.pt", map_location="cpu")
    modelo = RedDesercion(**ckpt["config_arquitectura"])
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()               # apaga el Dropout
    return info, scaler, modelo


def preparar_entrada(df_raw, info, scaler):
    # Idéntica a la del notebook: codifica y escala igual que en el entrenamiento
    d = df_raw[info["columnas_modelo"]].copy()
    if info["usar_onehot"]:
        for col in info["columnas_categoricas"]:
            d[col] = pd.Categorical(d[col], categories=info["niveles"][col])
        d = pd.get_dummies(d, columns=info["columnas_categoricas"], dtype=int)
    d = d.reindex(columns=info["columnas_finales"], fill_value=0)
    return scaler.transform(d.values)


ETIQUETAS = {
    "Mother's qualification": "Nivel de estudios de la madre (código)",
    "Father's qualification": "Nivel de estudios del padre (código)",
    "Mother's occupation": "Ocupación de la madre (código)",
    "Father's occupation": "Ocupación del padre (código)",
    "Debtor": "Tiene deudas con la institución",
    "Tuition fees up to date": "Matrícula al día",
    "Scholarship holder": "Tiene beca",
}
BINARIAS = ["Debtor", "Tuition fees up to date", "Scholarship holder"]


def etiqueta(col):
    if col in ETIQUETAS:
        return ETIQUETAS[col]
    sem = "1.er semestre" if "1st sem" in col else "2.º semestre"
    tipo = col[col.index("(") + 1:-1]
    nombres = {"credited": "unidades convalidadas", "enrolled": "unidades matriculadas",
               "evaluations": "evaluaciones presentadas", "approved": "unidades aprobadas",
               "grade": "promedio de notas (0 a 20)", "without evaluations": "unidades sin evaluaciones"}
    return f"{sem}: {nombres.get(tipo, tipo)}"


st.set_page_config(page_title="Riesgo de deserción", page_icon="🎓")
st.title("🎓 Predicción de deserción estudiantil")
st.caption("Red neuronal (PyTorch) entrenada con datos académicos y socioeconómicos. "
           "Es una herramienta de apoyo: no reemplaza la decisión de una persona.")

info, scaler, modelo = cargar_artefactos()
defecto = info["valores_defecto"]
datos = {}

with st.form("formulario"):
    st.subheader("Datos socioeconómicos")
    for col in info["columnas_modelo"]:
        if col in info["columnas_categoricas"]:
            opciones = info["niveles"][col]
            ini = opciones.index(int(defecto[col])) if int(defecto[col]) in opciones else 0
            datos[col] = st.selectbox(etiqueta(col), opciones, index=ini)
        elif col in BINARIAS:
            datos[col] = st.selectbox(etiqueta(col), [0, 1], index=int(defecto[col]),
                                      format_func=lambda v: "Sí" if v == 1 else "No")

    st.subheader("Datos académicos")
    for col in info["columnas_modelo"]:
        if col.startswith("Curricular"):
            if "(grade)" in col:
                datos[col] = st.number_input(etiqueta(col), 0.0, 20.0, float(defecto[col]), step=0.1)
            else:
                datos[col] = st.number_input(etiqueta(col), 0, 100, int(defecto[col]), step=1)

    enviar = st.form_submit_button("Calcular riesgo")

if enviar:
    entrada = preparar_entrada(pd.DataFrame([datos]), info, scaler)
    with torch.no_grad():
        prob = torch.sigmoid(modelo(torch.tensor(entrada, dtype=torch.float32))).item()

    st.subheader(f"Probabilidad de abandono: {prob * 100:.1f} %")
    st.progress(min(max(prob, 0.0), 1.0))
    if prob < 0.33:
        st.success("Riesgo BAJO")
    elif prob < 0.66:
        st.warning("Riesgo MEDIO: conviene hacer seguimiento.")
    else:
        st.error("Riesgo ALTO: se recomienda contactar al estudiante y ofrecer apoyo.")
