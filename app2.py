# =========================================================
# CGM + AI DIABETES PREDICTION APP
# STREAMLIT PROFESSIONAL VERSION
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Diabetes Dashboard",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #0e1117;
}

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("🩺 Predizione del Diabete con IA + Simulatore di Monitoraggio Continuo della Glicemia (CGM)")

st.markdown("""
Funzionalità principali:

✅ Machine Learning  
✅ Predizione rischio diabete  
✅ Simulazione glicemica  
✅ Stima dell'insulina  
✅ Dashboard clinica  

""")

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    return df

df = load_data()

# =========================================================
# DATASET PREVIEW
# =========================================================

st.header("📊 Dataset")

st.dataframe(df.head())

st.write(f"Dataset shape: {df.shape}")

# =========================================================
# DATA CLEANING
# =========================================================

df = df.dropna()

# =========================================================
# ENCODING CATEGORICAL VARIABLES
# =========================================================

label_encoders = {}

categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

# =========================================================
# FEATURES AND TARGET
# =========================================================

target_column = "diabetes"

X = df.drop(target_column, axis=1)

y = df[target_column]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# MODEL EVALUATION
# =========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# =========================================================
# DIABETES RISK CALCULATOR
# =========================================================

st.header("🩺 Calcolatore Rischio di Diabete")

st.write(
    "Inserisci i dati del paziente per ottenere "
    "una stima del rischio di diabete."
)

# ---------------------------------------------------------
# DATI DEL PAZIENTE
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Sesso",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Età (anni)",
        min_value=1,
        max_value=100,
        value=45
    )

    weight = st.number_input(
        "Peso (kg)",
        min_value=20.0,
        max_value=250.0,
        value=70.0,
        step=0.5
    )

    height = st.number_input(
        "Altezza (cm)",
        min_value=100.0,
        max_value=230.0,
        value=170.0,
        step=1.0
    )


with col2:

    blood_glucose_level = st.number_input(
        "Glicemia (mg/dL)",
        min_value=50,
        max_value=300,
        value=100,
        step=1
    )

    HbA1c_level = st.number_input(
        "HbA1c (%)",
        min_value=3.0,
        max_value=15.0,
        value=5.5,
        step=0.1
    )

    hypertension = st.selectbox(
        "Ipertensione",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Sì"
    )

    heart_disease = st.selectbox(
        "Malattia cardiovascolare",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Sì"
    )

    smoking_history = st.selectbox(
        "Abitudine al fumo",
        [
            "never",
            "former",
            "current",
            "not current"
            "No Info"
        ]
    )


# =========================================================
# CALCOLO BMI
# =========================================================

height_m = height / 100

bmi = weight / (height_m ** 2)

st.info(
    f"**BMI del paziente: {bmi:.1f}**"
)



# =========================================================
# MEMORIA DEL RISULTATO
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "risk_percentage" not in st.session_state:
    st.session_state.risk_percentage = None

if "prediction_proba" not in st.session_state:
    st.session_state.prediction_proba = None


# =========================================================
# PREDIZIONE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "risk_percentage" not in st.session_state:
    st.session_state.risk_percentage = None

if "prediction_proba" not in st.session_state:
    st.session_state.prediction_proba = None


if st.button("🔍 Calcola rischio di diabete"):

    # Encoding sesso
    gender_encoded = label_encoders["gender"].transform(
        [gender]
    )[0]

    # Encoding fumo
    smoking_encoded = label_encoders["smoking_history"].transform(
        [smoking_history]
    )[0]

    # Dati paziente
    input_data = pd.DataFrame({
        "gender": [gender_encoded],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_encoded],
        "bmi": [bmi],
        "HbA1c_level": [HbA1c_level],
        "blood_glucose_level": [blood_glucose_level]
    })

    # Predizione
    st.session_state.prediction = model.predict(
        input_data
    )[0]

    # Probabilità
    st.session_state.prediction_proba = model.predict_proba(
        input_data
    )[0][1]

    # Percentuale
    st.session_state.risk_percentage = (
        st.session_state.prediction_proba * 100
    )

    # Risultato
    st.subheader("📊 Risultato")

    col_risk, col_prediction = st.columns(2)

    with col_risk:

        st.metric(
            "Rischio stimato di diabete",
            f"{st.session_state.risk_percentage:.1f}%"
        )

    with col_prediction:

        if st.session_state.prediction == 1:

            st.metric(
                "Predizione del modello",
                "Alto rischio"
            )

        else:

            st.metric(
                "Predizione del modello",
                "Basso rischio"
            )

    # Messaggio
    if st.session_state.prediction == 1:

        st.error(
            f"⚠️ Il modello stima un rischio elevato "
            f"({st.session_state.risk_percentage:.1f}%)."
        )

    else:

        st.success(
            f"✅ Il modello stima un rischio basso "
            f"({st.session_state.risk_percentage:.1f}%)."
        )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.header("📈 Importanza delle Variabili")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig2, ax2 = plt.subplots(figsize=(10,5))

ax2.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax2.set_title("Importanza delle Variabili")

ax2.set_ylabel("Importance")

plt.xticks(rotation=45)

st.pyplot(fig2)

# =========================================================
# CGM SIMULATOR
# =========================================================

st.header("📉 Simulatore CGM")

foods = {
    "Pasta": 75,
    "Pane": 50,
    "Riso": 80,
    "Pizza": 33,
    "Mela": 14,
    "Banana": 23,
    "Gelato": 24
}

food = st.selectbox(
    "Seleziona Cibo",
    list(foods.keys())
)

grams = st.slider(
    "Quantità (g)",
    0,
    300,
    100
)

meal_time = st.slider(
    "Orario del Pasto",
    0,
    24,
    2
)

carbs = (grams * foods[food]) / 100

st.write(f"Carboidrati stimati: {carbs:.1f} g")

# =========================================================
# CALCOLO INSULINA
# =========================================================

suggested_insulin = 0
insulin_units = 0

if st.session_state.prediction == 1:

    st.subheader("💉 Simulazione insulina")

    carb_ratio = st.slider(
        "Razione di Carboidrati (g/U)",
        5,
        20,
        10
    )

    suggested_insulin = carbs / carb_ratio

    st.warning(
        f"⚠️ Valore teorico della simulazione: "
        f"{suggested_insulin:.1f} U"
    )

    insulin_units = st.slider(
        "Insulina simulata",
        0.0,
        10.0,
        float(min(round(suggested_insulin, 1), 10.0)),
        step=0.1
    )

else:

    st.info(
        "ℹ️ Il paziente non è classificato dal modello "
        "come ad alto rischio: la simulazione dell'insulina "
        "non viene effettuata."
    )
    
# =========================================================
# GLUCOSE SIMULATION
# =========================================================

time = np.linspace(0, 24, 400)

baseline = blood_glucose_level

glucose = np.ones_like(time) * baseline

# meal effect
meal_effect = carbs * np.exp(
    -0.5 * ((time - meal_time)/1.2)**2
)

glucose += meal_effect

# Effetto teorico dell'insulina sulla glicemia
# Usato solo per la simulazione grafica
insulin_sensitivity = 8

insulin_effect = insulin_units * insulin_sensitivity * np.exp(
    -0.5 * ((time - (meal_time + 1)) / 2)**2
)

glucose -= insulin_effect

# physiological noise
noise = np.random.normal(0, 4, len(time))

glucose += noise

# Limite minimo della simulazione
glucose = np.maximum(glucose, 70)

# limits
glucose = np.clip(glucose, 40, 400)

# =========================================================
# CGM GRAPH
# =========================================================

fig3, ax3 = plt.subplots(figsize=(12,5))

ax3.plot(
    time,
    glucose,
    linewidth=3,
    label="Glucosio"
)

ax3.axhline(
    70,
    linestyle="--",
    label="Ipoglicemia"
)

ax3.axhline(
    180,
    linestyle="--",
    label="Iperglicemia"
)

ax3.fill_between(
    time,
    70,
    180,
    alpha=0.1
)

ax3.set_title("CGM")

ax3.set_xlabel("Tempo (ore)")

ax3.set_ylabel("mg/dL")

ax3.legend()

st.pyplot(fig3)

# =========================================================
# CGM ALERTS
# =========================================================

if np.any(glucose < 70):

    st.error("⚠️ Ipoglicemia rilevata")

if np.any(glucose > 180):

    st.warning("⚠️ Iperglicemia rilevata")

# =========================================================
# FINAL REPORT
# =========================================================

st.header("📋 Report Clinico Finale ")

summary = pd.DataFrame({
    "Parametri": [
        "Età",
        "BMI",
        "HbA1c",
        "Glicemia",
        "Carboidrati",
        "Dose Insulina Suggerita"
    ],
    "Valori": [
        age,
        bmi,
        HbA1c_level,
        blood_glucose_level,
        round(carbs, 1),
        round(suggested_insulin, 1)
    ]
})

st.table(summary)

# =========================================================
# DOWNLOAD RESULTS
# =========================================================

csv = summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Report Clinico",
    data=csv,
    file_name="clinical_report.csv",
    mime="text/csv"
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Applicazione a scopo educativo: Non destinata all'uso clinico reale.
""")
