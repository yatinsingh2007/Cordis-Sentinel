"""
Cordis Sentinel — Heart Attack Risk Prediction UI
Modern dark AI-agent theme, single-file Streamlit app.

ASSUMPTIONS (update when sharing your actual dataset/model):
  - Model file  : models/logistic_regression_model.pkl  (LogisticRegression)
  - Scaler file : models/standard_scaler.pkl           (StandardScaler, 12 continuous cols)
  - Encoder file: models/ordinal_encoder.pkl            (OrdinalEncoder, 6 ordinal cols)
  - Feature order: 31 columns matching features_engineered.csv (see FEATURE_COLS below)
  - Metrics JSON: data/processed/model_metrics.json     (precomputed, no re-training on load)

# TODO: agentic reasoning layer — to be added once src/agent/ is built
#   Planned integration point:
#     from src.agent.graph import build_graph
#     agent_graph = build_graph(model, scaler, encoder)
#     agent_output = agent_graph.invoke({"patient_input": raw_inputs, "risk_prob": risk_prob})
#   The agent will add narrative explanations, retrieval-augmented clinical context,
#   and tool-calling for SHAP-based feature attribution.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "logistic_regression_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "standard_scaler.pkl")
ENCODER_PATH= os.path.join(BASE_DIR, "models", "ordinal_encoder.pkl")
METRICS_PATH= os.path.join(BASE_DIR, "data", "processed", "model_metrics.json")

# ─── Feature metadata ─────────────────────────────────────────────────────────
# Exact column order the model was trained on (31 features)
FEATURE_COLS = [
    "age", "resting_bp", "cholesterol", "fasting_blood_sugar", "max_heart_rate",
    "exercise_angina", "oldpeak", "st_slope", "num_major_vessels", "bmi",
    "smoking_status", "alcohol_consumption", "physical_activity",
    "family_history", "diabetes", "stress_level",
    "age_group", "bp_category",
    "hr_age_ratio", "heart_rate_reserve", "cholesterol_bp_ratio",
    "lifestyle_risk_score", "comorbidity_score",
    "gender_Male",
    "chest_pain_type_Atypical Angina", "chest_pain_type_Non-anginal Pain",
    "chest_pain_type_Typical Angina",
    "resting_ecg_Normal", "resting_ecg_ST-T Abnormality",
    "thalassemia_Normal", "thalassemia_Reversible Defect",
]

# Columns the StandardScaler was fitted on (12 continuous)
SCALE_COLS = [
    "age", "resting_bp", "cholesterol", "max_heart_rate", "oldpeak",
    "bmi", "stress_level", "hr_age_ratio", "heart_rate_reserve",
    "cholesterol_bp_ratio", "lifestyle_risk_score", "comorbidity_score",
]

# Human-readable labels for the feature importance chart
FEATURE_LABELS = {
    "age": "Age",
    "resting_bp": "Resting BP",
    "cholesterol": "Cholesterol",
    "fasting_blood_sugar": "Fasting Blood Sugar",
    "max_heart_rate": "Max Heart Rate",
    "exercise_angina": "Exercise Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "st_slope": "ST Slope",
    "num_major_vessels": "Major Vessels",
    "bmi": "BMI",
    "smoking_status": "Smoking Status",
    "alcohol_consumption": "Alcohol Consumption",
    "physical_activity": "Physical Activity",
    "family_history": "Family History",
    "diabetes": "Diabetes",
    "stress_level": "Stress Level",
    "age_group": "Age Group",
    "bp_category": "BP Category",
    "hr_age_ratio": "HR / Age Ratio",
    "heart_rate_reserve": "Heart Rate Reserve",
    "cholesterol_bp_ratio": "Cholesterol/BP Ratio",
    "lifestyle_risk_score": "Lifestyle Risk Score",
    "comorbidity_score": "Comorbidity Score",
    "gender_Male": "Gender (Male)",
    "chest_pain_type_Atypical Angina": "Chest Pain: Atypical Angina",
    "chest_pain_type_Non-anginal Pain": "Chest Pain: Non-anginal",
    "chest_pain_type_Typical Angina": "Chest Pain: Typical Angina",
    "resting_ecg_Normal": "Resting ECG: Normal",
    "resting_ecg_ST-T Abnormality": "Resting ECG: ST-T Abnormality",
    "thalassemia_Normal": "Thalassemia: Normal",
    "thalassemia_Reversible Defect": "Thalassemia: Reversible Defect",
}

# ─── Palette ──────────────────────────────────────────────────────────────────
BG        = "#0A0E1A"
PANEL     = "#121826"
BORDER    = "#232B42"
TEXT      = "#E8ECF4"
MUTED     = "#8891A8"
TEAL      = "#4FD1C5"
CORAL     = "#F2545B"
GREEN     = "#34D399"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cordis Sentinel",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global resets */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0E1A;
    color: #E8ECF4;
}

/* Headers */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #121826 !important;
    border-right: 1px solid #232B42;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #8891A8;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

/* Metric panel cards */
.metric-card {
    background: #121826;
    border: 1px solid #232B42;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4FD1C5; }
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #4FD1C5;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #8891A8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Risk display */
.risk-high { color: #F2545B !important; }
.risk-low  { color: #34D399 !important; }

/* Animated waveform bar */
.waveform-container {
    width: 100%;
    height: 56px;
    overflow: hidden;
    margin-bottom: 0;
    position: relative;
}
.waveform-svg {
    width: 200%;
    height: 100%;
    animation: scroll-wave 6s linear infinite;
}
@keyframes scroll-wave {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #4FD1C5;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #232B42;
}

/* Panel wrapper */
.panel {
    background: #121826;
    border: 1px solid #232B42;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}

/* Vitals readout numbers in sidebar */
.vitals-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #4FD1C5;
}

/* Footer */
.footer-disclaimer {
    background: #121826;
    border: 1px solid #232B42;
    border-left: 3px solid #4FD1C5;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 32px;
    font-size: 0.78rem;
    color: #8891A8;
    line-height: 1.6;
}

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Make plotly charts transparent */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, scaler, encoder, metrics


def engineer_features(raw: dict) -> dict:
    """Reproduce the feature engineering from notebooks/04_feature_engineering.ipynb."""
    age    = raw["age"]
    bp     = raw["resting_bp"]
    chol   = raw["cholesterol"]
    hr     = raw["max_heart_rate"]
    stress = raw["stress_level"]

    # Age group bins
    if   age < 40:  age_group = "<40"
    elif age < 50:  age_group = "40-49"
    elif age < 60:  age_group = "50-59"
    elif age < 70:  age_group = "60-69"
    else:           age_group = "70+"

    # BP category (AHA staging)
    if   bp < 120:  bp_cat = "Normal"
    elif bp < 130:  bp_cat = "Elevated"
    elif bp < 140:  bp_cat = "Stage1"
    else:           bp_cat = "Stage2"

    hr_age_ratio        = hr / (220 - age)
    heart_rate_reserve  = hr - 60
    cholesterol_bp_ratio= chol / (bp + 1e-5)

    smoking_map  = {"Never": 0, "Former": 1, "Current": 2}
    alcohol_map  = {"Non-drinker": 0, "Moderate": 1, "Heavy": 2}
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}

    lifestyle_risk_score = (
        smoking_map[raw["smoking_status"]] +
        alcohol_map[raw["alcohol_consumption"]] +
        activity_map[raw["physical_activity"]]
    )
    comorbidity_score = (
        raw["diabetes"] +
        raw["family_history"] +
        int(stress >= 7)
    )

    return {**raw,
            "age_group": age_group, "bp_category": bp_cat,
            "hr_age_ratio": hr_age_ratio, "heart_rate_reserve": heart_rate_reserve,
            "cholesterol_bp_ratio": cholesterol_bp_ratio,
            "lifestyle_risk_score": lifestyle_risk_score,
            "comorbidity_score": comorbidity_score}


def build_feature_vector(raw: dict, scaler) -> np.ndarray:
    """
    Apply ordinal encoding → OHE → scale → assemble 31-feature vector.
    Mirrors the pipeline in notebooks/04_feature_engineering.ipynb exactly.
    Ordinal maps are hardcoded to match the OrdinalEncoder categories in models/ordinal_encoder.pkl.
    """
    feat = engineer_features(raw)

    # Ordinal encoding (explicit maps, consistent with OrdinalEncoder categories)
    ordinal_maps = {
        "smoking_status":      {"Never": 0, "Former": 1, "Current": 2},
        "alcohol_consumption": {"Non-drinker": 0, "Moderate": 1, "Heavy": 2},
        "physical_activity":   {"Low": 0, "Moderate": 1, "High": 2},
        "st_slope":            {"Down": 0, "Flat": 1, "Up": 2},
        "age_group":           {"<40": 0, "40-49": 1, "50-59": 2, "60-69": 3, "70+": 4},
        "bp_category":         {"Normal": 0, "Elevated": 1, "Stage1": 2, "Stage2": 3},
    }
    for col, mapping in ordinal_maps.items():
        feat[col] = mapping[feat[col]]

    # OHE (drop_first=True — base categories are dropped)
    # gender base: Female   → gender_Male = 1 if Male
    feat["gender_Male"] = 1 if raw["gender"] == "Male" else 0
    # chest_pain_type base: Asymptomatic
    feat["chest_pain_type_Atypical Angina"] = 1 if raw["chest_pain_type"] == "Atypical Angina"  else 0
    feat["chest_pain_type_Non-anginal Pain"]= 1 if raw["chest_pain_type"] == "Non-anginal Pain"  else 0
    feat["chest_pain_type_Typical Angina"]  = 1 if raw["chest_pain_type"] == "Typical Angina"    else 0
    # resting_ecg base: Left Ventricular Hypertrophy
    feat["resting_ecg_Normal"]              = 1 if raw["resting_ecg"]     == "Normal"             else 0
    feat["resting_ecg_ST-T Abnormality"]    = 1 if raw["resting_ecg"]     == "ST-T Abnormality"   else 0
    # thalassemia base: Fixed Defect
    feat["thalassemia_Normal"]              = 1 if raw["thalassemia"]     == "Normal"             else 0
    feat["thalassemia_Reversible Defect"]   = 1 if raw["thalassemia"]     == "Reversible Defect"  else 0

    # Build row in correct column order
    row = pd.DataFrame([{col: feat[col] for col in FEATURE_COLS}])

    # Scale continuous columns in-place
    row[SCALE_COLS] = scaler.transform(row[SCALE_COLS])

    return row.values


# ─── Plotly chart builders ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT),
    margin=dict(l=0, r=0, t=30, b=0),
)


def make_gauge(prob: float) -> go.Figure:
    risk_color = CORAL if prob >= 0.5 else GREEN
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"family": "JetBrains Mono", "size": 40, "color": risk_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER,
                     "tickfont": {"size": 11, "color": MUTED}},
            "bar": {"color": risk_color, "thickness": 0.25},
            "bgcolor": PANEL,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  50],  "color": "#0D1F1A"},
                {"range": [50, 100], "color": "#1F0D0E"},
            ],
            "threshold": {
                "line": {"color": risk_color, "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                      title=dict(text="Risk Probability", font=dict(family="Space Grotesk", size=13, color=MUTED), x=0.5))
    return fig


def make_importance_chart(model) -> go.Figure:
    """Horizontal bar chart of LR coefficients (absolute value ranked)."""
    coefs = model.coef_[0]
    labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
    df_imp = pd.DataFrame({"feature": labels, "coef": coefs})
    df_imp["abs"] = df_imp["coef"].abs()
    df_imp = df_imp.nlargest(15, "abs").sort_values("abs")

    colors = [CORAL if c > 0 else GREEN for c in df_imp["coef"]]

    fig = go.Figure(go.Bar(
        x=df_imp["coef"], y=df_imp["feature"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(title="Logistic Regression Coefficient", gridcolor=BORDER, zeroline=True,
                   zerolinecolor=BORDER, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11)),
        title=dict(text="Top 15 Feature Importances (|LR Coefficients|)",
                   font=dict(family="Space Grotesk", size=13, color=MUTED), x=0),
        bargap=0.3,
    )
    return fig


def make_confusion_matrix(cm_dict: dict) -> go.Figure:
    tn, fp = cm_dict["TN"], cm_dict["FP"]
    fn, tp = cm_dict["FN"], cm_dict["TP"]
    z  = [[tn, fp], [fn, tp]]
    annotations = [
        [f"TN<br>{tn}", f"FP<br>{fp}"],
        [f"FN<br>{fn}", f"TP<br>{tp}"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted: No Risk", "Predicted: At Risk"],
        y=["Actual: No Risk", "Actual: At Risk"],
        text=annotations,
        texttemplate="%{text}",
        colorscale=[[0, "#0D1828"], [1, "#1A4A6E"]],
        showscale=False,
        hovertemplate="<b>%{y} → %{x}</b><br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis=dict(tickfont=dict(size=11, color=MUTED)),
        yaxis=dict(tickfont=dict(size=11, color=MUTED)),
        title=dict(text="Confusion Matrix — Test Set (n=1,400)",
                   font=dict(family="Space Grotesk", size=13, color=MUTED), x=0),
    )
    return fig


# ─── Waveform SVG ─────────────────────────────────────────────────────────────
WAVEFORM_HTML = """
<div class="waveform-container">
  <svg class="waveform-svg" viewBox="0 0 1200 56" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
    <polyline
      points="
        0,28 60,28 80,28 85,8 90,48 95,18 100,38 105,28
        160,28 180,28 185,8 190,48 195,18 200,38 205,28
        260,28 280,28 285,8 290,48 295,18 300,38 305,28
        360,28 380,28 385,8 390,48 395,18 400,38 405,28
        460,28 480,28 485,8 490,48 495,18 500,38 505,28
        560,28 580,28 585,8 590,48 595,18 600,38 605,28
        660,28 680,28 685,8 690,48 695,18 700,38 705,28
        760,28 780,28 785,8 790,48 795,18 800,38 805,28
        860,28 880,28 885,8 890,48 895,18 900,38 905,28
        960,28 980,28 985,8 990,48 995,18 1000,38 1005,28
        1060,28 1080,28 1085,8 1090,48 1095,18 1100,38 1105,28
        1160,28 1180,28 1185,8 1190,48 1195,18 1200,38
      "
      fill="none"
      stroke="#4FD1C5"
      stroke-width="1.5"
      stroke-opacity="0.45"
    />
  </svg>
</div>
"""


# ─── Sidebar: patient input form ───────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown('<p style="font-family:\'Space Grotesk\',sans-serif;font-size:1.1rem;font-weight:700;color:#E8ECF4;margin-bottom:2px;">Patient Vitals</p>', unsafe_allow_html=True)
        st.markdown('<p>Enter clinical parameters below</p>', unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
        age    = st.slider("Age", 18, 100, 55)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        bmi    = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.5, step=0.1, format="%.1f")

        st.markdown('<div class="section-header">Cardiac Readings</div>', unsafe_allow_html=True)
        resting_bp     = st.number_input("Resting BP (mmHg)", 70, 220, 125, step=1)
        cholesterol    = st.number_input("Cholesterol (mg/dL)", 80, 700, 220, step=1)
        fasting_bs     = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                       format_func=lambda x: "Yes" if x else "No")
        max_hr         = st.number_input("Max Heart Rate (bpm)", 60, 220, 150, step=1)
        oldpeak        = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, step=0.1, format="%.1f")

        st.markdown('<div class="section-header">ECG & Clinical</div>', unsafe_allow_html=True)
        chest_pain = st.selectbox("Chest Pain Type",
                                   ["Asymptomatic", "Atypical Angina", "Non-anginal Pain", "Typical Angina"])
        resting_ecg= st.selectbox("Resting ECG",
                                   ["Left Ventricular Hypertrophy", "Normal", "ST-T Abnormality"])
        exercise_a = st.radio("Exercise-Induced Angina", ["No", "Yes"], horizontal=True)
        st_slope   = st.selectbox("ST Slope", ["Down", "Flat", "Up"])
        num_vessels= st.slider("Major Vessels (Fluoroscopy)", 0, 4, 1)
        thalassemia= st.selectbox("Thalassemia", ["Fixed Defect", "Normal", "Reversible Defect"])

        st.markdown('<div class="section-header">Lifestyle & Comorbidities</div>', unsafe_allow_html=True)
        smoking    = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol    = st.selectbox("Alcohol Consumption", ["Non-drinker", "Moderate", "Heavy"])
        activity   = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        family_hx  = st.radio("Family History of CVD", ["No", "Yes"], horizontal=True)
        diabetes   = st.radio("Diabetes", ["No", "Yes"], horizontal=True)
        stress     = st.slider("Stress Level (1–10)", 1.0, 10.0, 5.0, step=0.5)

        st.divider()
        submitted = st.button("Run Risk Assessment", use_container_width=True, type="primary")

    return {
        "age": age, "gender": gender, "bmi": float(bmi),
        "resting_bp": float(resting_bp), "cholesterol": float(cholesterol),
        "fasting_blood_sugar": float(fasting_bs), "max_heart_rate": float(max_hr),
        "oldpeak": float(oldpeak), "chest_pain_type": chest_pain,
        "resting_ecg": resting_ecg,
        "exercise_angina": 1 if exercise_a == "Yes" else 0,
        "st_slope": st_slope, "num_major_vessels": num_vessels,
        "thalassemia": thalassemia,
        "smoking_status": smoking, "alcohol_consumption": alcohol,
        "physical_activity": activity,
        "family_history": 1 if family_hx == "Yes" else 0,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "stress_level": float(stress),
        "_submitted": submitted,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    model, scaler, encoder, metrics_data = load_artifacts()
    raw = render_sidebar()
    submitted = raw.pop("_submitted")

    # Header
    st.markdown(WAVEFORM_HTML, unsafe_allow_html=True)
    st.markdown(
        '<h1 style="font-family:\'Space Grotesk\',sans-serif;font-size:2rem;font-weight:700;'
        'color:#E8ECF4;margin:12px 0 4px;">Cordis Sentinel</h1>'
        '<p style="color:#8891A8;font-size:0.9rem;margin-bottom:24px;">'
        'Heart Attack Risk Prediction · Portfolio / Educational Project</p>',
        unsafe_allow_html=True,
    )

    if not submitted:
        st.markdown(
            '<div class="panel" style="text-align:center;padding:48px;">'
            '<p style="font-family:\'Space Grotesk\',sans-serif;font-size:1.1rem;color:#8891A8;">'
            'Configure patient vitals in the sidebar and click <strong style="color:#4FD1C5;">'
            'Run Risk Assessment</strong> to see the prediction.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        # ── Prediction ──────────────────────────────────────────────────────
        X = build_feature_vector(raw, scaler)
        prob = float(model.predict_proba(X)[0, 1])
        label = "High Risk" if prob >= 0.5 else "Low Risk"
        risk_color = CORAL if prob >= 0.5 else GREEN

        # ── Row 1: Gauge + quick vitals ─────────────────────────────────────
        col_gauge, col_vitals = st.columns([1, 1], gap="large")

        with col_gauge:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(prob), use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<p style="text-align:center;font-family:\'Space Grotesk\',sans-serif;'
                f'font-size:1.6rem;font-weight:700;color:{risk_color};margin-top:-12px;">'
                f'{label}</p>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_vitals:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)
            vitals = {
                "Age": f"{raw['age']} yrs",
                "Gender": raw["gender"],
                "BMI": f"{raw['bmi']:.1f}",
                "Resting BP": f"{int(raw['resting_bp'])} mmHg",
                "Cholesterol": f"{int(raw['cholesterol'])} mg/dL",
                "Max HR": f"{int(raw['max_heart_rate'])} bpm",
                "ST Depression": f"{raw['oldpeak']:.1f}",
                "Stress Level": f"{raw['stress_level']:.1f} / 10",
                "Chest Pain": raw["chest_pain_type"],
                "Thalassemia": raw["thalassemia"],
            }
            for k, v in vitals.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:5px 0;border-bottom:1px solid #232B42;">'
                    f'<span style="color:#8891A8;font-size:0.82rem;">{k}</span>'
                    f'<span class="vitals-mono">{v}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 2: Feature importances ───────────────────────────────────────
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(make_importance_chart(model), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            '<p style="font-size:0.75rem;color:#8891A8;margin-top:-8px;">'
            'Positive coefficients (coral) push toward higher risk. '
            'Negative coefficients (green) push toward lower risk. '
            'Top 15 by absolute magnitude shown.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Evaluation Panel (always visible) ──────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:8px;">Model Performance</div>',
                unsafe_allow_html=True)

    m = metrics_data["metrics"]
    metric_cols = st.columns(5)
    labels_map = [("Accuracy", "Accuracy"), ("Precision", "Precision"),
                  ("Recall", "Recall"), ("F1", "F1 Score"), ("ROC_AUC", "ROC AUC")]
    for col, (key, display) in zip(metric_cols, labels_map):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{m[key]:.3f}</div>'
                f'<div class="metric-label">{display}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    cm_col, note_col = st.columns([1.4, 1], gap="large")
    with cm_col:
        st.markdown('<div class="panel" style="margin-top:16px;">', unsafe_allow_html=True)
        st.plotly_chart(make_confusion_matrix(metrics_data["confusion_matrix"]),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with note_col:
        st.markdown(
            '<div class="panel" style="margin-top:16px;">'
            '<div class="section-header">Evaluation Notes</div>'
            f'<p style="font-size:0.83rem;color:#8891A8;line-height:1.7;">'
            f'Model: <span style="color:#E8ECF4;">Logistic Regression</span><br>'
            f'Dataset: <span style="color:#E8ECF4;">7,000 patients</span><br>'
            f'Train/Test split: <span style="color:#E8ECF4;">80 / 20 (stratified)</span><br>'
            f'Class balancing: <span style="color:#E8ECF4;">SMOTE on train set only</span><br>'
            f'Test set size: <span style="color:#E8ECF4;">1,400 samples</span><br><br>'
            f'The ROC AUC of <span style="color:#4FD1C5;">0.861</span> indicates strong '
            f'discriminative ability. Recall of <span style="color:#4FD1C5;">0.740</span> '
            f'means ~74% of true positive cases are correctly flagged.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Footer disclaimer ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer-disclaimer">'
        '<strong style="color:#E8ECF4;">Educational / Portfolio Project — Not a Medical Tool.</strong> '
        'Cordis Sentinel is a machine learning demonstration built for portfolio purposes only. '
        'It is not validated for clinical use, does not constitute medical advice, and should '
        'never be used to make or delay health decisions. If you have concerns about cardiovascular '
        'health, please consult a qualified medical professional.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
