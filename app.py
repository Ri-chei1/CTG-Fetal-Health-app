import json
from pathlib import Path
import pandas as pd
import streamlit as st


ART = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ART / "model.joblib"
COLS_PATH  = ART / "columns.json"

st.set_page_config(page_title="CTG NSP Predictor", page_icon="🩺", layout="wide")
st.title("🩺 CTG Fetal Health Predictor (NSP)")

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_proba" not in st.session_state:
    st.session_state.last_proba = None


if not MODEL_PATH.exists() or not COLS_PATH.exists():
    st.error("Missing files. Put artifacts/model.joblib and artifacts/columns.json next to this app.")
    st.stop()

model = joblib.load(MODEL_PATH)

with open(COLS_PATH, "r") as f:
    columns = json.load(f)

DEFAULTS_PATH = ART / "feature_defaults.json"
TOP_PATH = ART / "top_features.json"

if not DEFAULTS_PATH.exists() or not TOP_PATH.exists():
    st.error("Missing feature defaults. Put artifacts/feature_defaults.json and artifacts/top_features.json next to this app.")
    st.stop()

defaults = json.load(open(DEFAULTS_PATH))
medians = defaults["medians"]          # dict of {col: median}

top_features = json.load(open(TOP_PATH))

# ---- Friendly names (add once here) ----
FRIENDLY = {
    "LB": "Baseline fetal heart rate (bpm)",
    "AC": "Accelerations (count)",
    "FM": "Fetal movements (count)",
    "UC": "Uterine contractions (count)",
    "DL": "Light decelerations (count)",
    "DS": "Severe decelerations (count)",
    "DP": "Prolonged decelerations (count)",
    "ASTV": "Abnormal short-term variability (%)",
    "MSTV": "Mean short-term variability",
    "ALTV": "Abnormal long-term variability (%)",
    "MLTV": "Mean long-term variability",
    "Width": "Histogram width",
    "Min": "Histogram min",
    "Max": "Histogram max",
    "Nmax": "Histogram peaks (Nmax)",
    "Nzeros": "Histogram zeros (Nzeros)",
    "Mode": "Histogram mode",
    "Mean": "Histogram mean",
    "Median": "Histogram median",
    "Variance": "Histogram variance",
    "Tendency": "Histogram tendency",
}
def pretty(col: str) -> str:
    return FRIENDLY.get(col, col)

# ---- Inputs (choose one mode so values don't overwrite) ----
inputs = {col: float(medians.get(col, 0.0)) for col in columns}

mode = st.radio("Input mode", ["Top features (recommended)", "Advanced (all features)"], horizontal=True)

n_per_row = 3  # change to 2 for bigger boxes

if mode == "Top features (recommended)":
    st.subheader("Inputs (top features)")
    st.caption("Only the most important features are shown. All other features use dataset medians.")

    for i in range(0, len(top_features), n_per_row):
        row_cols = st.columns(n_per_row)
        chunk = top_features[i:i + n_per_row]
        for j, col in enumerate(chunk):
            with row_cols[j]:
                inputs[col] = st.number_input(
                    label=pretty(col),
                    value=float(inputs[col]),
                    key=f"top_{col}",
                    format="%.4f"
                )
else:
    st.subheader("Inputs (advanced)")
    st.caption("All features are editable.")

    for i in range(0, len(columns), n_per_row):
        row_cols = st.columns(n_per_row)
        chunk = columns[i:i + n_per_row]
        for j, col in enumerate(chunk):
            with row_cols[j]:
                inputs[col] = st.number_input(
                    label=pretty(col),
                    value=float(inputs[col]),
                    key=f"all_{col}",
                    format="%.4f"
                )

x = pd.DataFrame([inputs], columns=columns)



with st.expander("Show input row"):
    st.dataframe(x)


# If your model uses 0/1/2 labels (NSP-1), keep this:
label_map = {0: "Normal", 1: "Suspect", 2: "Pathologic"}

st.subheader("Prediction")

colA, colB = st.columns([1, 1])
with colA:
    do_predict = st.button("Predict", type="primary")
with colB:
    do_reset = st.button("Reset")

if do_reset:
    st.session_state.last_pred = None
    st.session_state.last_proba = None


if do_predict:
    pred = int(model.predict(x)[0])
    st.session_state.last_pred = pred

    if hasattr(model, "predict_proba"):
        st.session_state.last_proba = model.predict_proba(x)[0]
    else:
        st.session_state.last_proba = None

if st.session_state.last_pred is None:
    st.info("Set inputs, then click **Predict**.")
else:
    pred = st.session_state.last_pred
    st.metric("Predicted NSP", f"{pred} — {label_map.get(pred, 'Unknown')}")

    if st.session_state.last_proba is not None:
        proba = st.session_state.last_proba
        classes = list(getattr(model, "classes_", [0, 1, 2]))

        prob_df = pd.DataFrame({
            "Class": [f"{int(c)} — {label_map.get(int(c), 'Unknown')}" for c in classes],
            "Probability": proba
        }).sort_values("Probability", ascending=False)

        st.subheader("Class probabilities")
        st.bar_chart(prob_df.set_index("Class"))
        st.dataframe(prob_df, hide_index=True)



