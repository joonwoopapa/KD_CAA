
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("Coronary Aneurysm Prediction (XGBoost Model)")

model = joblib.load("xgb_model.pkl")
explainer = joblib.load("shap_explainer.pkl")

features = [
    "initial_echo_LAD_Z", "initial_echo_LMCA_Z", "initial_echo_RCA_Z", "initial_echo_LCx_Z",
    "fever_duration", "Sex", "ALT_before", "HCT_before", "P_before", "CRP_before",
    "TB_before", "CO2_before", "K_before", "Glu_before", "ALP_before"
]

st.write("### Input 15 variables for prediction")
user_input = {}
for feat in features:
    if feat == "Sex":
        user_input[feat] = st.selectbox(feat, [0, 1], format_func=lambda x: "남자(1)" if x == 1 else "여자(0)")
    else:
        user_input[feat] = st.number_input(feat, value=0.0)

X_input = pd.DataFrame([user_input])

if st.button("Predict"):
    pred_prob = model.predict_proba(X_input)[0, 1]
    st.success(f"Coronary Aneurysm Predicted Probability: {pred_prob:.2%}")

    # SHAP 값 계산
    shap_values = explainer(X_input)

    st.subheader("SHAP Waterfall Plot")
    fig1 = shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig1)

    st.subheader("SHAP Force Plot")
    force_plot_html = shap.plots.force(shap_values[0], matplotlib=False).html()
    st.components.v1.html(shap.getjs(), height=0)
    st.components.v1.html(force_plot_html, height=300)
