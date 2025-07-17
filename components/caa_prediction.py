import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt


def show(model, explainer):
    """Display the Coronary Aneurysm prediction page"""
    st.title("Coronary Aneurysm Prediction")
    st.write("*Prediction of coronary aneurysm development using XGBoost model*")
    

    
    col1, col2, col3 = st.columns(3)
    
    user_input = {}
    
    with col1:
        st.markdown("**Laboratory Parameters**")
        st.markdown("""
        <div style='font-size: 0.8rem; color: #64748b; margin-bottom: 1rem; line-height: 1.3;'>
        Results within 3 days prior to the 1st IVIG administration
        </div>
        """, unsafe_allow_html=True)
        user_input["CRP_before"] = st.number_input("C-Reactive Protein (mg/dL)", value=None, placeholder="0.00", format="%.2f")
        user_input["P_before"] = st.number_input("Phosphorus (mg/dL)", value=None, placeholder="0.00", format="%.2f")
        user_input["TB_before"] = st.number_input("Total Bilirubin (mg/dL)", value=None, placeholder="0.00", format="%.2f")
        user_input["ALT_before"] = st.number_input("Alanine Aminotransferase (IU/L)", value=None, placeholder="0.00", format="%.2f")
        user_input["HCT_before"] = st.number_input("Hematocrit (%)", value=None, placeholder="0.00", format="%.2f")
        user_input["CO2_before"] = st.number_input("Carbon Dioxide (mEq/L)", value=None, placeholder="0.00", format="%.2f")
        user_input["K_before"] = st.number_input("Potassium (mEq/L)", value=None, placeholder="0.00", format="%.2f")
        user_input["Glu_before"] = st.number_input("Glucose (mg/dL)", value=None, placeholder="0.00", format="%.2f")
        user_input["ALP_before"] = st.number_input("Alkaline Phosphatase (IU/L)", value=None, placeholder="0.00", format="%.2f")
    
    with col2:
        st.markdown("**Echocardiographic Parameters**")
        st.markdown("""
        <div style='font-size: 0.8rem; color: #64748b; margin-bottom: 1rem; line-height: 1.3;'>
        Initial echocardiography Z-scores calculated using Dallaire and Dahdah nomograms
        </div>
        """, unsafe_allow_html=True)
        user_input["initial_echo_RCA_Z"] = st.number_input("Right Coronary Artery Z-score", value=None, placeholder="0.00", format="%.2f")
        user_input["initial_echo_LMCA_Z"] = st.number_input("Left Main Coronary Artery Z-score", value=None, placeholder="0.00", format="%.2f")
        user_input["initial_echo_LAD_Z"] = st.number_input("Left Anterior Descending Z-score", value=None, placeholder="0.00", format="%.2f")
        user_input["initial_echo_LCx_Z"] = st.number_input("Left Circumflex Z-score", value=None, placeholder="0.00", format="%.2f")
    
    with col3:
        st.markdown("**Clinical Parameters**")
        user_input["fever_duration"] = st.number_input("Fever Duration (days)", value=None, placeholder="0.0", format="%.1f")
        user_input["Sex"] = st.selectbox(
            "Sex", [None, 0, 1], 
            format_func=lambda x: "--- Select ---" if x is None else ("Male" if x == 1 else "Female")
        )
    
    feature_order = [
        "initial_echo_LAD_Z", "initial_echo_LMCA_Z", "initial_echo_RCA_Z", "initial_echo_LCx_Z",
        "fever_duration", "Sex", "ALT_before", "HCT_before", "P_before", "CRP_before",
        "TB_before", "CO2_before", "K_before", "Glu_before", "ALP_before"
    ]
    
    if st.button("Predict Coronary Aneurysm Risk", type="primary"):
        # Validate all fields are filled
        missing_fields = []
        for field, value in user_input.items():
            if value is None:
                missing_fields.append(field)
        
        if missing_fields:
            st.error(f"⚠️ Please fill in all required fields. {len(missing_fields)} field(s) are missing.")
            st.warning("All laboratory parameters, echocardiographic measurements, fever duration, and sex must be provided.")
            st.stop()
        
        if model is not None:
            X_input = pd.DataFrame([user_input])
            X_input = X_input[feature_order]
            pred_prob = model.predict_proba(X_input)[0, 1]
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(
                    label="Coronary Aneurysm Probability",
                    value=f"{pred_prob:.1%}",
                    delta=f"{'High Risk' if pred_prob > 0.5 else 'Low Risk'}"
                )
            
            with col2:
                if pred_prob > 0.7:
                    st.error("High risk: Enhanced monitoring recommended")
                elif pred_prob > 0.3:
                    st.warning("Moderate risk: Careful surveillance required")
                else:
                    st.success("Low risk: Standard monitoring")
            
            if explainer is not None:
                try:
                    shap_values = explainer(X_input)
                    
                    st.write("---")
                    st.subheader("Feature Importance Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Waterfall Plot**")
                        fig1, ax1 = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig1)
                        plt.close(fig1)
                    
                    with col2:
                        st.write("**Feature Importance (Bar Chart)**")
                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        shap.plots.bar(shap_values[0], show=False)
                        st.pyplot(fig3)
                        plt.close(fig3)
                        
                except Exception as e:
                    st.error(f"SHAP analysis error: {str(e)}")
        else:
            st.error("Model not loaded properly.") 