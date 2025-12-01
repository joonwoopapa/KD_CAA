import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt


def show(model, explainer):
    """Display the IVIG Resistance prediction page"""
    st.title("IVIG Resistance Prediction")
    st.write("*Prediction of IVIG resistance using RandomForest model*")
    

    
    col1, col2, col3 = st.columns(3)
    
    user_input = {}
    
    with col1:
        st.markdown("**Laboratory Parameters**")
        st.markdown("""
        <div style='font-size: 0.8rem; color: #64748b; margin-bottom: 1rem; line-height: 1.3;'>
        Results within 3 days prior to the 1st IVIG administration
        </div>
        """, unsafe_allow_html=True)
        user_input["Lympho_before"] = st.number_input("Lymphocyte (%)", value=None, placeholder="0.00", format="%.2f")
        user_input["Seg_before"] = st.number_input("Neutrophil (%)", value=None, placeholder="0.00", format="%.2f")
        user_input["PLT_before"] = st.number_input("Platelet Count (10³/μL)", value=None, placeholder="0.00", format="%.2f")
        user_input["Chol_before"] = st.number_input("Total Cholesterol (mg/dL)", value=None, placeholder="0.00", format="%.2f")
        user_input["CRP_before"] = st.number_input("C-Reactive Protein (mg/dL)", value=None, placeholder="0.00", format="%.2f")
        user_input["TB_before"] = st.number_input("Total Bilirubin (mg/dL)", value=None, placeholder="0.00", format="%.2f")
        user_input["P_before"] = st.number_input("Phosphorus (mg/dL)", value=None, placeholder="0.00", format="%.2f")
        user_input["ANC_before"] = st.number_input("Absolute Neutrophil Count (10⁹/L)", value=None, placeholder="0.00", format="%.2f")
        user_input["Ca_before"] = st.number_input("Calcium (mg/dL)", value=None, placeholder="0.00", format="%.2f")
        user_input["AST_before"] = st.number_input("Aspartate Aminotransferase (IU/L)", value=None, placeholder="0.00", format="%.2f")
        user_input["PCT_before"] = st.number_input("Plateletcrit (%)", value=None, placeholder="0.00", format="%.2f")
        user_input["CO2_before"] = st.number_input("Carbon Dioxide (mEq/L)", value=None, placeholder="0.00", format="%.2f")

    
    with col2:
        st.markdown("**Echocardiographic Parameters**")
        st.markdown("""
        <div style='font-size: 0.8rem; color: #64748b; margin-bottom: 1rem; line-height: 1.3;'>
        Initial echocardiography Z-scores calculated using Dallaire and Dahdah nomograms
        </div>
        """, unsafe_allow_html=True)
        user_input["initial_echo_LAD_Z"] = st.number_input("Left Anterior Descending Z-score", value=None, placeholder="0.00", format="%.2f")
    
    with col3:
        st.markdown("**Model Information**")
        st.markdown("""
            <div style='
                background-color: #f1f5f9;
                padding: 1rem;
                border-radius: 4px;
                border-left: 3px solid #64748b;
                margin-top: 1rem;
            '>
                <p style='margin: 0; color: #475569; font-size: 0.85rem; line-height: 1.4;'>
                    <strong>Clinical Variables:</strong><br/>
                    This model utilizes 13 clinical variables to predict IVIG resistance. 
                    Most variables are laboratory parameters with one echocardiographic measurement.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    ivig_feature_order = [
        "PLT_before", "Lympho_before", "Seg_before", "Chol_before", "CRP_before", "P_before", 
        "TB_before", "Ca_before", "AST_before", "PCT_before", "initial_echo_LAD_Z", 
        "ANC_before", "CO2_before"
    ]
    
    # Feature names mapping for SHAP display
    feature_names_map = {
        "PLT_before": "Platelet Count (10³/ml)",
        "Lympho_before": "Lymphocyte (%)",
        "Seg_before": "Neutrophil (%)",
        "Chol_before": "Cholesterol (mg/dL)",
        "CRP_before": "C-Reactive Protein (mg/dL)",
        "P_before": "Phosphorus (mg/dL)",
        "TB_before": "Total Bilirubin (mg/dL)",
        "Ca_before": "Calcium (mg/dL)",
        "AST_before": "Aspartate Aminotransferase (IU/L)",
        "PCT_before": "Plateletcrit (%)",
        "initial_echo_LAD_Z": "LAD Z Score",
        "ANC_before": "Absolute Neutrophil Count (10⁹/L)",
        "CO2_before": "Carbon Dioxide (mEq/L)",
    }
    
    if st.button("Predict IVIG Resistance", type="primary"):
        # Validate all fields are filled
        missing_fields = []
        for field, value in user_input.items():
            if value is None:
                missing_fields.append(field)
        
        if missing_fields:
            st.error(f"⚠️ Please fill in all required fields. {len(missing_fields)} field(s) are missing.")
            st.warning("All laboratory parameters and echocardiographic measurement must be provided.")
            st.stop()
        
        if model is not None:
            X_input = pd.DataFrame([user_input])
            X_input = X_input[ivig_feature_order]
            pred_prob = model.predict_proba(X_input)[0, 1]
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(
                    label="IVIG Resistance Probability",
                    value=f"{pred_prob:.1%}",
                    delta=f"{'Resistant' if pred_prob > 0.5 else 'Responsive'}"
                )
            
            with col2:
                if pred_prob > 0.7:
                    st.error("High resistance likelihood: Consider alternative therapy")
                elif pred_prob > 0.3:
                    st.warning("Moderate resistance risk: Enhanced monitoring advised")
                else:
                    st.success("Low resistance probability: IVIG likely effective")
            
            if explainer is not None:
                try:
                    shap_values = explainer(X_input)
                    
                    # Update feature names for better display
                    if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
                        enhanced_names = []
                        for i, name in enumerate(shap_values.feature_names):
                            readable_name = feature_names_map.get(name, name)
                            enhanced_names.append(readable_name)
                        shap_values.feature_names = enhanced_names
                    
                    st.write("---")
                    st.subheader("Feature Importance Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Waterfall Plot**")
                        fig2, ax2 = plt.subplots(figsize=(12, 8))
                        plt.rcParams.update({'font.size': 10})
                        if len(shap_values[0].shape) > 1:
                            shap.plots.waterfall(shap_values[0, :, 1], max_display=14, show=False)
                        else:
                            shap.plots.waterfall(shap_values[0], max_display=14, show=False)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)
                    
                    with col2:
                        st.write("**Feature Importance (Bar Chart)**")
                        fig4, ax4 = plt.subplots(figsize=(12, 8))
                        plt.rcParams.update({'font.size': 10})
                        if len(shap_values[0].shape) > 1:
                            shap.plots.bar(shap_values[0, :, 1], max_display=14, show=False)
                        else:
                            shap.plots.bar(shap_values[0], max_display=14, show=False)
                        plt.tight_layout()
                        st.pyplot(fig4)
                        plt.close(fig4)
                        
                except Exception as e:
                    st.error(f"SHAP analysis error: {str(e)}")
        else:
            st.error("Model not loaded properly.") 
