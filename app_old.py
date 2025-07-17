
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Kawasaki Disease Prediction System",
    page_icon="📊",
    layout="wide"
)

# Model loading function
@st.cache_resource
def load_models():
    """Load all models and explainers"""
    models = {}
    explainers = {}
    
    try:
        # Coronary Aneurysm model
        models['caa'] = joblib.load("models/xgb_model.pkl")
        explainers['caa'] = joblib.load("models/shap_explainer.pkl")
        st.sidebar.success("CAA model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"CAA model loading failed: {str(e)}")
        models['caa'] = None
        explainers['caa'] = None
    
    try:
        # IVIG Resistance model
        models['ivig'] = joblib.load("models/rf_model.pkl")
        explainers['ivig'] = joblib.load("models/shap_explainer_rf.pkl")
        st.sidebar.success("IVIG model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"IVIG model loading failed: {str(e)}")
        models['ivig'] = None
        explainers['ivig'] = None
    
    return models, explainers

# Coronary aneurysm prediction page
def coronary_aneurysm_page(model, explainer):
    st.title("Coronary Aneurysm Prediction")
    st.write("*Prediction of coronary aneurysm development using XGBoost model*")
    
    # Required fields notice
    st.markdown("""
        <div style='
            background-color: #f8fafc;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin-bottom: 1.5rem;
            border-left: 3px solid #64748b;
        '>
            <p style='margin: 0; color: #475569; font-size: 0.9rem;'>
                <strong>Note:</strong> All clinical parameters are required for accurate prediction.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input fields organized in 3 sections
    col1, col2, col3 = st.columns(3)
    
    user_input = {}
    
    # Blood Test section
    with col1:
        st.markdown("**Laboratory Parameters**")
        user_input["CRP_before"] = st.number_input("C-Reactive Protein", value=0.0, format="%.2f", help="mg/dL")
        user_input["P_before"] = st.number_input("Phosphorus", value=0.0, format="%.2f", help="mg/dL")
        user_input["TB_before"] = st.number_input("Total Bilirubin", value=0.0, format="%.2f", help="mg/dL")
        user_input["ALT_before"] = st.number_input("Alanine Aminotransferase", value=0.0, format="%.2f", help="IU/L")
        user_input["HCT_before"] = st.number_input("Hematocrit", value=0.0, format="%.2f", help="Percentage (%)")
        user_input["CO2_before"] = st.number_input("Carbon Dioxide", value=0.0, format="%.2f", help="mEq/L")
        user_input["K_before"] = st.number_input("Potassium", value=0.0, format="%.2f", help="mEq/L")
        user_input["Glu_before"] = st.number_input("Glucose", value=0.0, format="%.2f", help="mg/dL")
        user_input["ALP_before"] = st.number_input("Alkaline Phosphatase", value=0.0, format="%.2f", help="IU/L")
    
    # Echocardiography section
    with col2:
        st.markdown("**Echocardiographic Parameters**")
        st.markdown("""
        <div style='font-size: 0.8rem; color: #64748b; margin-bottom: 1rem; line-height: 1.3;'>
        Initial echocardiography Z-scores calculated using Dallaire and Dahdah nomograms
        </div>
        """, unsafe_allow_html=True)
        user_input["initial_echo_RCA_Z"] = st.number_input("Right Coronary Artery Z-score", value=0.0, format="%.2f")
        user_input["initial_echo_LMCA_Z"] = st.number_input("Left Main Coronary Artery Z-score", value=0.0, format="%.2f")
        user_input["initial_echo_LAD_Z"] = st.number_input("Left Anterior Descending Z-score", value=0.0, format="%.2f")
        user_input["initial_echo_LCx_Z"] = st.number_input("Left Circumflex Z-score", value=0.0, format="%.2f")
    
    # Clinical parameters section
    with col3:
        st.markdown("**Clinical Parameters**")
        user_input["fever_duration"] = st.number_input("Fever Duration (days)", value=0.0, format="%.1f")
        user_input["Sex"] = st.selectbox(
            "Sex", [0, 1], 
            format_func=lambda x: "Male" if x == 1 else "Female"
        )
    
    # Reorder features according to model training order
    feature_order = [
        "initial_echo_LAD_Z", "initial_echo_LMCA_Z", "initial_echo_RCA_Z", "initial_echo_LCx_Z",
        "fever_duration", "Sex", "ALT_before", "HCT_before", "P_before", "CRP_before",
        "TB_before", "CO2_before", "K_before", "Glu_before", "ALP_before"
    ]
    
    X_input = pd.DataFrame([user_input])
    X_input = X_input[feature_order]
    
    if st.button("Predict Coronary Aneurysm Risk", type="primary"):
        if model is not None:
            pred_prob = model.predict_proba(X_input)[0, 1]
            
            # Display results
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
            
            # SHAP analysis
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

# IVIG resistance prediction page
def ivig_resistance_page(model, explainer):
    st.title("IVIG Resistance Prediction")
    st.write("*Prediction of IVIG resistance using RandomForest model*")
    
    # Required fields notice
    st.markdown("""
        <div style='
            background-color: #f8fafc;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin-bottom: 1.5rem;
            border-left: 3px solid #64748b;
        '>
            <p style='margin: 0; color: #475569; font-size: 0.9rem;'>
                <strong>Note:</strong> All clinical parameters are required for accurate prediction.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input fields organized in 3 sections
    col1, col2, col3 = st.columns(3)
    
    user_input = {}
    
    # Blood test section
    with col1:
        st.markdown("**Laboratory Parameters**")
        user_input["Lympho_before"] = st.number_input("Lymphocyte", value=0.0, format="%.2f", help="Percentage (%)")
        user_input["Seg_before"] = st.number_input("Neutrophil", value=0.0, format="%.2f", help="Percentage (%)")
        user_input["PLT_before"] = st.number_input("Platelet Count", value=0.0, format="%.2f", help="10³/μL")
        user_input["Chol_before"] = st.number_input("Total Cholesterol", value=0.0, format="%.2f", help="mg/dL")
        user_input["CRP_before"] = st.number_input("C-Reactive Protein", value=0.0, format="%.2f", help="mg/dL")
        user_input["TB_before"] = st.number_input("Total Bilirubin", value=0.0, format="%.2f", help="mg/dL")
        user_input["P_before"] = st.number_input("Phosphorus", value=0.0, format="%.2f", help="mg/dL")
        user_input["ANC_before"] = st.number_input("Absolute Neutrophil Count", value=0.0, format="%.2f", help="10⁹/L")
        user_input["Ca_before"] = st.number_input("Calcium", value=0.0, format="%.2f", help="mg/dL")
        user_input["AST_before"] = st.number_input("Aspartate Aminotransferase", value=0.0, format="%.2f", help="IU/L")
        user_input["PCT_before"] = st.number_input("Procalcitonin", value=0.0, format="%.2f", help="ng/mL")
        user_input["CO2_before"] = st.number_input("Carbon Dioxide", value=0.0, format="%.2f", help="mEq/L")
        user_input["MPV_before"] = st.number_input("Mean Platelet Volume", value=0.0, format="%.2f", help="fL")
    
    # Echocardiography section  
    with col2:
        st.markdown("**Echocardiographic Parameters**")
        st.markdown("""
        <div style='font-size: 0.8rem; color: #64748b; margin-bottom: 1rem; line-height: 1.3;'>
        Initial echocardiography Z-scores calculated using Dallaire and Dahdah nomograms
        </div>
        """, unsafe_allow_html=True)
        user_input["initial_echo_LAD_Z"] = st.number_input("Left Anterior Descending Z-score", value=0.0, format="%.2f")
    
    # Model information section
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
                    This model utilizes 14 clinical variables to predict IVIG resistance. 
                    Most variables are laboratory parameters with one echocardiographic measurement.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Reorder features according to model training order
    ivig_feature_order = [
        "PLT_before", "Lympho_before", "Seg_before", "Chol_before", "CRP_before", "P_before", 
        "TB_before", "Ca_before", "AST_before", "PCT_before", "initial_echo_LAD_Z", 
        "ANC_before", "CO2_before", "MPV_before"
    ]
    
    X_input = pd.DataFrame([user_input])
    X_input = X_input[ivig_feature_order]
    
    if st.button("Predict IVIG Resistance", type="primary"):
        if model is not None:
            pred_prob = model.predict_proba(X_input)[0, 1]
            
            # Display results
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
            
            # SHAP analysis
            if explainer is not None:
                try:
                    shap_values = explainer(X_input)
                    
                    st.write("---")
                    st.subheader("Feature Importance Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Waterfall Plot**")
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        # Use IVIG resistance (positive class) SHAP values
                        if len(shap_values[0].shape) > 1:
                            shap.plots.waterfall(shap_values[0, :, 1], show=False)  # multi-output model
                        else:
                            shap.plots.waterfall(shap_values[0], show=False)  # single output model
                        st.pyplot(fig2)
                        plt.close(fig2)
                    
                    with col2:
                        st.write("**Feature Importance (Bar Chart)**")
                        fig4, ax4 = plt.subplots(figsize=(10, 6))
                        if len(shap_values[0].shape) > 1:
                            shap.plots.bar(shap_values[0, :, 1], show=False)
                        else:
                            shap.plots.bar(shap_values[0], show=False)
                        st.pyplot(fig4)
                        plt.close(fig4)
                        
                except Exception as e:
                    st.error(f"SHAP analysis error: {str(e)}")
        else:
            st.error("Model not loaded properly.")

# Home page
def home_page():
    st.title("Kawasaki Disease Prediction System")
    st.write("**Academic Research Platform for Kawasaki Disease Clinical Outcome Prediction**")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Coronary Aneurysm Prediction")
        st.write("Predicts the probability of coronary aneurysm development using **XGBoost model**.")
        st.write("• **Input:** 15 clinical variables")
        st.write("• **Output:** Coronary aneurysm probability")
        st.write("• **Analysis:** Explainable AI through SHAP")
        
        if st.button("Start CAA Prediction", key="caa_start", type="primary"):
            st.session_state.page = "caa"
            st.rerun()
    
    with col2:
        st.subheader("IVIG Resistance Prediction")
        st.write("Predicts IVIG resistance probability using **RandomForest model**.")
        st.write("• **Input:** 14 clinical variables")
        st.write("• **Output:** IVIG resistance probability")
        st.write("• **Analysis:** Explainable AI through SHAP")
        
        if st.button("Start IVIG Prediction", key="ivig_start", type="primary"):
            st.session_state.page = "ivig"
            st.rerun()
    
    st.write("---")
    
    # Additional information
    with st.expander("System Information"):
        st.write("""
        **Purpose:** Clinical decision support system for predicting Kawasaki disease outcomes
        
        **Disclaimer:** 
        • This system is designed to assist healthcare professionals in clinical decision-making
        • Final diagnosis and treatment decisions must be made by qualified medical personnel
        • Prediction results should be used as supplementary information only
        
        **Technical Stack:** Streamlit, XGBoost, RandomForest, SHAP
        
        **Reference:** Model performance and validation metrics are available in the accompanying research documentation.
        """)

# Main application
def main():
    # Sidebar configuration
    st.sidebar.title("Navigation")
    
    # Load models
    models, explainers = load_models()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Sidebar navigation
    if st.sidebar.button("Home", key="nav_home"):
        st.session_state.page = "home"
        st.rerun()
    
    if st.sidebar.button("Coronary Aneurysm", key="nav_caa"):
        st.session_state.page = "caa"
        st.rerun()
    
    if st.sidebar.button("IVIG Resistance", key="nav_ivig"):
        st.session_state.page = "ivig"
        st.rerun()
    
    st.sidebar.write("---")
    st.sidebar.write("**Current Page**")
    if st.session_state.page == "home":
        st.sidebar.info("Home")
    elif st.session_state.page == "caa":
        st.sidebar.info("Coronary Aneurysm Prediction")
    elif st.session_state.page == "ivig":
        st.sidebar.info("IVIG Resistance Prediction")
    
    # Page routing
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "caa":
        coronary_aneurysm_page(models['caa'], explainers['caa'])
    elif st.session_state.page == "ivig":
        ivig_resistance_page(models['ivig'], explainers['ivig'])

if __name__ == "__main__":
    main()
