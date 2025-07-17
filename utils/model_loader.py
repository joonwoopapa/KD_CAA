import streamlit as st
import joblib


@st.cache_resource
def load_models():
    """Load all models and explainers"""
    models = {}
    explainers = {}
    
    try:
        models['caa'] = joblib.load("models/xgb_model.pkl")
        explainers['caa'] = joblib.load("models/shap_explainer.pkl")
        st.sidebar.success("CAA model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"CAA model loading failed: {str(e)}")
        models['caa'] = None
        explainers['caa'] = None
    
    try:
        models['ivig'] = joblib.load("models/rf_model.pkl")
        explainers['ivig'] = joblib.load("models/shap_explainer_rf.pkl")
        st.sidebar.success("IVIG model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"IVIG model loading failed: {str(e)}")
        models['ivig'] = None
        explainers['ivig'] = None
    
    return models, explainers 