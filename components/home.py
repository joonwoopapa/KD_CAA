import streamlit as st


def show():
    """Display the home page"""
    st.title("Kawasaki Disease Prediction System")
    st.write("**Academic Research Platform for Kawasaki Disease Clinical Outcome Prediction**")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Coronary Aneurysm Prediction")
        st.write("Predicts the probability of coronary aneurysm development using **Cat Boost model**.")
        st.write("• **Input:** 16 clinical variables")
        st.write("• **Output:** Coronary aneurysm probability")
        st.write("• **Analysis:** Explainable AI through SHAP")
        
        if st.button("Start CAA Prediction", key="caa_start", type="primary"):
            st.session_state.page = "caa"
            st.rerun()
    
    with col2:
        st.subheader("IVIG Resistance Prediction")
        st.write("Predicts IVIG resistance probability using **RandomForest model**.")
        st.write("• **Input:** 13 clinical variables")
        st.write("• **Output:** IVIG resistance probability")
        st.write("• **Analysis:** Explainable AI through SHAP")
        
        if st.button("Start IVIG Prediction", key="ivig_start", type="primary"):
            st.session_state.page = "ivig"
            st.rerun()
    
    st.write("---")
    
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
