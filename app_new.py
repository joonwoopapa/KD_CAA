import streamlit as st

from utils.model_loader import load_models
from pages import home, caa_prediction, ivig_prediction

st.set_page_config(
    page_title="Kawasaki Disease Prediction System",
    page_icon="📊",
    layout="wide"
)


def main():
    st.sidebar.title("Navigation")
    
    models, explainers = load_models()
    
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
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
    
    if st.session_state.page == "home":
        home.show()
    elif st.session_state.page == "caa":
        caa_prediction.show(models['caa'], explainers['caa'])
    elif st.session_state.page == "ivig":
        ivig_prediction.show(models['ivig'], explainers['ivig'])


if __name__ == "__main__":
    main() 