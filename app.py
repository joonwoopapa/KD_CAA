import streamlit as st
import pickle
from components import home, caa_prediction, ivig_prediction

st.set_page_config(
    page_title="Kawasaki Disease Prediction System",
    page_icon="📊",
    layout="wide"
)

def load_models():
    """Load all models and explainers"""
    models = {}
    explainers = {}
    
    try:
        # CAA 모델
        with open('models/agb_model.pkl', 'rb') as f:
            models['caa'] = pickle.load(f)
        with open('models/agb_shap.pkl', 'rb') as f:
            explainers['caa'] = pickle.load(f)
        
        # IVIG 모델
        with open('models/ivig_model.pkl', 'rb') as f:
            models['ivig'] = pickle.load(f)
        with open('models/ivig_shap.pkl', 'rb') as f:
            explainers['ivig'] = pickle.load(f)
            
    except Exception as e:
        st.error(f"❌ 모델 로딩 실패: {str(e)}")
        return None, None
    
    return models, explainers

def main():
    st.sidebar.title("Navigation")
    
    # 모델 로드
    models, explainers = load_models()
    
    if models is None or explainers is None:
        st.error("⚠️ 모델을 로드할 수 없습니다.")
        st.stop()
    
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
    
    # 페이지 렌더링
    if st.session_state.page == "home":
        home.show()
    elif st.session_state.page == "caa":
        caa_prediction.show(models['caa'], explainers['caa'])
    elif st.session_state.page == "ivig":
        ivig_prediction.show(models['ivig'], explainers['ivig'])

if __name__ == "__main__":
    main()
