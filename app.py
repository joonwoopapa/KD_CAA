
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(
    page_title="Kawasaki Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# 모델 로딩 함수
@st.cache_resource
def load_models():
    """모든 모델과 explainer를 로딩"""
    models = {}
    explainers = {}
    
    try:
        # Coronary Aneurysm 모델
        models['caa'] = joblib.load("models/xgb_model.pkl")
        explainers['caa'] = joblib.load("models/shap_explainer.pkl")
        st.sidebar.success("✅ CAA 모델 로딩 완료")
    except Exception as e:
        st.sidebar.error(f"❌ CAA 모델 로딩 실패: {str(e)}")
        models['caa'] = None
        explainers['caa'] = None
    
    try:
        # IVIG Resistance 모델
        models['ivig'] = joblib.load("models/rf_model.pkl")
        explainers['ivig'] = joblib.load("models/shap_explainer_rf.pkl")
        st.sidebar.success("✅ IVIG 모델 로딩 완료")
    except Exception as e:
        st.sidebar.error(f"❌ IVIG 모델 로딩 실패: {str(e)}")
        models['ivig'] = None
        explainers['ivig'] = None
    
    return models, explainers

# 관상동맥류 예측 페이지
def coronary_aneurysm_page(model, explainer):
    st.title("🫀 Coronary Aneurysm Prediction")
    st.write("*XGBoost 모델을 사용한 관상동맥류 발생 예측*")
    
    # 필수 필드 안내
    st.markdown("""
        <div style='
            background-color: #f8fafc;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            border-left: 3px solid #3182ce;
        '>
            <p style='margin: 0; color: #4a5568; font-size: 0.9rem;'>
                <span style='color: #3182ce;'>ℹ️</span>
                <strong>All fields are required.</strong> 예측을 위해 모든 측정값을 입력해주세요.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 3개 섹션으로 나누어 입력 필드 배치
    col1, col2, col3 = st.columns(3)
    
    user_input = {}
    
    # Blood Test 섹션
    with col1:
        st.markdown("**🩸 Blood Test**")
        user_input["CRP_before"] = st.number_input("CRP (mg/dL)", value=0.0, format="%.2f")
        user_input["P_before"] = st.number_input("Phosphorus (mg/dL)", value=0.0, format="%.2f")
        user_input["TB_before"] = st.number_input("Total bilirubin (mg/dL)", value=0.0, format="%.2f")
        user_input["ALT_before"] = st.number_input("ALT (U/L)", value=0.0, format="%.2f")
        user_input["HCT_before"] = st.number_input("Hematocrit (%)", value=0.0, format="%.2f")
        user_input["CO2_before"] = st.number_input("CO2 (mEq/L)", value=0.0, format="%.2f")
        user_input["K_before"] = st.number_input("Potassium (mEq/L)", value=0.0, format="%.2f")
        user_input["Glu_before"] = st.number_input("Glucose (mg/dL)", value=0.0, format="%.2f")
        user_input["ALP_before"] = st.number_input("ALP (U/L)", value=0.0, format="%.2f")
    
    # Echocardiography 섹션
    with col2:
        st.markdown("**🫀 Echocardiography**")
        user_input["initial_echo_RCA_Z"] = st.number_input("RCA z score", value=0.0, format="%.2f")
        user_input["initial_echo_LMCA_Z"] = st.number_input("LMCA z score", value=0.0, format="%.2f")
        user_input["initial_echo_LAD_Z"] = st.number_input("LAD z score", value=0.0, format="%.2f")
        user_input["initial_echo_LCx_Z"] = st.number_input("LCx z score", value=0.0, format="%.2f")
    
    # Clinical Symptom 섹션
    with col3:
        st.markdown("**🩺 Clinical Symptom & Demographics**")
        user_input["fever_duration"] = st.number_input("Fever duration (days)", value=0.0, format="%.1f")
        user_input["Sex"] = st.selectbox(
            "Sex", [0, 1], 
            format_func=lambda x: "남자 (Male)" if x == 1 else "여자 (Female)"
        )
    
    X_input = pd.DataFrame([user_input])
    
    if st.button("🔍 Coronary Aneurysm 예측", type="primary"):
        if model is not None:
            pred_prob = model.predict_proba(X_input)[0, 1]
            
            # 결과 표시
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(
                    label="관상동맥류 발생 확률",
                    value=f"{pred_prob:.1%}",
                    delta=f"{'High Risk' if pred_prob > 0.5 else 'Low Risk'}"
                )
            
            with col2:
                if pred_prob > 0.7:
                    st.error("⚠️ 고위험: 정밀 검사 권장")
                elif pred_prob > 0.3:
                    st.warning("⚠️ 중위험: 지속적인 모니터링 필요")
                else:
                    st.success("✅ 저위험: 정상 범위")
            
            # SHAP 분석
            if explainer is not None:
                try:
                    shap_values = explainer(X_input)
                    
                    st.write("---")
                    st.subheader("📊 SHAP 분석 결과")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Waterfall Plot**")
                        fig1 = shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig1)
                    
                    with col2:
                        st.write("**Force Plot**")
                        force_plot_html = shap.plots.force(shap_values[0], matplotlib=False).html()
                        st.components.v1.html(shap.getjs(), height=0)
                        st.components.v1.html(force_plot_html, height=300)
                        
                except Exception as e:
                    st.error(f"SHAP 분석 중 오류: {str(e)}")
        else:
            st.error("모델이 로딩되지 않았습니다.")

# IVIG 저항성 예측 페이지
def ivig_resistance_page(model, explainer):
    st.title("💉 IVIG Resistance Prediction")
    st.write("*RandomForest 모델을 사용한 IVIG 저항성 예측*")
    
    # 필수 필드 안내
    st.markdown("""
        <div style='
            background-color: #f8fafc;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            border-left: 3px solid #3182ce;
        '>
            <p style='margin: 0; color: #4a5568; font-size: 0.9rem;'>
                <span style='color: #3182ce;'>ℹ️</span>
                <strong>All fields are required.</strong> 예측을 위해 모든 측정값을 입력해주세요.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 3개 섹션으로 나누어 입력 필드 배치
    col1, col2, col3 = st.columns(3)
    
    user_input = {}
    
    # Blood Test 섹션
    with col1:
        st.markdown("**🩸 Blood Test**")
        user_input["Lympho_before"] = st.number_input("Lymphocyte (%)", value=0.0, format="%.2f")
        user_input["Seg_before"] = st.number_input("Neutrophil (%)", value=0.0, format="%.2f")
        user_input["PLT_before"] = st.number_input("Platelet count (10³/ml)", value=0.0, format="%.2f")
        user_input["Chol_before"] = st.number_input("Cholesterol (mg/dL)", value=0.0, format="%.2f")
        user_input["CRP_before"] = st.number_input("CRP (mg/dL)", value=0.0, format="%.2f")
        user_input["TB_before"] = st.number_input("Total bilirubin (mg/dL)", value=0.0, format="%.2f")
        user_input["P_before"] = st.number_input("Phosphorus (mg/dL)", value=0.0, format="%.2f")
        user_input["ANC_before"] = st.number_input("Absolute Neutrophil count (10⁹/L)", value=0.0, format="%.2f")
        user_input["Ca_before"] = st.number_input("Calcium (mg/dL)", value=0.0, format="%.2f")
        user_input["AST_before"] = st.number_input("AST (U/L)", value=0.0, format="%.2f")
        user_input["PCT_before"] = st.number_input("Procalcitonin (ng/mL)", value=0.0, format="%.2f")
        user_input["CO2_before"] = st.number_input("CO2 (mEq/L)", value=0.0, format="%.2f")
        user_input["MPV_before"] = st.number_input("Mean Platelet Volume (fL)", value=0.0, format="%.2f")
    
    # Echocardiography 섹션  
    with col2:
        st.markdown("**🫀 Echocardiography**")
        user_input["initial_echo_LAD_Z"] = st.number_input("LAD z score", value=0.0, format="%.2f")
    
    # 추가 정보 섹션
    with col3:
        st.markdown("**📊 Additional Information**")
        st.markdown("""
            <div style='
                background-color: #f0f9ff;
                padding: 1rem;
                border-radius: 6px;
                border-left: 3px solid #0ea5e9;
                margin-top: 1rem;
            '>
                <p style='margin: 0; color: #0c4a6e; font-size: 0.85rem; line-height: 1.4;'>
                    <strong>📝 Note:</strong><br/>
                    이 모델은 14개의 임상 변수를 사용하여 IVIG 저항성을 예측합니다. 
                    대부분의 변수는 혈액검사 결과이며, LAD z-score는 심초음파 검사 결과입니다.
                </p>
            </div>
        """)
    
    X_input = pd.DataFrame([user_input])
    
    if st.button("🔍 IVIG Resistance 예측", type="primary"):
        if model is not None:
            pred_prob = model.predict_proba(X_input)[0, 1]
            
            # 결과 표시
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(
                    label="IVIG 저항성 확률",
                    value=f"{pred_prob:.1%}",
                    delta=f"{'Resistant' if pred_prob > 0.5 else 'Responsive'}"
                )
            
            with col2:
                if pred_prob > 0.7:
                    st.error("⚠️ 고저항성: 대체 치료법 고려")
                elif pred_prob > 0.3:
                    st.warning("⚠️ 중저항성: 신중한 모니터링")
                else:
                    st.success("✅ 저저항성: IVIG 치료 효과적")
            
            # SHAP 분석
            if explainer is not None:
                try:
                    shap_values = explainer(X_input)
                    
                    st.write("---")
                    st.subheader("📊 SHAP 분석 결과")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Waterfall Plot**")
                        fig1 = shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig1)
                    
                    with col2:
                        st.write("**Force Plot**")
                        force_plot_html = shap.plots.force(shap_values[0], matplotlib=False).html()
                        st.components.v1.html(shap.getjs(), height=0)
                        st.components.v1.html(force_plot_html, height=300)
                        
                except Exception as e:
                    st.error(f"SHAP 분석 중 오류: {str(e)}")
        else:
            st.error("모델이 로딩되지 않았습니다.")

# 홈페이지
def home_page():
    st.title("🏥 Kawasaki Disease Prediction System")
    st.write("**가와사키병 예측 시스템에 오신 것을 환영합니다!**")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🫀 Coronary Aneurysm Prediction")
        st.write("**XGBoost 모델**을 사용하여 관상동맥류 발생 가능성을 예측합니다.")
        st.write("- **입력**: 15개 임상 변수")
        st.write("- **출력**: 관상동맥류 발생 확률")
        st.write("- **분석**: SHAP을 통한 설명 가능한 AI")
        
        if st.button("🫀 관상동맥류 예측 시작", key="caa_start", type="primary"):
            st.session_state.page = "caa"
            st.rerun()
    
    with col2:
        st.subheader("💉 IVIG Resistance Prediction")
        st.write("**RandomForest 모델**을 사용하여 IVIG 저항성을 예측합니다.")
        st.write("- **입력**: 14개 임상 변수")
        st.write("- **출력**: IVIG 저항성 확률")
        st.write("- **분석**: SHAP을 통한 설명 가능한 AI")
        
        if st.button("💉 IVIG 저항성 예측 시작", key="ivig_start", type="primary"):
            st.session_state.page = "ivig"
            st.rerun()
    
    st.write("---")
    
    # 추가 정보
    with st.expander("ℹ️ 시스템 정보"):
        st.write("""
        **개발 목적**: 가와사키병 환자의 임상 결과 예측을 통한 치료 방향 결정 지원
        
        **주의사항**: 
        - 이 시스템은 의료진의 판단을 보조하는 도구입니다
        - 최종 진단과 치료 결정은 반드시 의료진이 내려야 합니다
        - 예측 결과는 참고 자료로만 활용하세요
        
        **기술 스택**: Streamlit, XGBoost, RandomForest, SHAP
        """)

# 메인 애플리케이션
def main():
    # 사이드바 설정
    st.sidebar.title("🏥 Navigation")
    
    # 모델 로딩
    models, explainers = load_models()
    
    # 세션 상태 초기화
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # 사이드바 네비게이션
    if st.sidebar.button("🏠 홈페이지", key="nav_home"):
        st.session_state.page = "home"
        st.rerun()
    
    if st.sidebar.button("🫀 관상동맥류 예측", key="nav_caa"):
        st.session_state.page = "caa"
        st.rerun()
    
    if st.sidebar.button("💉 IVIG 저항성 예측", key="nav_ivig"):
        st.session_state.page = "ivig"
        st.rerun()
    
    st.sidebar.write("---")
    st.sidebar.write("**현재 페이지**")
    if st.session_state.page == "home":
        st.sidebar.info("🏠 홈페이지")
    elif st.session_state.page == "caa":
        st.sidebar.info("🫀 관상동맥류 예측")
    elif st.session_state.page == "ivig":
        st.sidebar.info("💉 IVIG 저항성 예측")
    
    # 페이지 라우팅
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "caa":
        coronary_aneurysm_page(models['caa'], explainers['caa'])
    elif st.session_state.page == "ivig":
        ivig_resistance_page(models['ivig'], explainers['ivig'])

if __name__ == "__main__":
    main()
