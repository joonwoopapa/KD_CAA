# 🏥 Kawasaki Disease Prediction System

통합된 가와사키병 예측 시스템으로, 두 가지 주요 임상 결과를 예측할 수 있습니다:
- **관상동맥류 발생 예측** (CATBoost 모델)
- **IVIG 저항성 예측** (RandomForest 모델)

## 🌟 주요 기능

### 🫀 Coronary Aneurysm Prediction
- **모델**: CATBoost
- **입력**: 16개 임상 변수 (심초음파 Z-score, 발열기간, 성별, 혈액검사 결과)
- **출력**: 관상동맥류 발생 확률
- **설명**: SHAP을 통한 예측 근거 제공 (모든 15개 변수 개별 표시)

### 💉 IVIG Resistance Prediction  
- **모델**: RandomForest
- **입력**: 13개 임상 변수 (혈소판수치, 염증지표, 간기능검사 등)
- **출력**: IVIG 저항성 확률
- **설명**: SHAP을 통한 예측 근거 제공 (모든 14개 변수 개별 표시)

### 📊 SHAP 분석 (최신 개선 버전)
- **Waterfall Plot**: 각 변수의 기여도를 단위와 함께 명확하게 시각화
- **Bar Chart**: 변수 중요도를 가독성 높은 형태로 표시
- **변수명 개선**: 모든 변수에 단위 포함 (예: "C-Reactive Protein (mg/dL)")
- **성별 표시**: 0/1 대신 "Male"/"Female"로 직관적 표시
- **완전한 변수 표시**: "6 other features" 없이 모든 변수를 개별적으로 표시
- **표준화된 스타일**: 일관된 그래프 크기 (12x8)와 폰트 크기 (10pt)

## 🚀 빠른 시작

### Docker를 사용한 실행 (권장)

```bash
# 1. 리포지토리 클론
git clone https://github.com/joonwoopapa/KD_CAA.git
cd KD_CAA

# 2. Docker 이미지 빌드
docker build -t kd-caa-app .

# 3. 컨테이너 실행
docker run -p 8501:8501 kd-caa-app

# 4. 웹 브라우저에서 접속
# http://localhost:8501
```

### Docker Compose 사용

```bash
# 백그라운드에서 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

## 📁 프로젝트 구조

```
KD_CAA/
├── app.py                      # 메인 Streamlit 애플리케이션
├── components/                 # 모듈화된 컴포넌트
│   ├── __init__.py
│   ├── home.py                # 홈페이지 컴포넌트
│   ├── caa_prediction.py      # 관상동맥류 예측 컴포넌트
│   └── ivig_prediction.py     # IVIG 저항성 예측 컴포넌트
├── utils/                      # 유틸리티 모듈
│   ├── __init__.py
│   └── model_loader.py        # 모델 로딩 유틸리티
├── models/                     # ML 모델 파일들
│   ├── xgb_model.pkl          # XGBoost 관상동맥류 예측 모델
│   ├── shap_explainer.pkl     # CAA SHAP explainer
│   ├── rf_model.pkl           # RandomForest IVIG 저항성 모델
│   └── shap_explainer_rf.pkl  # IVIG SHAP explainer
├── docs/                       # API 문서
│   └── API.md                 # API 참조 문서
├── requirements.txt            # Python 의존성
├── runtime.txt                # Python 버전 (3.12)
├── streamlit_config.toml      # Streamlit 설정
├── Dockerfile                 # Docker 이미지 빌드 설정
├── docker-compose.yml         # Docker Compose 설정
├── render.yaml                # Render.com 배포 설정
└── README.md                  # 이 파일
```

## 🔧 기술 스택

- **Frontend**: Streamlit (모듈화된 컴포넌트 구조)
- **ML Models**: XGBoost, RandomForest
- **Explainability**: SHAP (최신 시각화 기능)
- **Deployment**: Docker, Render.com
- **Language**: Python 3.12

## 📦 의존성

주요 라이브러리:
- `streamlit>=1.28.0` - 웹 애플리케이션 프레임워크
- `xgboost>=2.0.0` - 그래디언트 부스팅 모델
- `scikit-learn>=1.2.0` - 랜덤포레스트 모델
- `shap>=0.43.0` - 모델 해석 및 시각화
- `pandas>=2.0.0` - 데이터 처리
- `matplotlib>=3.7.0` - 고품질 시각화
- `numpy>=1.24.0` - 수치 연산

전체 의존성은 `requirements.txt` 파일을 참조하세요.

## 🌐 배포

### Render.com 배포

프로젝트는 Render.com에서 자동 배포됩니다:

1. **GitHub 리포지토리**: https://github.com/joonwoopapa/KD_CAA
2. **자동 배포**: main 브랜치 푸시 시 자동 배포
3. **설정 파일**: `render.yaml`을 통한 자동 설정

### 로컬 개발 환경

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
streamlit run app.py
```

## 📖 사용법

### 1. 홈페이지
- 두 가지 예측 모델 중 선택
- 각 모델의 특징과 용도 확인

### 2. 관상동맥류 예측
**입력 변수 (16개)**:
- Echo Z-scores: LAD, LMCA, RCA, LCx
- 임상 정보: 발열기간, 성별
- 혈액검사: ALT, AST, Hb, HCT, 인, CRP, ESR, 총빌리루빈, CO2, 알부민

**출력**:
- 예측 확률과 위험도 평가
- SHAP 분석을 통한 각 변수의 기여도 시각화

### 3. IVIG 저항성 예측
**입력 변수 (13개)**:
- 혈액검사: 혈소판수치, 림프구, 호중구, 콜레스테롤
- 염증지표: CRP, 인, 총빌리루빈, 칼슘
- 간기능: AST, Plateletcrit(%)
- 기타: Echo LAD Z-score, ANC, CO2

**출력**:
- IVIG 치료 반응성 예측 결과
- 상세한 변수별 영향도 분석

### 4. SHAP 분석 해석
- **Waterfall Plot**: 각 변수가 예측에 미치는 영향을 단위와 함께 표시
- **Bar Chart**: 변수 중요도를 명확하게 시각화
- **개선된 가독성**: 모든 변수명에 단위 포함, 성별은 직관적 표시

## 🆕 최근 업데이트 (v2.1)

### SHAP 시각화 개선
- ✅ 모든 변수를 개별적으로 표시 ("6 other features" 제거)
- ✅ 변수명에 단위 포함으로 가독성 향상
- ✅ 성별 변수를 "Male"/"Female"로 직관적 표시
- ✅ 그래프 크기 및 스타일 표준화 (12x8, 폰트 크기 10pt)

### 변수명 업데이트
- ✅ IVIG 예측에서 Procalcitonin → Plateletcrit(%)로 변경
- ✅ 모든 혈액검사 변수에 단위 명시

### 코드 구조 개선
- ✅ 모듈화된 컴포넌트 구조
- ✅ 유틸리티 함수 분리
- ✅ 코드 가독성 및 유지보수성 향상

## ⚠️ 주의사항

- **의료용 도구**: 이 시스템은 의료진의 판단을 보조하는 도구입니다
- **최종 판단**: 진단과 치료 결정은 반드시 의료진이 내려야 합니다
- **참고 자료**: 예측 결과는 참고 자료로만 활용하세요
- **데이터 보안**: 환자 정보 입력 시 개인정보 보호에 유의하세요
- **모델 버전**: XGBoost v1.6.1로 훈련된 모델 (v1.7.0에서 경고 메시지 정상)

## 🤝 기여

프로젝트 개선에 기여하고 싶으시다면:
1. Fork this repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면 GitHub Issues를 생성해 주세요:
- 저장소: https://github.com/joonwoopapa/KD_CAA

---

**개발**: 박우영
**최종 업데이트**: 2025년 7월 
