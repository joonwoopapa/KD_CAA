# 🏥 Kawasaki Disease Prediction System

통합된 가와사키병 예측 시스템으로, 두 가지 주요 임상 결과를 예측할 수 있습니다:
- **관상동맥류 발생 예측** (XGBoost 모델)
- **IVIG 저항성 예측** (RandomForest 모델)

## 🌟 주요 기능

### 🫀 Coronary Aneurysm Prediction
- **모델**: XGBoost
- **입력**: 15개 임상 변수
- **출력**: 관상동맥류 발생 확률
- **설명**: SHAP을 통한 예측 근거 제공

### 💉 IVIG Resistance Prediction  
- **모델**: RandomForest
- **입력**: 14개 임상 변수
- **출력**: IVIG 저항성 확률
- **설명**: SHAP을 통한 예측 근거 제공

### 📊 SHAP 분석
- Waterfall Plot: 각 변수의 기여도 시각화
- Force Plot: 상호작용 효과 분석
- 설명 가능한 AI를 통한 임상적 해석 지원

## 🚀 빠른 시작

### Docker를 사용한 실행 (권장)

```bash
# 1. 리포지토리 클론
git clone <repository-url>
cd KD_CAA

# 2. Docker 이미지 빌드
docker build -t kd-unified-app .

# 3. 컨테이너 실행
docker run -p 8501:8501 kd-unified-app

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
├── app.py                    # 메인 Streamlit 애플리케이션
├── models/                   # ML 모델 파일들
│   ├── xgb_model.pkl        # XGBoost 관상동맥류 예측 모델
│   ├── shap_explainer.pkl   # CAA SHAP explainer
│   ├── rf_model.pkl         # RandomForest IVIG 저항성 모델
│   └── shap_explainer_rf.pkl # IVIG SHAP explainer
├── docs/                     # 문서 파일들
├── requirements.txt          # Python 의존성
├── runtime.txt              # Python 버전 (3.12)
├── Dockerfile               # Docker 이미지 빌드 설정
├── docker-compose.yml       # Docker Compose 설정
├── render.yaml              # Render.com 배포 설정
└── README.md                # 이 파일
```

## 🔧 기술 스택

- **Frontend**: Streamlit
- **ML Models**: XGBoost, RandomForest
- **Explainability**: SHAP
- **Deployment**: Docker, Render.com
- **Language**: Python 3.12

## 📦 의존성

주요 라이브러리:
- `streamlit>=1.28.0` - 웹 애플리케이션 프레임워크
- `xgboost>=2.0.0` - 그래디언트 부스팅 모델
- `scikit-learn>=1.2.0` - 랜덤포레스트 모델
- `shap>=0.43.0` - 모델 해석
- `pandas>=2.0.0` - 데이터 처리
- `matplotlib>=3.7.0` - 시각화

전체 의존성은 `requirements.txt` 파일을 참조하세요.

## 🌐 배포

### Render.com 배포

1. **GitHub 리포지토리 생성 및 푸시**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Render.com 설정**
- Render.com에서 새 Web Service 생성
- GitHub 리포지토리 연결
- `render.yaml` 파일이 자동으로 감지됨

3. **환경 변수 (선택사항)**
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`

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
- 15개 임상 변수 입력
- Echo Z-score, 혈액검사 결과 등
- 예측 확률과 위험도 평가 확인

### 3. IVIG 저항성 예측  
- 14개 임상 변수 입력
- 혈액검사 결과, 염증 수치 등
- IVIG 치료 반응성 예측 결과 확인

### 4. SHAP 분석 해석
- **Waterfall Plot**: 각 변수가 예측에 미치는 영향
- **Force Plot**: 기준값 대비 변화량과 방향

## ⚠️ 주의사항

- **의료용 도구**: 이 시스템은 의료진의 판단을 보조하는 도구입니다
- **최종 판단**: 진단과 치료 결정은 반드시 의료진이 내려야 합니다
- **참고 자료**: 예측 결과는 참고 자료로만 활용하세요
- **데이터 보안**: 환자 정보 입력 시 개인정보 보호에 유의하세요

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

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

**개발**: 가와사키병 예측 시스템 개발팀  
**최종 업데이트**: 2025년 7월 