# API 문서

## 모델 입력 변수

### 관상동맥류 예측 모델 (CAA)

**총 16개 변수**

| 변수명 | 설명 | 단위 | 범위 |
|--------|------|------|------|
| initial_echo_LAD_Z | 좌전하행지 Z-score | - | -5.0 ~ 15.0 |
| initial_echo_LMCA_Z | 좌주관상동맥 Z-score | - | -5.0 ~ 15.0 |
| initial_echo_RCA_Z | 우관상동맥 Z-score | - | -5.0 ~ 15.0 |
| initial_echo_LCx_Z | 좌회선지 Z-score | - | -5.0 ~ 15.0 |
| fever_duration | 발열 지속 기간 | 일 | 1 ~ 30 |
| Sex | 성별 | - | 0(여), 1(남) |
| ALT_before | 알라닌 아미노전이효소 | IU/L | 5 ~ 500 |
| AST_before | AST | IU/L | 5 ~ 500 |
| HCT_before | 헤마토크릿 | % | 20 ~ 50 |
| CRP_before | C-반응성 단백질 | mg/dL | 0.0 ~ 100.0 |
| ESR_before | ESR | mm/hr | 0.0 ~ 9.0 |

| P_before | 인 | mg/dL | 2.0 ~ 8.0 |
| TB_before | 총 빌리루빈 | mg/dL | 0.1 ~ 100.0 |
| Alb_before | 알부민 | g/dL | 3.3 ~ 15.0 |
| Hb_before | 헤모글로빈 | g/dL | 10.5 ~ 25.0 |
| Protein_before | 총 단백질 | g/dL | 6.0 ~ 10.0 |

### IVIG 저항성 예측 모델

**총 13개 변수**

| 변수명 | 설명 | 단위 | 범위 |
|--------|------|------|------|
| PLT_before | 혈소판 수 | ×10³/μL | 100 ~ 1000 |
| Lympho_before | 림프구 수 | ×10³/μL | 0.5 ~ 10.0 |
| Seg_before | 분절호중구 | % | 20 ~ 90 |
| Chol_before | 총 콜레스테롤 | mg/dL | 100 ~ 300 |
| CRP_before | C-반응성 단백질 | mg/dL | 0.0 ~ 30.0 |
| P_before | 인 | mg/dL | 2.0 ~ 8.0 |
| TB_before | 총 빌리루빈 | mg/dL | 0.1 ~ 10.0 |
| Ca_before | 칼슘 | mg/dL | 7.0 ~ 12.0 |
| AST_before | 아스파르테이트 아미노전이효소 | U/L | 10 ~ 400 |
| PCT_before | 프로칼시토닌 | ng/mL | 0.0 ~ 10.0 |
| initial_echo_LAD_Z | 좌전하행지 Z-score | - | -5.0 ~ 15.0 |
| ANC_before | 절대호중구수 | ×10³/μL | 1.0 ~ 20.0 |
| CO2_before | 이산화탄소 | mEq/L | 10 ~ 35 |

## 출력 형식

### 예측 결과
```json
{
  "probability": 0.65,
  "risk_level": "High Risk",
  "recommendation": "정밀 검사 권장"
}
```

### 위험도 분류

**관상동맥류**
- 고위험 (>70%): 정밀 검사 권장
- 중위험 (30-70%): 지속적인 모니터링 필요  
- 저위험 (<30%): 정상 범위

**IVIG 저항성**
- 고저항성 (>70%): 대체 치료법 고려
- 중저항성 (30-70%): 신중한 모니터링
- 저저항성 (<30%): IVIG 치료 효과적 
