# 🎓 학생 성과 예측 프로젝트 (Student Performance Prediction)

## 2차 프로그래밍 과제

### 기존 수업 회귀 코드 참고 수정 및 Cursor 활용

---

**배포 링크**: https://studentperformancefactors-regression-predict.streamlit.app

---

## 📋 프로젝트 개요

### **프로젝트 목적**

학생들의 다양한 요인(공부 시간, 출석률, 부모 참여도 등)을 분석하여 시험 성적을 예측하는 회귀 분석 프로젝트입니다.

### **개발 환경**

- **언어**: Python
- **주요 라이브러리**: pandas, numpy, scikit-learn, streamlit, plotly
- **개발 도구**: Jupyter Notebook, Cursor AI
- **배포 플랫폼**: Streamlit Community Cloud

## 📊 데이터셋 정보

### **데이터 소개**

- **데이터셋**: `StudentPerformanceFactors.csv`
- **출처**: Kaggle Dataset
- **데이터 크기**: 6,607개 샘플, 20개 특성
- **타겟 변수**: `Exam_Score` (시험 성적, 0-100점)

### **특성 (Features)**

| 특성명                     | 설명                      | 타입       |
| -------------------------- | ------------------------- | ---------- |
| Hours_Studied              | 공부 시간                 | 수치형     |
| Attendance                 | 출석률                    | 수치형     |
| Gender                     | 성별                      | 범주형     |
| Previous_Scores            | 이전 성적                 | 수치형     |
| Motivation_Level           | 동기 수준                 | 범주형     |
| Tutoring_Sessions          | 과외 횟수                 | 수치형     |
| Extracurricular_Activities | 과외 활동 참여 여부       | 범주형     |
| Parental_Involvement       | 부모 참여도               | 범주형     |
| Access_to_Resources        | 자원 접근성               | 범주형     |
| Internet_Access            | 인터넷 접근 여부          | 범주형     |
| Family_Income              | 가족 소득                 | 범주형     |
| Teacher_Quality            | 교사 품질                 | 범주형     |
| School_Type                | 학교 유형 (공립/사립)     | 범주형     |
| Sleep_Hours                | 수면 시간                 | 수치형     |
| Physical_Activity          | 신체 활동                 | 수치형     |
| Learning_Disabilities      | 학습 장애 여부            | 범주형     |
| Peer_Influence             | 동료 영향                 | 범주형     |
| Parental_Education_Level   | 부모 교육 수준            | 범주형     |
| Distance_from_Home         | 집에서의 거리             | 범주형     |
| **Exam_Score**             | **시험 성적 (타겟 변수)** | **수치형** |

## 🔧 데이터 전처리 과정

### 1. **데이터 정제**

- 컬럼명 공백 제거 및 언더스코어 변환
- 불필요한 컬럼 제거

### 2. **결측치 처리**

- `dropna()` 함수를 사용한 결측치 제거
- 전처리 후 최종 데이터: 5,915개 샘플

### 3. **이상치 처리**

- IQR(Interquartile Range) 방법 사용
- 수치형 변수에만 적용하여 극값 제거

### 4. **범주형 변수 인코딩**

- **라벨 인코딩** 적용
- 순서형: Low(0), Medium(1), High(2)
- 이진형: No(0), Yes(1)

### 5. **특성 스케일링**

- **StandardScaler** 사용
- **개선사항**: 기존 수업에서는 전체 데이터를 스케일링 후 train/test 분할했지만, 이번 프로젝트에서는 **분할 후 train 데이터만으로 스케일링**하여 데이터 누출 방지

## 🤖 사용된 모델 알고리즘

### **회귀 모델**

1. **Linear Regression** (다중 선형 회귀)
2. **RidgeCV** (릿지 회귀)
3. **LassoCV** (라쏘 회귀)
4. **Polynomial Regression** (다항 회귀) - 2차 다항식 특성
5. **Polynomial + RidgeCV** - 다항식 + 정규화

## 📈 회귀 성능 결과

| 모델                  | R² Score   | MSE    | MAE    | RMSE   |
| --------------------- | ---------- | ------ | ------ | ------ |
| **LassoCV**           | **0.6405** | 5.8532 | 0.5449 | 2.4193 |
| **RidgeCV**           | **0.6405** | 5.8544 | 0.5437 | 2.4196 |
| **Linear Regression** | **0.6404** | 5.8554 | 0.5436 | 2.4198 |
| **Poly+RidgeCV**      | 0.6241     | 6.1216 | 0.6950 | 2.4742 |
| **Polynomial**        | 0.6231     | 6.1377 | 0.6997 | 2.4774 |

### **주요 성과**

- **최고 성능**: LassoCV 모델 (R² = 0.6405)
- **정규화 효과**: Ridge와 Lasso 모델이 유사한 성능으로 과적합 방지 효과 확인
- **특성 선택**: Lasso 회귀를 통한 중요 특성 자동 선택

## 🌐 Streamlit 웹 애플리케이션

### **주요 기능**

1. **모델 선택**: 5개 회귀 모델 중 원하는 모델 선택 가능
2. **실시간 예측**: 19개 특성 입력을 통한 즉시 성적 예측
3. **결과 시각화**:
   - 모델별 예측 점수 비교 차트
   - 예측 신뢰도 표시
   - 성적 등급 자동 계산 (A+, A, B, C, D)
4. **사용자 친화적 인터페이스**:
   - 탭 구조로 체계적인 입력 폼
   - 각 입력 항목별 도움말 제공
   - 반응형 레이아웃

### **배포 과정**

- **Streamlit Community Cloud** 활용
- **Github 연동**: 레포지토리 연결 후 업로드
- **배포 환경 설정**: requirements.txt를 통한 의존성 관리
- **자동 배포**: Github push 시 자동 재배포

## 😭 아쉬운 점

- 더 좋은 모델들도 많지만 기초 학습부터 해보았기에 모델의 정확도가 뛰어나진 않았던것이 아쉬웠던 경험입니다.
