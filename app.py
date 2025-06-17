import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="학생 성과 요인에 따른 성적 예측", page_icon="🎓")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 모델 로딩
@st.cache_resource
def load_models():
    try:
        models = {
            'linear': joblib.load("linear_model.pkl"),
            'ridge': joblib.load("ridge_model.pkl"),
            'lasso': joblib.load("lasso_model.pkl"),
            'poly': joblib.load("polynomial_model.pkl"),
            'poly_ridge': joblib.load("poly_ridge_model.pkl"),
            'scaler': joblib.load("scaler.pkl"),
            'poly_features': joblib.load("poly_features.pkl"),
            'features': joblib.load("features.pkl"),
            'r2_scores': joblib.load("r2_scores.pkl")  
        }
        return models
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None

# 사이드바 - 정보
with st.sidebar:
    st.header("20190838_유원근_2차 프로그래밍 과제")
    
    # R² 점수에 따라 모델 정렬
    r2_scores = joblib.load("r2_scores.pkl")
    sorted_models = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
    
    st.markdown("### 모델 성능 순위 (R² 기준)")
    for i, (model_name, r2_score) in enumerate(sorted_models, 1):
        st.markdown(f"{i}. **{model_name}**: R² = {r2_score:.3f}")
    
    st.markdown("*R² 값이 1에 가까울수록 모델의 예측력이 높습니다*")

st.title('🎓 학생 성과 요인에 따른 성적 예측')

# 모델 로드
models = load_models()
if not models:
    st.stop()

# 탭으로 구분
tab1, tab2, tab3 = st.tabs(['📚 학습 정보', '👨‍👩‍👧‍👦 가정 환경', '🏫 학교 환경'])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider('주간 학습 시간 (시간/주)', 0, 50, 20)
        attendance = st.slider('출석률 (%)', 0, 100, 85)
        prev_score = st.slider('이전 시험 점수', 0, 100, 75)
        sleep_hours = st.slider('평균 수면 시간 (시간/일)', 0, 12, 7)
        exercise_hours = st.slider('주간 운동 시간 (시간/주)', 0, 20, 3)
        tutoring = st.slider('월간 과외 횟수 (회/월)', 0, 20, 2)
    
    with col2:
        motivation = st.selectbox('학습 동기 수준', ['Low', 'Medium', 'High'])
        learning_disability = st.selectbox('학습 장애 유무', ['No', 'Yes'])
        internet_access = st.selectbox('인터넷 사용 가능', ['No', 'Yes'])
        club_activity = st.selectbox('동아리 활동 참여', ['No', 'Yes'])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        parent_edu = st.selectbox('부모 교육 수준', ['High School', 'College', 'Postgraduate'])
        family_income = st.selectbox('가족 소득 수준', ['Low', 'Medium', 'High'])
        parent_involvement = st.selectbox('부모 개입도', ['Low', 'Medium', 'High'])
    
    with col2:
        distance_home = st.selectbox('집에서 학교까지 거리', ['Near', 'Moderate', 'Far'])
        peer_influence = st.selectbox('학급 또래 영향', ['Negative', 'Neutral', 'Positive'])

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        school_type = st.selectbox('학교 유형', ['Public', 'Private'])
        teacher_quality = st.selectbox('교사 수준', ['Low', 'Medium', 'High'])
    
    with col2:
        access_resources = st.selectbox('교육 자원 접근성', ['Low', 'Medium', 'High'])
        gender = st.selectbox('성별', ['Female', 'Male'])

# 예측 실행
if st.button('🎯 성적 예측하기', type='primary'):
    try:
        # 입력 데이터 구성
        input_data = {
            'Hours_Studied': study_hours,
            'Attendance': attendance,
            'Previous_Scores': prev_score,
            'Sleep_Hours': sleep_hours,
            'Physical_Activity': exercise_hours,
            'Tutoring_Sessions': tutoring,
            'Motivation_Level': motivation,
            'Learning_Disabilities': learning_disability,
            'Internet_Access': internet_access,
            'Extracurricular_Activities': club_activity,
            'Parental_Education_Level': parent_edu,
            'Family_Income': family_income,
            'Parental_Involvement': parent_involvement,
            'Distance_from_Home': distance_home,
            'Peer_Influence': peer_influence,
            'School_Type': school_type,
            'Teacher_Quality': teacher_quality,
            'Access_to_Resources': access_resources,
            'Gender': gender
        }
        
        # 범주형 변수 인코딩
        categorical_mappings = {
            'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
            'Learning_Disabilities': {'No': 0, 'Yes': 1},
            'Internet_Access': {'No': 0, 'Yes': 1},
            'Extracurricular_Activities': {'No': 0, 'Yes': 1},
            'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
            'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
            'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
            'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
            'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
            'School_Type': {'Public': 0, 'Private': 1},
            'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
            'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
            'Gender': {'Female': 0, 'Male': 1}
        }
        
        # 데이터프레임 생성 및 인코딩
        input_df = pd.DataFrame([input_data])
        for col, mapping in categorical_mappings.items():
            input_df[col] = input_df[col].map(mapping)
        
        # 특성 순서 맞추기
        input_df = input_df.reindex(columns=models['features'])
        
        # 예측 수행
        input_scaled = models['scaler'].transform(input_df)
        input_poly = models['poly_features'].transform(input_scaled)
        
        predictions = {
            'Linear': float(models['linear'].predict(input_scaled)[0]),
            'Ridge': float(models['ridge'].predict(input_scaled)[0]),
            'Lasso': float(models['lasso'].predict(input_scaled)[0]),
            'Polynomial': float(models['poly'].predict(input_poly)[0]),
            'Poly+Ridge': float(models['poly_ridge'].predict(input_poly)[0])
        }
        
        # 결과 표시
        st.success('✅ 예측 완료!')
        
        # 최고 점수와 평균 점수 표시
        best_score = max(predictions.values())
        avg_score = np.mean(list(predictions.values()))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('최고 예측 점수', f'{best_score:.1f}점')
        with col2:
            st.metric('평균 예측 점수', f'{avg_score:.1f}점')
        
        # 모델별 예측 결과 시각화
        st.subheader('모델별 예측 결과')
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(predictions.keys(), predictions.values())
        
        # 막대 위에 점수 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
        
        ax.set_ylabel('예측 점수')
        ax.set_title('모델별 성적 예측')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # 상세 결과 테이블
        st.subheader('상세 예측 결과')
        results_df = pd.DataFrame({
            '모델': list(predictions.keys()),
            '예측 점수': [f'{v:.2f}' for v in predictions.values()]
        })
        st.dataframe(results_df, use_container_width=True)
        
    except Exception as e:
        st.error(f'예측 중 오류가 발생했습니다: {str(e)}')
