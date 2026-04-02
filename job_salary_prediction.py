import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 모델 및 인코더 로드

model = joblib.load('model/job_salary_prediction_model02.pkl')
encoders = joblib.load('model/job_salary_prediction_labelencoders02.pkl')
st.set_page_config(page_title="Job Salary Predictor", page_icon="💼")

st.title("Job Salary Prediction App 💼")
st.markdown("---")
st.write('직업 관련 정보를 입력하시면 인공지능이 예상 연봉을 예측해 드립니다.')

# 레이아웃 구성 (컬럼 활용)
col1, col2 = st.columns(2)
# 사용자 입력
with col1:
    st.subheader("📋 기본 정보")
    # Categorical: job_title
    job_title = st.selectbox('직업명 (Job Title)', encoders['job_title'].classes_)
    
    # Numerical: experience_years
    experience_years = st.slider("경력 기간 (Experience Years)", 0, 30, 5)
    
    # Categorical: education_level
    education_level = st.selectbox('학력 수준 (Education Level)', encoders['education_level'].classes_)
    
    # Numerical: skills_count
    skills_count = st.number_input('보유 스킬 개수 (Skills Count)', 0, 20, 5)
    
    # Categorical: industry
    industry = st.selectbox('산업 분야 (Industry)', encoders['industry'].classes_)

with col2:
    st.subheader("🏢 근무 환경")
    # Categorical: company_size
    company_size = st.selectbox('회사 규모 (Company Size)', encoders['company_size'].classes_)
    
    # Categorical: location
    location = st.selectbox('근무 지역 (Location)', encoders['location'].classes_)
    
    # Categorical: remote_work
    remote_work = st.selectbox('원격 근무 여부 (Remote Work)', encoders['remote_work'].classes_)
    
    # Numerical: certifications
    certifications = st.number_input('보유 자격증 개수 (Certifications)', 0, 5, 1)

# 예측을 위한 데이터 변환 (Label Encoding 적용)
job_title_val = encoders['job_title'].transform([job_title])[0]
education_level_val = encoders['education_level'].transform([education_level])[0]
industry_val = encoders['industry'].transform([industry])[0]
company_size_val = encoders['company_size'].transform([company_size])[0]
location_val = encoders['location'].transform([location])[0]
remote_work_val = encoders['remote_work'].transform([remote_work])[0]

# 학습 데이터와 동일한 컬럼 순서 유지 (노트북 분석 기반)
# [job_title, experience_years, education_level, skills_count, industry, company_size, location, remote_work, certifications]
input_data = pd.DataFrame({
    'job_title': [job_title_val],
    'experience_years': [experience_years],
    'education_level': [education_level_val],
    'skills_count': [skills_count],
    'industry': [industry_val],
    'company_size': [company_size_val],
    'location': [location_val],
    'remote_work': [remote_work_val],
    'certifications': [certifications]
})

st.markdown("---")

# 예측 실행
if st.button("예상 연봉 확인하기", use_container_width=True):
    # 모델 예측
    prediction = model.predict(input_data)[0]
    
    # 결과 출력
    st.success(f"당신의 예상 연봉은 약 **${prediction:,.0f}** 입니다.")
    
    # 입력 데이터 상세 요약
    with st.expander("입력하신 데이터 상세 보기"):
        display_df = pd.DataFrame({
            '항목': ['직업명', '경력', '학력', '스킬수', '산업군', '회사규모', '지역', '원격근무', '자격증'],
            '입력값': [job_title, f"{experience_years}년", education_level, skills_count, industry, company_size, location, remote_work, certifications]
        })
        st.table(display_df)

st.sidebar.info("이 앱은 학습된 머신러닝 모델을 기반으로 연봉을 예측합니다.")