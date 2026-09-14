# 💼 Job Salary Prediction (직무 연봉 예측 프로젝트)

직무, 근무 경력, 요구 기술 등 다양한 구인/직무 데이터를 기반으로 예상 연봉을 분석하고 예측하는 머신러닝 회귀(Regression) 프로젝트입니다.

---

## 📌 주요 특징 및 기능
- **데이터 분석 및 전처리**: 직무 관련 데이터셋 정제, 결측치 처리, 피처 엔지니어링 수행
- **연봉 예측 모델링**: 머신러닝 회귀 알고리즘을 활용한 최적의 연봉 예측 모델 훈련
- **분석 보고서**: 모델 평가 지표(RMSE, R² 등) 및 피처 중요도 분석 리포트 생성

---

## 🛠️ 기술 스택
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment**: Jupyter Notebook, VS Code Dev Containers

---

## 📁 디렉터리 구조
```text
├── .devcontainer/              # 개발 환경 컨테이너 구성 파일
├── dataset/                   # 분석 및 학습용 원본/가공 데이터셋
├── model/                     # 학습된 모델 파일 저장소 (.pkl 등)
├── report/                    # 모델 평가 지표 및 EDA 시각화 리포트
├── job_salary_prediction.ipynb # 데이터 탐색(EDA) 및 모델 실험 노트북
├── job_salary_prediction.py    # 데이터 전처리, 학습 및 추론 실행 스크립트
└── requirements.txt           # 의존성 패키지 목록
