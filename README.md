# Credit Scoring & Portfolio Optimization System

## 1. 프로젝트 개요 (Overview)
본 프로젝트는 **Lending Club P2P 대출 데이터**를 분석하여 부도 위험을 최소화하고 수익률을 극대화하는 것을 목표로 한다.

### 🌟 핵심 기능 (Key Features)
* **Dual-Model Engine:**
  * **Risk Model (XGBoost Classifier):** 대출 신청자의 부도 확률(PD) 예측.
  * **Profit Model (XGBoost Regressor):** 대출 실행 시 예상 수익률(Return) 예측.

* **Custom Sharpe Ratio Optimization:**
  * 무위험 자산(국채) 대비 초과 수익을 평가.
  * **통계적 엄밀성:** 표본 분산 공식($N-1$)을 적용하여 수치적 정밀도 확보.

* **Strict Data Leakage Prevention:**
  * 학습 시점에 알 수 없는 미래 변수(사후 정보)를 원천적으로 배제하여 데이터 누수 방지.

* **Validation Modes:**
  * **Full Test Set Heatmap:** 전체 테스트 셋에 대한 전수 조사를 통해 최적의 임계값($Th_{PD}, Th_{Ret}$) 탐색.
  * **Permutation Test (순열 검정):** 발견된 전략이 통계적으로 유의미한지(단순 운이 아닌지) p-value를 통해 검증.


  ```bash
├── 📁             # [High-Performance] C++ 원본 소스
│   ├── main.cpp              # 메인 실행 파일 (진입점)
│   ├── ExperimentManager.cpp # 실험 설계, 히트맵 생성, 검증 로직 구현 (핵심)
│   ├── ExperimentManager.h   # 클래스 및 구조체 선언 헤더
│   ├── ExperimentManager.cpp
│   └── CsvLoader.h           # 멀티스레드 CSV 파싱 및 전처리 (Data Leakage 방지 포함)
│
├── 📁 script/           # [Accessibility] Python 이식 버전
│   ├── environment.yml           # Conda 사용자용 환경 설정 파일 (권장)
│   └── .py               # C++ 로직을 1:1로 이식한 통합 스크립트 (팀원 실행용)
│
└── README.md                 # 프로젝트 설명서



# 1. 가상 환경 생성 (이름: credit_scoring_env)
conda env create -f env.yml

# 2. 가상 환경 활성화
conda activate loan_status