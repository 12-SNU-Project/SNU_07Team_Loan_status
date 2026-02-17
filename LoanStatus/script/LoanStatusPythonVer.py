import pandas as pd
import numpy as np
import xgboost as xgb
import os
from tqdm import tqdm  

# =========================================================
# 모델 하이퍼파라미터 (ModelConfig와 동일)
# =========================================================
CLS_CONFIG = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 7,
    'eta': 0.05,
    'min_child_weight': 1.0,
    'scale_pos_weight': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'nthread': os.cpu_count() // 2  # C++과 동일하게 절반 사용
}

REG_CONFIG = {
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae', 
    'max_depth': 5,
    'eta': 0.05,
    'min_child_weight': 1.0,
    #'scale_pos_weight': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'nthread': os.cpu_count() // 2
}

NUM_ROUNDS = 400

# =========================================================
#  샤프 지수 계산 함수
# =========================================================
def calculate_sharpe_ratio(actual_returns, bond_yields, approval_mask):
    """
    C++의 CalculateSharpeRatio와 수학적으로 동일한 로직.
    - 거절된 건은 초과 수익 0으로 처리
    - 분모는 전체 N (Test Size)
    - 분산 계산 시 N-1 (Sample Variance) 사용
    """
    n = len(actual_returns)
    if n <= 1:
        return 0.0

    # 1. 초과 수익률 계산 (승인: 수익-국채, 거절: 0)
    excess_returns = np.where(approval_mask, actual_returns - bond_yields, 0.0)

    # 2. 평균 및 표준편차 계산
    # np.std(ddof=1)은 N-1로 나누는 표본 표준편차를 의미함 (C++ 로직과 동일)
    mean_excess = np.mean(excess_returns)
    std_dev = np.std(excess_returns, ddof=1)

    # 3. 샤프 지수 반환
    if std_dev < 1e-9:
        return 0.0
    
    return mean_excess / std_dev

# =========================================================
# ExperimentManager 클래스
# =========================================================
class ExperimentManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.ignore_cols = [
            "Actual_term", "total_pymnt", "last_pymnt_amnt", "내부수익률",
            "loan_status", "Return", "Bond" # Bond는 별도 추출 후 feature에선 제외
        ]
        self.target_col = "loan_status"

    def load_and_preprocess(self, split_ratio=0.8):
        print(">>> [1/4] Data Loading & Preprocessing...")
        
        # 한글 깨짐 방지 (utf-8-sig)
        df = pd.read_csv(self.file_path, encoding='utf-8-sig') # 또는 'cp949'
        
        # 데이터 분할 지점 계산
        total_rows = len(df)
        split_point = int(total_rows * split_ratio)
        test_size = total_rows - split_point

        # 1. Target & Returns & Bond 분리
        y_cls = df[self.target_col].values
        y_reg = df['Return'].values
        bond_yields = df['Bond'].values

        # 2. Features 분리 (Ignore List 제거)
        X = df.drop(columns=self.ignore_cols + ['Bond'], errors='ignore')
        feature_names = X.columns.tolist()
        X = X.values # numpy array로 변환

        # 3. Train / Test Split (Sequential Split - 셔플 없음)
        X_train, X_test = X[:split_point], X[split_point:]
        y_cls_train, _ = y_cls[:split_point], y_cls[split_point:] # Test Label은 예측에 안씀
        y_reg_train, _ = y_reg[:split_point], y_reg[split_point:]

        # 검증용 실제 데이터 (Test Set)
        actual_returns = y_reg[split_point:]
        test_bond_yields = bond_yields[split_point:]

        # 4. XGBoost DMatrix 생성
        dtrain_cls = xgb.DMatrix(X_train, label=y_cls_train, feature_names=feature_names)
        dtest_cls = xgb.DMatrix(X_test, feature_names=feature_names) # Label 불필요

        dtrain_reg = xgb.DMatrix(X_train, label=y_reg_train, feature_names=feature_names)
        dtest_reg = xgb.DMatrix(X_test, feature_names=feature_names)

        return {
            'dtrain_cls': dtrain_cls, 'dtest_cls': dtest_cls,
            'dtrain_reg': dtrain_reg, 'dtest_reg': dtest_reg,
            'actual_returns': actual_returns,
            'bond_yields': test_bond_yields,
            'test_size': test_size
        }

    def train_and_predict(self, data):
        print(">>> [2/4] Training Models (Classification & Regression)...")
        
        # 분류 모델 학습
        bst_cls = xgb.train(CLS_CONFIG, data['dtrain_cls'], num_boost_round=NUM_ROUNDS)
        
        # 회귀 모델 학습
        bst_reg = xgb.train(REG_CONFIG, data['dtrain_reg'], num_boost_round=NUM_ROUNDS)

        print(">>> [3/4] Predicting Test Set...")
        pred_pd = bst_cls.predict(data['dtest_cls'])
        pred_ret = bst_reg.predict(data['dtest_reg'])

        return pred_pd, pred_ret

    def run_heatmap_full_test_set(self, data, pred_pd, pred_ret):
        print("\n" + "="*60)
        print(">>> [Mode 4] Full Test Set Heatmap (Python Version)")
        print("="*60)

        # 결과 저장용 리스트
        results = []
        
        # 최적 파라미터 추적용
        best_sharpe = -999.0
        best_params = {'pd_th': 0.0, 'ret_th': 0.0}

        # Grid Search Loops
        pd_thresholds = [i * 0.01 for i in range(1, 41)]   # 0.01 ~ 0.40
        ret_thresholds = [j * 0.005 for j in range(1, 21)] # 0.005 ~ 0.100

        total_iter = len(pd_thresholds) * len(ret_thresholds)
        
        print(f"Scanning {total_iter} combinations...")

        for pd_th in tqdm(pd_thresholds, desc="Grid Search"):
            for ret_th in ret_thresholds:
                # 1. 필터링 (Vectorized Operation)
                # 조건: 부도확률 < 임계값 AND 기대수익률 > 임계값
                approval_mask = (pred_pd < pd_th) & (pred_ret > ret_th)
                count = np.sum(approval_mask)

                mean = 0.0
                std_dev = 0.0
                sharpe = 0.0
                rate = (count / data['test_size']) * 100.0

                if count >= 10:
                    # 2. 통계 계산
                    # 실제 샤프지수 계산
                    sharpe = calculate_sharpe_ratio(
                        data['actual_returns'], 
                        data['bond_yields'], 
                        approval_mask
                    )
                    
                    # CSV 출력용 Mean, StdDev 계산 (N-1 표본분산 기준)
                    excess = np.where(approval_mask, data['actual_returns'] - data['bond_yields'], 0.0)
                    mean = np.mean(excess)
                    std_dev = np.std(excess, ddof=1) # 표본 분산(N-1) 공식

                    # 3. 최고점 갱신 (Best Tracker)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params['pd_th'] = pd_th
                        best_params['ret_th'] = ret_th

                # 결과 추가
                results.append({
                    'PD_Threshold': pd_th,
                    'Return_Threshold': ret_th,
                    'Sharpe_Ratio': sharpe,
                    'Mean_Excess_Return': mean,
                    'Std_Dev': std_dev,
                    'Approved_Count': count,
                    'Approved_Rate': rate
                })

        # CSV 저장
        res_df = pd.DataFrame(results)
        res_df.to_csv('heatmap_full_testset_python.csv', index=False)
        print(">>> Heatmap saved to 'heatmap_full_testset_python.csv'")

        # 순열 검정 호출
        if best_sharpe > -900:
            self.perform_permutation_test(
                pred_pd, pred_ret, 
                data['actual_returns'], data['bond_yields'],
                best_params['pd_th'], best_params['ret_th']
            )

    def perform_permutation_test(self, pred_pd, pred_ret, actual_ret, bond_yields, best_pd, best_ret, iterations=1000):
        print("\n" + "="*60)
        print(">>> [Permutation Test] Verifying Best Strategy Significance")
        print("="*60)
        print(f"Best Config Found: PD < {best_pd:.2f}, Ret > {best_ret:.3f}")

        # 1. 고정된 승인 마스크 생성
        fixed_mask = (pred_pd < best_pd) & (pred_ret > best_ret)
        approved_cnt = np.sum(fixed_mask)
        
        # 2. 오리지널 샤프지수
        original_sharpe = calculate_sharpe_ratio(actual_ret, bond_yields, fixed_mask)
        print(f"Original Sharpe Ratio: {original_sharpe:.5f} (Count: {approved_cnt})")
        print(f"Running {iterations} permutations...")

        better_count = 0
        
        # 3. 순열 검정 루프 (Numpy Shuffle 활용)
        # 데이터를 복사해서 섞음 (원본 보존)
        shuffled_ret = actual_ret.copy()
        shuffled_bond = bond_yields.copy()

        # 인덱스 생성
        indices = np.arange(len(actual_ret))

        for _ in tqdm(range(iterations), desc="Permutation"):
            # 인덱스 셔플
            np.random.shuffle(indices)
            
            # 정답지(Return, Bond)만 뒤섞음 (Mask는 고정)
            current_shuffled_ret = shuffled_ret[indices]
            current_shuffled_bond = shuffled_bond[indices]

            # 섞인 데이터로 샤프지수 계산
            random_sharpe = calculate_sharpe_ratio(
                current_shuffled_ret, 
                current_shuffled_bond, 
                fixed_mask
            )

            if random_sharpe >= original_sharpe:
                better_count += 1

        # 4. 결과 판정
        p_value = (better_count + 1) / (iterations + 1)
        
        print("-" * 60)
        print(f"Permutation Test Result:")
        print(f" - Better Random Strategies: {better_count} / {iterations}")
        print(f" - P-Value: {p_value:.5f}")
        
        if p_value < 0.05:
            print(">>> SUCCESS: The strategy is Statistically Significant! (Not Luck)")
        else:
            print(">>> WARNING: The strategy might be due to randomness.")
        print("=" * 60)

# =========================================================
# 메인 실행부
# =========================================================
if __name__ == "__main__":
    # 데이터 파일 경로 확인 필요
    csv_file = "loan_status.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found.")
    else:
       
        manager = ExperimentManager(csv_file)
        data = manager.load_and_preprocess(split_ratio=0.8)
        
        #학습 및 예측
        pred_pd, pred_ret = manager.train_and_predict(data)
        
        # 3. Full Test Set Heatmap & Permutation Test 실행
        manager.run_heatmap_full_test_set(data, pred_pd, pred_ret)