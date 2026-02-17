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

# =========================================================
# CsvLoader 
# =========================================================
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

# ========================================================= 
# Initialization 
# =========================================================
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

# =========================================================
# [Mode 1] Reliability Check
# =========================================================
    def run_reliability_check_bootstrap(self, data, pred_pd, pred_ret):
        print("\n" + "="*60)
        print(">>> [Mode 1] Reliability Check (Bootstrapping)")
        print("="*60)

        NUM_SIMULATIONS = 1000
        SAMPLE_SIZE = 10000
        PD_TH = 0.20
        RET_TH = 0.075

        print(f"Settings: Iter={NUM_SIMULATIONS}, Sample={SAMPLE_SIZE}")
        print(f"Target Threshold: PD < {PD_TH}, Ret > {RET_TH}")

        results = []
        
        # 전체 데이터 인덱스 준비
        total_size = len(data['actual_returns'])
        indices = np.arange(total_size)
        
        # 실제 데이터 참조
        full_actual = data['actual_returns']
        full_bond = data['bond_yields']

        print(f"Running {NUM_SIMULATIONS} simulations...")

        for s in tqdm(range(1, NUM_SIMULATIONS + 1), desc="Bootstrapping"):
            # 1. 셔플 및 샘플링 (Subsetting)
            np.random.shuffle(indices)
            selected_idx = indices[:SAMPLE_SIZE] # 앞부분 10,000개 추출

            # 2. 샘플 데이터 생성
            sub_pd = pred_pd[selected_idx]
            sub_ret = pred_ret[selected_idx]
            sub_actual = full_actual[selected_idx]
            sub_bond = full_bond[selected_idx]

            # 3. 필터링 (Vectorized)
            mask = (sub_pd < PD_TH) & (sub_ret > RET_TH)
            count = np.sum(mask)

            mean_val = 0.0
            std_dev = 0.0
            sharpe = 0.0
            rate = (count / SAMPLE_SIZE) * 100.0

            if count >= 10:
                # 4. 샤프지수 계산 (Helper 함수 재사용)
                # sub_actual의 길이는 10,000이므로 분모는 자동으로 10,000이 됨
                sharpe = calculate_sharpe_ratio(sub_actual, sub_bond, mask)

                # 5. 통계치 계산 (CSV 기록용)
                # 거절된 건은 0으로 처리 (초과수익)
                excess = np.where(mask, sub_actual - sub_bond, 0.0)
                
                # Mean: 합계 / 10,000
                mean_val = np.mean(excess)
                
                # StdDev: 합계 / 9,999 (N-1, Unbiased)
                # ddof=1 옵션이 N-1 계산을 수행함
                std_dev = np.std(excess, ddof=1)

            results.append({
                'Sim_ID': s,
                'Sharpe_Ratio': sharpe,
                'Mean_Excess_Return': mean_val,
                'Std_Dev': std_dev,
                'Approved_Count': count,
                'Approved_Rate': rate
            })

        # CSV 저장
        df_res = pd.DataFrame(results)
        df_res.to_csv('bootstrapping_detailed_results_python.csv', index=False)
        
        # 요약 통계 출력
        print("-" * 60)
        print(f"Bootstrap Summary ({NUM_SIMULATIONS} runs):")
        print(f" - Avg Sharpe: {df_res['Sharpe_Ratio'].mean():.4f}")
        print(f" - Min Sharpe: {df_res['Sharpe_Ratio'].min():.4f}")
        print(f" - Max Sharpe: {df_res['Sharpe_Ratio'].max():.4f}")
        print(">>> Results saved to 'bootstrapping_detailed_results_python.csv'")

# =========================================================
# [Mode 2] Auto Grid Search (Hyperparams + Thresholds)
# =========================================================
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
# [Mode 0] Auto Grid Search (Hyperparams + Thresholds)
# =========================================================
    def run_grid_search_auto(self, data):
        print("\n" + "="*60)
        print(">>> [Mode 0] Automatic Grid Search (Params + Thresholds)")
        print("="*60)

        # 1. 탐색할 하이퍼파라미터 그리드 정의
        candidate_depths = [5, 7]
        candidate_etas = [0.05, 0.01]
        
        # 조합 생성 (C++ GenerateGrid와 동일 로직)
        cls_configs = []
        reg_configs = []

        # Grid 생성 함수 (내부 헬퍼)
        def generate_config(depths, etas):
            configs = []
            for d in depths:
                for e in etas:
                    # Eta에 따라 Round 자동 계산 (C++ 로직: 20 / eta)
                    rounds = int(20.0 / e)
                    rounds = max(100, min(rounds, 3000)) # 안전장치
                    configs.append({'depth': d, 'eta': e, 'rounds': rounds})
            return configs

        cls_configs = generate_config(candidate_depths, candidate_etas)
        reg_configs = generate_config(candidate_depths, candidate_etas)

        total_iter = len(cls_configs) * len(reg_configs)
        print(f">>> Total Model Combinations: {total_iter}")
        print(">>> Output Log: 'grid_search_auto_python.csv'\n")

        # 결과 저장 리스트
        results = []
        best_result = {
            'sharpe': -999.0, 
            'config': None, 
            'metrics': None
        }

        current_iter = 0

        # 2. 이중 루프로 모델 탐색
        for c_conf in cls_configs:
            for r_conf in reg_configs:
                current_iter += 1
                
                # 2-1. 현재 설정으로 모델 학습 및 예측
                # (기존 train_and_predict를 재사용하되, 파라미터 오버라이딩 적용)
                
                # 분류 모델 설정 업데이트
                curr_cls_opts = CLS_CONFIG.copy()
                curr_cls_opts['max_depth'] = c_conf['depth']
                curr_cls_opts['eta'] = c_conf['eta']
                
                # 회귀 모델 설정 업데이트
                curr_reg_opts = REG_CONFIG.copy()
                curr_reg_opts['max_depth'] = r_conf['depth']
                curr_reg_opts['eta'] = r_conf['eta']

                # 학습 수행
                # (진행바가 너무 많아지므로 verbose_eval=False 처리 권장)
                bst_cls = xgb.train(curr_cls_opts, data['dtrain_cls'], num_boost_round=c_conf['rounds'])
                bst_reg = xgb.train(curr_reg_opts, data['dtrain_reg'], num_boost_round=r_conf['rounds'])

                pred_pd = bst_cls.predict(data['dtest_cls'])
                pred_ret = bst_reg.predict(data['dtest_reg'])

                # 2-2. 예측값으로 최적 임계값 탐색 (Helper 호출)
                metrics = self._find_best_thresholds(data, pred_pd, pred_ret)

                # 2-3. 로그 출력 및 최고 기록 갱신
                is_best = False
                if metrics['sharpe'] > best_result['sharpe']:
                    best_result['sharpe'] = metrics['sharpe']
                    best_result['config'] = (c_conf, r_conf)
                    best_result['metrics'] = metrics
                    is_best = True

                log_msg = (
                    f"[{current_iter}/{total_iter}] "
                    f"Cls(d{c_conf['depth']} e{c_conf['eta']}) "
                    f"Reg(d{r_conf['depth']} e{r_conf['eta']}) | "
                    f"BestTh({metrics['best_pd']:.2f}/{metrics['best_ret']:.2f}) "
                    f"-> Sharpe: {metrics['sharpe']:.5f}"
                )
                if is_best:
                    log_msg += " [★ NEW BEST]"
                print(log_msg)

                # 결과 저장
                results.append({
                    'Iter': current_iter,
                    'Cls_Depth': c_conf['depth'], 'Cls_Eta': c_conf['eta'],
                    'Reg_Depth': r_conf['depth'], 'Reg_Eta': r_conf['eta'],
                    'Best_PD_Thresh': metrics['best_pd'],
                    'Best_Ret_Thresh': metrics['best_ret'],
                    'Approved_Cnt': metrics['count'],
                    'Avg_Return': metrics['avg_ret'],
                    'Avg_PD': metrics['avg_pd'],
                    'Sharpe_Ratio': metrics['sharpe']
                })

        # CSV 저장
        res_df = pd.DataFrame(results)
        res_df.to_csv('grid_search_auto_python.csv', index=False)
        print(f"\n>>> Grid Search Complete. Best Sharpe: {best_result['sharpe']:.5f}")


    def find_best_thresholds(self, data, pred_pd, pred_ret):
        """
        [Internal Helper] 주어진 예측값에 대해 최적의 임계값을 찾는 함수
        """
        best_metrics = {
            'sharpe': -999.0, 'best_pd': 0.0, 'best_ret': 0.0,
            'count': 0, 'avg_ret': 0.0, 'avg_pd': 0.0
        }

        # 탐색 후보군 (C++과 동일하게 설정)
        pd_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        ret_candidates = [0.02, 0.04, 0.05, 0.06, 0.07, 0.08]

        actual_returns = data['actual_returns']
        bond_yields = data['bond_yields']
        n = len(actual_returns)

        for pd_th in pd_candidates:
            for ret_th in ret_candidates:
                # Numpy 마스킹 (Vectorized)
                mask = (pred_pd < pd_th) & (pred_ret > ret_th)
                count = np.sum(mask)

                if count < 10:
                    continue

                # 샤프지수 계산 (Standalone 함수 호출)
                current_sharpe = calculate_sharpe_ratio(actual_returns, bond_yields, mask)

                if current_sharpe > best_metrics['sharpe']:
                    # 통계치 계산
                    excess = np.where(mask, actual_returns - bond_yields, 0.0)
                    # 승인된 건들에 대한 평균 수익률/부도율 계산
                    avg_ret = np.mean(actual_returns[mask])
                    avg_pd = np.mean(pred_pd[mask])

                    best_metrics.update({
                        'sharpe': current_sharpe,
                        'best_pd': pd_th,
                        'best_ret': ret_th,
                        'count': count,
                        'avg_ret': avg_ret,
                        'avg_pd': avg_pd
                    })
        
        return best_metrics
    
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