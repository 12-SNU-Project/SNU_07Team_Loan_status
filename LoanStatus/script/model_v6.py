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
            "loan_status", "Return", "Bond", "초과수익률" # Bond는 별도 추출 후 feature에선 제외
        ]
        self.target_col = "loan_status"

# =========================================================
# CsvLoader 
# =========================================================

    def load_and_preprocess(self, split_ratio=0.8, val_ratio=0.2):
        print(">>> [1/4] Data Loading & Preprocessing...")
        
        df = pd.read_csv(self.file_path, encoding='utf-8-sig')

        total_rows = len(df)
        split_point = int(total_rows * split_ratio)   # train+val 끝, 뒤는 test
        test_size = total_rows - split_point

        # 1. Target & Returns & Bond 분리
        y_cls = df[self.target_col].values
        y_reg = df['Return'].values
        bond_yields = df['Bond'].values

        # 2. Features 분리 (Ignore List 제거)
        X = df.drop(columns=self.ignore_cols + ['Bond'], errors='ignore')
        feature_names = X.columns.tolist()
        X = X.values

        # 3. (train+val) / test Split (Sequential)
        X_trval, X_test = X[:split_point], X[split_point:]
        y_cls_trval, _ = y_cls[:split_point], y_cls[split_point:]
        y_reg_trval, _ = y_reg[:split_point], y_reg[split_point:]

        # test 실제 데이터 (평가용)
        test_actual_returns = y_reg[split_point:]
        test_bond_yields = bond_yields[split_point:]

        # 4. train / val Split (trval 내부 sequential)
        trval_rows = len(X_trval)
        val_point = int(trval_rows * (1 - val_ratio))  # train 끝, 뒤가 val

        X_train, X_val = X_trval[:val_point], X_trval[val_point:]
        y_cls_train, _y_cls_val = y_cls_trval[:val_point], y_cls_trval[val_point:]
        y_reg_train, _y_reg_val = y_reg_trval[:val_point], y_reg_trval[val_point:]

        # val 실제 데이터 (threshold 탐색용)
        val_actual_returns = y_reg_trval[val_point:]
        val_bond_yields = bond_yields[:split_point][val_point:]

        # 5. XGBoost DMatrix 생성
        dtrain_cls = xgb.DMatrix(X_train, label=y_cls_train, feature_names=feature_names)
        dtest_cls  = xgb.DMatrix(X_test, feature_names=feature_names)

        dtrain_reg = xgb.DMatrix(X_train, label=y_reg_train, feature_names=feature_names)
        dtest_reg  = xgb.DMatrix(X_test, feature_names=feature_names)

        # (추가) val 예측을 위해 dval도 만들어둠
        dval_cls = xgb.DMatrix(X_val, feature_names=feature_names)
        dval_reg = xgb.DMatrix(X_val, feature_names=feature_names)

        return {
            'dtrain_cls': dtrain_cls, 'dtest_cls': dtest_cls, 'dval_cls': dval_cls,
            'dtrain_reg': dtrain_reg, 'dtest_reg': dtest_reg, 'dval_reg': dval_reg,

            # (기존) test 평가용 → 이름만 명확히
            'test_actual_returns': test_actual_returns,
            'test_bond_yields': test_bond_yields,
            'test_size': test_size,

            # (추가) val 탐색용
            'val_actual_returns': val_actual_returns,
            'val_bond_yields': val_bond_yields,
            'val_size': len(val_actual_returns)
        }


    # ========================================================= 
    # Initialization 
    # =========================================================

    def train_and_predict(self, data):
        print(">>> [2/4] Training Models (Classification & Regression)...")
        
        bst_cls = xgb.train(CLS_CONFIG, data['dtrain_cls'], num_boost_round=NUM_ROUNDS)
        bst_reg = xgb.train(REG_CONFIG, data['dtrain_reg'], num_boost_round=NUM_ROUNDS)

        print(">>> [3/4] Predicting Val & Test Set...")
        # (추가) val 예측
        pred_pd_val  = bst_cls.predict(data['dval_cls'])
        pred_ret_val = bst_reg.predict(data['dval_reg'])

        # (기존) test 예측
        pred_pd_test  = bst_cls.predict(data['dtest_cls'])
        pred_ret_test = bst_reg.predict(data['dtest_reg'])

        return pred_pd_val, pred_ret_val, pred_pd_test, pred_ret_test

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

    def run_heatmap_full_test_set(self, data, pred_pd, pred_ret,
                              actual_returns, bond_yields, denom_size,
                              run_permutation=False):
        print("\n" + "="*60)
        print(">>> [Mode 4] Full Set Heatmap (Python Version)")
        print("="*60)

        results = []
        best_sharpe = -999.0
        best_params = {'pd_th': 0.0, 'ret_th': 0.0}

        pd_thresholds = [i * 0.01 for i in range(1, 41)]   # 0.01 ~ 0.40
        ret_thresholds = [j * 0.005 for j in range(1, 41)] # 0.005 ~ 0.200

        total_iter = len(pd_thresholds) * len(ret_thresholds)
        print(f"Scanning {total_iter} combinations...")

        for pd_th in tqdm(pd_thresholds, desc="Grid Search"):
            for ret_th in ret_thresholds:
                approval_mask = (pred_pd < pd_th) & (pred_ret > ret_th)
                count = np.sum(approval_mask)

                mean = 0.0
                std_dev = 0.0
                sharpe = 0.0
                rate = (count / denom_size) * 100.0   # ✅ 분모를 denom_size로

                if count >= 10:
                    sharpe = calculate_sharpe_ratio(
                        actual_returns, 
                        bond_yields, 
                        approval_mask
                    )

                    excess = np.where(approval_mask, actual_returns - bond_yields, 0.0)
                    mean = np.mean(excess)
                    std_dev = np.std(excess, ddof=1)

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params['pd_th'] = pd_th
                        best_params['ret_th'] = ret_th

                results.append({
                    'PD_Threshold': pd_th,
                    'Return_Threshold': ret_th,
                    'Sharpe_Ratio': sharpe,
                    'Mean_Excess_Return': mean,
                    'Std_Dev': std_dev,
                    'Approved_Count': count,
                    'Approved_Rate': rate
                })

        # CSV 저장(원하면 파일명만 그대로 두거나 val/test로 나눠도 됨)
        res_df = pd.DataFrame(results)
        res_df.to_csv('heatmap_full_set_python.csv', index=False)
        print(">>> Heatmap saved to 'heatmap_full_set_python.csv'")

        # (변경) permutation은 여기서 “옵션”
        if run_permutation and best_sharpe > -900:
            self.perform_permutation_test(
                pred_pd, pred_ret,
                actual_returns, bond_yields,
                best_params['pd_th'], best_params['ret_th']
            )

        # ✅ best를 밖에서 test 평가에 쓰도록 반환
        return best_params, best_sharpe


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
    csv_file = "NewData.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found.")
    else:
        manager = ExperimentManager(csv_file)
        data = manager.load_and_preprocess(split_ratio=0.8, val_ratio=0.2)

        # 학습 및 예측 (val/test 둘 다)
        pred_pd_val, pred_ret_val, pred_pd_test, pred_ret_test = manager.train_and_predict(data)

        # ✅ 1) threshold 탐색은 validation에서만
        best_params, best_val_sharpe = manager.run_heatmap_full_test_set(
            data,
            pred_pd_val, pred_ret_val,
            data['val_actual_returns'], data['val_bond_yields'],
            data['val_size'],
            run_permutation=False
        )
        print(f"[VAL] Best Sharpe={best_val_sharpe:.5f}, params={best_params}")

        # ✅ 2) test는 고정 threshold로 최종 평가(딱 1번)
        fixed_mask_test = (pred_pd_test < best_params['pd_th']) & (pred_ret_test > best_params['ret_th'])
        test_sharpe = calculate_sharpe_ratio(
            data['test_actual_returns'],
            data['test_bond_yields'],
            fixed_mask_test
        )
        print(f"[TEST] Sharpe={test_sharpe:.5f}, Approved={fixed_mask_test.sum()} / {len(fixed_mask_test)}")

        # ✅ 3) permutation test는 test에서 (고정 threshold로)
        manager.perform_permutation_test(
            pred_pd_test, pred_ret_test,
            data['test_actual_returns'], data['test_bond_yields'],
            best_params['pd_th'], best_params['ret_th']
        )
