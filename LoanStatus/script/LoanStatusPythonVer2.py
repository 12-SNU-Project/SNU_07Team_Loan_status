import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split

CLS_CONFIG = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 7,
    'eta': 0.05,
    'min_child_weight': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'nthread': os.cpu_count() // 2
}

REG_CONFIG = {
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae',
    'max_depth': 5,
    'eta': 0.05,
    'min_child_weight': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'nthread': os.cpu_count() // 2
}

def calculate_sharpe_ratio(actual_returns, bond_yields, approval_mask):
    n = len(actual_returns)
    if n <= 1:
        return 0.0

    # 1. 초과 수익률 계산 (승인: 수익-국채, 거절: 0)
    excess_returns = np.where(approval_mask, actual_returns - bond_yields, 0.0)

    # 2. 평균 및 표준편차 계산 (ddof=1 -> N-1로 나눔)
    mean_excess = np.mean(excess_returns)
    std_dev = np.std(excess_returns, ddof=1)

    # 3. 샤프 지수 반환 
    if std_dev < 1e-9:
        return 0.0
    
    return mean_excess / std_dev

class ExperimentManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.ignore_cols = [
            "Actual_term", "total_pymnt", "last_pymnt_amnt", "내부수익률",
            "loan_status", "Return", "Bond"
        ]
        self.target_col = "loan_status"

    # 데이터 로드 및 6:2:2 분할 (Shuffle 필수)
    def load_and_split_622(self):
        print(">>> [1/3] Loading Data & Splitting (6:2:2 with Shuffle)...")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # 한글 깨짐 방지
        df = pd.read_csv(self.file_path, encoding='utf-8-sig')
        
        # Features & Labels 준비
        X = df.drop(columns=self.ignore_cols + ['Bond'], errors='ignore')
        feature_names = X.columns.tolist()
        X = X.values # Numpy 배열 변환
        
        y_cls = df[self.target_col].values
        y_reg = df['Return'].values
        
        # 샤프지수 계산용 원본 데이터
        raw_returns = df['Return'].values
        raw_bonds = df['Bond'].values

     
        #Scikit-Learn을 이용한 6:2:2 분할
        # 1. 전체를 80(Train+Val) : 20(Test) 으로 분할
        X_tv, X_test, y_c_tv, y_c_test, y_r_tv, y_r_test, ret_tv, ret_test, bond_tv, bond_test = train_test_split(
            X, y_cls, y_reg, raw_returns, raw_bonds, 
            test_size=0.2, random_state=42, shuffle=True
        )

        # 2. 남은 80을 다시 75(Train) : 25(Val) 로 분할
        # (전체의 0.8 * 0.25 = 0.2 이므로 결과적으로 전체의 20%가 Val이 됨)
        X_train, X_val, y_c_train, y_c_val, y_r_train, y_r_val, ret_train, ret_val, bond_train, bond_val = train_test_split(
            X_tv, y_c_tv, y_r_tv, ret_tv, bond_tv,
            test_size=0.25, random_state=42, shuffle=True
        )

        print(f"Split Result -> Train: {len(X_train)} / Val: {len(X_val)} / Test: {len(X_test)}")

        # DataPack (DMatrix 미리 생성하여 속도 최적화)
        data_pack = {
            'train': {
                'dcls': xgb.DMatrix(X_train, label=y_c_train, feature_names=feature_names),
                'dreg': xgb.DMatrix(X_train, label=y_r_train, feature_names=feature_names),
                'rows': len(X_train)
            },
            'val': {
                'dcls': xgb.DMatrix(X_val, feature_names=feature_names),
                'dreg': xgb.DMatrix(X_val, feature_names=feature_names),
                'actual_returns': ret_val,
                'bond_yields': bond_val,
                'rows': len(X_val)
            },
            'test': {
                'dcls': xgb.DMatrix(X_test, feature_names=feature_names),
                'dreg': xgb.DMatrix(X_test, feature_names=feature_names),
                'actual_returns': ret_test,
                'bond_yields': bond_test,
                'rows': len(X_test)
            }
        }
        return data_pack

    # Grid Search (Train 학습 -> Val 검증)
    def run_grid_search_auto(self, data_pack):
        print("\n" + "="*60)
        print(">>> [Mode 0] Grid Search on Validation Set")
        print(">>> Training on 60% (Train), Evaluating on 20% (Val)")
        print("="*60)

       
        candidate_depths = [5, 7]
        candidate_etas = [0.05, 0.01]

        configs = []
        for d in candidate_depths:
            for e in candidate_etas:
                rounds = int(20.0 / e)
                rounds = max(100, min(rounds, 3000))
                configs.append({'depth': d, 'eta': e, 'rounds': rounds})

        print(f">>> Total Model Combinations: {len(configs) * len(configs)}")

        best_result = {
            'sharpe': -999.0, 
            'cls_config': None, 'reg_config': None, 'metrics': None
        }

        # 이중 루프 (Classification Config x Regression Config)
        iter_count = 0
        for c_conf in configs:
            for r_conf in configs:
                iter_count += 1
                
                # 1. 모델 설정
                curr_cls_opts = CLS_CONFIG.copy()
                curr_cls_opts.update({'max_depth': c_conf['depth'], 'eta': c_conf['eta']})
                
                curr_reg_opts = REG_CONFIG.copy()
                curr_reg_opts.update({'max_depth': r_conf['depth'], 'eta': r_conf['eta']})

                # 2. Train Set(60%)으로 학습
                bst_cls = xgb.train(curr_cls_opts, data_pack['train']['dcls'], num_boost_round=c_conf['rounds'])
                bst_reg = xgb.train(curr_reg_opts, data_pack['train']['dreg'], num_boost_round=r_conf['rounds'])

                # 3. Validation Set(20%)으로 예측
                pred_pd_val = bst_cls.predict(data_pack['val']['dcls'])
                pred_ret_val = bst_reg.predict(data_pack['val']['dreg'])

                # 4. Validation Set 기준 최적 임계값 탐색
                metrics = self._find_best_thresholds(
                    pred_pd_val, pred_ret_val, 
                    data_pack['val']['actual_returns'], 
                    data_pack['val']['bond_yields']
                )

                # 5. 최고 기록 갱신 확인
                is_best = False
                if metrics['sharpe'] > best_result['sharpe']:
                    best_result['sharpe'] = metrics['sharpe']
                    best_result['cls_config'] = c_conf
                    best_result['reg_config'] = r_conf
                    best_result['metrics'] = metrics
                    is_best = True

                # 로그 출력
                log_msg = (
                    f"[{iter_count}] "
                    f"C(d{c_conf['depth']} e{c_conf['eta']}) R(d{r_conf['depth']} e{r_conf['eta']}) | "
                    f"Val-Sharpe: {metrics['sharpe']:.4f}"
                )
                if is_best: log_msg += " [★ NEW BEST]"
                print(log_msg)

        print(f"\n>>> Best Validation Sharpe: {best_result['sharpe']:.4f}")
        return best_result

    # ---------------------------------------------------------
    # [Step 3] Standard Validation (Train 학습 -> Test 최종 검증)
    # ---------------------------------------------------------
    def run_standard_validation(self, data_pack, best_result):
        print("\n" + "="*60)
        print(">>> [Mode 3] Standard Validation (6:2:2 Split)")
        print(">>> Using Best Config & Thresholds from Grid Search")
        print("="*60)

        # Grid Search에서 찾은 최적 설정들
        c_conf = best_result['cls_config']
        r_conf = best_result['reg_config']
        metrics = best_result['metrics']
        
        best_pd = metrics['best_pd']
        best_ret = metrics['best_ret']

        print(f"[Best Config] CLS: d{c_conf['depth']}/e{c_conf['eta']}, REG: d{r_conf['depth']}/e{r_conf['eta']}")
        print(f"[Best Threshold] PD < {best_pd:.2f}, Return > {best_ret:.3f}")

        # 1. Train Set(60%)으로 모델 재학습 (파라미터 고정)
        print(f">>> [Step 1] Retraining on Train Set ({data_pack['train']['rows']} rows)...")
        
        final_cls_opts = CLS_CONFIG.copy()
        final_cls_opts.update({'max_depth': c_conf['depth'], 'eta': c_conf['eta']})
        
        final_reg_opts = REG_CONFIG.copy()
        final_reg_opts.update({'max_depth': r_conf['depth'], 'eta': r_conf['eta']})

        bst_cls = xgb.train(final_cls_opts, data_pack['train']['dcls'], num_boost_round=c_conf['rounds'])
        bst_reg = xgb.train(final_reg_opts, data_pack['train']['dreg'], num_boost_round=r_conf['rounds'])

        # 2. Test Set(20%)으로 최종 예측
        print(f">>> [Step 2] Final Verification on Test Set ({data_pack['test']['rows']} rows)...")
        
        pred_pd_test = bst_cls.predict(data_pack['test']['dcls'])
        pred_ret_test = bst_reg.predict(data_pack['test']['dreg'])

        # 3. Validation에서 찾은 임계값 적용
        final_mask = (pred_pd_test < best_pd) & (pred_ret_test > best_ret)
        approved_count = np.sum(final_mask)
        total_test = len(final_mask)

        # 4. 최종 샤프지수 계산 (안전장치 포함)
        final_sharpe = 0.0
        if approved_count >= 10:
            final_sharpe = calculate_sharpe_ratio(
                data_pack['test']['actual_returns'],
                data_pack['test']['bond_yields'],
                final_mask
            )
        else:
            print(f">>> [WARNING] Not enough approved samples ({approved_count} < 10). Sharpe set to 0.0.")

        # 5. 결과 출력
        print(f"\n>>> [Final Result]")
        print(f"1. Validation Sharpe : {best_result['sharpe']:.4f}")
        print(f"2. Test Set Sharpe   : {final_sharpe:.4f}")
        print(f"3. Approved Count    : {approved_count} / {total_test} ({approved_count/total_test*100:.2f}%)")

        diff = abs(final_sharpe - best_result['sharpe'])
        if diff < 0.5:
            print(">>> SUCCESS: Model is Robust! (Val & Test scores are similar)")
        else:
            print(">>> WARNING: Large gap detected. Check for Overfitting.")

        # 6. (Bonus) 순열 검정 수행
        self.perform_permutation_test(
            pred_pd_test, pred_ret_test,
            data_pack['test']['actual_returns'],
            data_pack['test']['bond_yields'],
            best_pd, best_ret
        )

    # ---------------------------------------------------------
    # [Helper] 내부 함수들
    # ---------------------------------------------------------
    def _find_best_thresholds(self, pred_pd, pred_ret, actuals, bonds):
        """ Validation Set을 탐색하여 최적의 임계값 반환 """
        best_metrics = {'sharpe': -999.0, 'best_pd': 0.0, 'best_ret': 0.0, 'count': 0}

        pd_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        ret_candidates = [0.02, 0.04, 0.05, 0.06, 0.07, 0.08]

        for pd_th in pd_candidates:
            for ret_th in ret_candidates:
                mask = (pred_pd < pd_th) & (pred_ret > ret_th)
                count = np.sum(mask)
                
                if count < 10: continue

                s = calculate_sharpe_ratio(actuals, bonds, mask)
                if s > best_metrics['sharpe']:
                    best_metrics['sharpe'] = s
                    best_metrics['best_pd'] = pd_th
                    best_metrics['best_ret'] = ret_th
                    best_metrics['count'] = count
        
        return best_metrics

    def perform_permutation_test(self, pred_pd, pred_ret, actual_ret, bond_yields, best_pd, best_ret, iterations=1000):
        print("\n" + "="*60)
        print(">>> [Bonus] Permutation Test on Test Set")
        print("="*60)
        
        # 고정된 승인 마스크 (모델의 판단)
        fixed_mask = (pred_pd < best_pd) & (pred_ret > best_ret)
        original_sharpe = calculate_sharpe_ratio(actual_ret, bond_yields, fixed_mask)
        
        print(f"Original Test Sharpe: {original_sharpe:.4f}")
        print(f"Running {iterations} permutations...")

        better_count = 0
        indices = np.arange(len(actual_ret))
        shuffled_ret = actual_ret.copy()
        shuffled_bond = bond_yields.copy()

        for _ in range(iterations):
            np.random.shuffle(indices)
            # 정답지(수익률, 국채)만 뒤섞음
            random_sharpe = calculate_sharpe_ratio(
                shuffled_ret[indices], 
                shuffled_bond[indices], 
                fixed_mask
            )
            if random_sharpe >= original_sharpe:
                better_count += 1

        p_value = (better_count + 1) / (iterations + 1)
        print(f"Result -> Better: {better_count}/{iterations}, P-Value: {p_value:.5f}")
        
        if p_value < 0.05:
            print(">>> SUCCESS: Statistically Significant!")
        else:
            print(">>> WARNING: Might be Luck.")

# =========================================================
# 메인 실행부
# =========================================================
if __name__ == "__main__":
    csv_file = "loan_status.csv"
    
    if os.path.exists(csv_file):
        manager = ExperimentManager(csv_file)
        
        # 1. 데이터 로드 및 6:2:2 분할 (Shuffle)
        data_pack = manager.load_and_split_622()
        
        # 2. Grid Search (Train -> Val)
        best_result = manager.run_grid_search_auto(data_pack)
        
        # 3. Standard Validation (Train -> Test using Best Config)
        manager.run_standard_validation(data_pack, best_result)
        
    else:
        print(f"Error: '{csv_file}' not found.")