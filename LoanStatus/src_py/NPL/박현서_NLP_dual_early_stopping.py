import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# 데이터 칼럼명 수정

df = pd.read_csv("distribution_model_MLP.csv")

# 1. 실제 데이터 컬럼명에 맞게 명단 생성
# 0th_2th_prob, 2th_4th_prob, ..., 98th_100th_prob (총 50개)
prob_cols = [f'{i}th_{i+2}th_prob' for i in range(0, 100, 2)]

# 특성(X)으로 쓸 51개: 확률분포 50개 + 국채수익률 1개
feature_cols = prob_cols + ['Bond']

# 타겟(Y)으로 쓸 1개: 실제 수익률
target_col = ['Return']

# 전체 사용할 컬럼 통합
all_cols = feature_cols + target_col

# 2. 원본 df에서 해당 컬럼들만 추출하여 복사본 생성
# 24GB 램을 활용해 안전하게 .copy()로 복제
df_copy = df[all_cols].copy()


#------------------------------------------------------------------------------

# 최적화 및 학습 (GPU 최적화 버전)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = nn.Sequential(
    nn.Linear(51, 256),
    nn.LeakyReLU(negative_slope=0.01),

    nn.Linear(256, 256),
    nn.LeakyReLU(negative_slope=0.01),

    nn.Linear(256, 256),
    nn.LeakyReLU(negative_slope=0.01),

    nn.Linear(256, 256),
    nn.LeakyReLU(negative_slope=0.01),

    nn.Linear(256, 2), 
    nn.Softmax(dim=-1) 
).to(device)


# 최적화 도구: Adam을 사용하며, 학습률(lr)은 0.001로 설정
optimizer = optim.Adam(model.parameters(), lr=0.001)





# train data set 준비

# 0. 설정 값 정의 
PER_SET = 1000
TRAIN_SIZE = 300000  # 30팀
VAL_SIZE = 200000     # 20팀
TEST_SIZE = 200000    # 20팀

# 실제 팀 수 계산 
train_sets = TRAIN_SIZE // PER_SET  # 30팀
val_sets = VAL_SIZE // PER_SET 
test_sets = TEST_SIZE // PER_SET 

# 1. 인덱스 분리 (중복 방지)
all_indices = np.arange(len(df_copy))


# 학습용 추출
train_idx = np.random.choice(all_indices, size=TRAIN_SIZE, replace=False)

# 학습용 제외한 나머지 np.setdiff1d: 차집합
remaining_idx = np.setdiff1d(all_indices, train_idx)


# 나머지 중 검증용 추출
val_idx = np.random.choice(remaining_idx, size=VAL_SIZE, replace=False)

# 나머지 중 테스트용 추출
remaining_idx = np.setdiff1d(remaining_idx, val_idx)
test_idx = np.random.choice(remaining_idx, size=TEST_SIZE, replace=False)

# 1. 스케일러 준비
scaler = StandardScaler()

# 2. Train 인덱스를 기준으로 '피처(51개)'만 학습
# feature_cols는 아까 정의하신 [50개 확률 + Bond] 입니다.
scaler.fit(df_copy.loc[train_idx, feature_cols])

# 3. df_copy의 피처 부분만 스케일링 된 값으로 교체
df_copy[feature_cols] = scaler.transform(df_copy[feature_cols])

# 2. 3D 텐서 변환 함수 (반복 작업 최적화)
def make_3d_tensor(indices, num_sets):
    # 데이터 추출 및 Reshape (팀수, 팀당인원, 컬럼수)
    data_np = df_copy.iloc[indices].values
    data_3d = data_np.reshape(num_sets, PER_SET, 52)
    
    # 텐서 변환 및 장치 이동
    tensor = torch.tensor(data_3d, dtype=torch.float32).to(device)
    
    # X(피처 51개)와 Y(타겟 1개) 분리
    x = tensor[:, :, :51]
    y = tensor[:, :, 51:]
    return x, y

# 3. 최종 데이터 생성
train_x, train_y = make_3d_tensor(train_idx, train_sets)
val_x, val_y     = make_3d_tensor(val_idx, val_sets)
test_x, test_y   = make_3d_tensor(test_idx, test_sets)


batch_size = 34  # 한 번에 몇 '팀'을 학습할지 
train_dataset = TensorDataset(train_x, train_y)  # train_x: (T, N, 51), train_y: (T, N, 1)
val_dataset   = TensorDataset(val_x, val_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# 결과 확인
print("✅")








def criterion(probs_3d, x_3d, y_3d):
    # x_3d의 마지막 컬럼(-1)이 국채 수익률(Bond1)이라고 가정
    r_tsy_3d = x_3d[:, :, -1]          
    actual_ret_3d = y_3d.squeeze(-1)   # 실제 수익률 (3D로 맞춤)
    
    # 전략 수익률 계산 (모델이 승인하면 실제 수익률, 거절하면 국채 수익률)
    # 끝까지 스무스하게 하는건?
    individual_returns = (probs_3d * actual_ret_3d) + ((1 - probs_3d) * r_tsy_3d)
    
    # 팀별(Batch) 평균 및 표준편차 계산
    mean_ret = individual_returns.mean(dim=1)
    std_ret = individual_returns.std(dim=1) 
    mean_rtsy = r_tsy_3d.mean(dim=1)
    
    # 샤프 지수 계산 (분모에 아주 작은 값 1e-8을 더해 0으로 나누기 방지)
    batch_sharpe = (mean_ret - mean_rtsy) / (std_ret + 1e-8)
    
    # Loss는 마이너스 샤프 지수의 평균 (이 값이 낮아질수록 샤프 지수는 올라감)
    return -batch_sharpe.mean()



def train_with_early_stopping(model, train_loader, val_loader, optimizer, num_epochs=200, patience=30):
    best_val_loss = float('inf')
    best_model_state = None
    warning = 0
    train_loss_history = []
    val_loss_history = []

    print("🚀 학습 및 실시간 검증을 시작합니다...")

    for epoch in range(num_epochs):
        # --- 1. 학습 단계 (Train Step) ---
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device)  # (B, N, 51)
            by = by.to(device)  # (B, N, 1)
            N = bx.size(1)  # 팀당 인원 수 (하드코딩 금지)

        # Forward: 데이터를 모델에 입력하고 승인 확률 추출
        probs_flat = model(bx.reshape(-1, bx.size(-1)))
        probs_3d = probs_flat.view(-1, N, 2)[:, :, 1]
        
        # Loss 계산: 샤프 지수 기반의 손실함수 최적화
        loss = criterion(probs_3d, bx, by)
        
        # Optimization: 가중치 수정 실행
        optimizer.zero_grad() # 기울기 초기화
        loss.backward()    # 역전파 실행
        optimizer.step()     # 가중치 업데이트
        
        train_loss_history.append(loss.item())


        # --- 2. 검증 단계 (Validation Step) ---
        model.eval()
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)

                N = vx.size(1)
                # 검증용 데이터(20팀)로 현재 모델 실력 측정
                v_probs_flat = model(vx.reshape(-1, vx.size(-1)))
                v_probs_3d = v_probs_flat.view(-1, N, 2)[:, :, 1]
                v_loss = criterion(v_probs_3d, vx, vy)
            
                # 매 에폭의 결과(Loss) 저장
        
                val_loss_history.append(v_loss.item())

        print(f"Epoch [{epoch+1:3d}] | Train Loss: {loss.item():.4f} | Val Loss: {v_loss.item():.4f}")


        # 최적 모델 저장 (가장 낮은 Val Loss 기록 시)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            warning = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            warning +=1
            
        if warning >= patience:
            model.load_state_dict(best_model_state)
            print(' 학습 종료 ')
            break

   
            

    return train_loss_history, val_loss_history, best_val_loss







# test

# 1. 데이터를 펴고 모델을 통과시켜 복구하는 함수
def get_model_predictions(test_x_3d, test_y_3d):
    model.eval() # 평가모드
    with torch.no_grad(): # 기울기 계산 금지 - 메모리, 속도
        # (20, 10000, 51) -> (20000, 51)로 펴기
        x_flat = test_x_3d.view(-1, 51)
        
        # 모델 통과 (결과: 20000,)
        probs_flat = model(x_flat)
        
        # 원래 구조로 복구 (결과: 20, 1000)
        probs_3d_full = probs_flat.view(-1, 1000,2)
        probs_3d = probs_3d_full[:, :, 1] # 승인 확률만 추출

        r_tsy_3d = test_x_3d[:, :, -1] # 국채 수익률 분리
        actual_ret_3d = test_y_3d.squeeze(-1) # 실제 수익률

    # dim=-1은 마지막 차원(2개 점수) 중 큰 인덱스를 고름
    #        = (probs_3d >= 0.5).float() 
    decisions = torch.argmax(probs_3d_full, dim=-1).float() # 결과: (20, 1000)
    
    # 3. 개별 수익률 결정
    # 승인(1)인 사람은 actual_loan_ret을, 미승인(0)인 사람은 r_tsy를 가짐
    # 개개인의 수익률 (20, 1000)행렬을 결과값으로
    individual_returns = (decisions * actual_ret_3d) + ((1 - decisions) * r_tsy_3d)
    
    # 4. 3000명의 평균 수익률과 표준편차 계산
    mean_ret = individual_returns.mean(dim=1)   
    std_ret = individual_returns.std(dim=1)     
    mean_rtsy = r_tsy_3d.mean(dim=1)
    # 5. Sharpe Ratio 계산 => 평균 수익률 - 평균 국채 수익률 평균(수익률-국채수익률)
    final_sharpe = (mean_ret - mean_rtsy)/ (std_ret + 1e-8)
    
    return final_sharpe, decisions, mean_ret, mean_rtsy, std_ret






# 1. 모델 학습 실행
print("\n" + "="*50)
print(" [Step 1] 모델 학습(Training) 시작 (Early Stopping 적용)...")
print("="*50)
train_loss_history, val_loss_history, loss_history = train_with_early_stopping(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=1000,
    patience=30
)


val_sharpe_hist = [-x for x in val_loss_history]

# 3. 최종 결과 출력
print("\n" + "="*30)
avg_sharpe = np.mean(val_sharpe_hist)
print(f"📊 전체 에폭 검증 샤프 지수 평균: {avg_sharpe:.4f}")
print(f"🏆 역대 최고 검증 샤프 지수: {max(val_sharpe_hist):.4f}")
print("="*30)

print(">> 학습 완료!")

# 2. 테스트 데이터 결과 도출
print("\n" + "="*50)
print(" [Step 2] 테스트 데이터(Test Data) 평가 시작...")
print("="*50)
sharpe_results, decisions, mean_ret, mean_rtsy, std_ret = get_model_predictions(test_x, test_y)



# 3. 상세 수치 계산 (텐서를 넘파이로 변환하여 계산)
avg_sharpe = sharpe_results.mean().item()
max_sharpe = sharpe_results.max().item()
min_sharpe = sharpe_results.min().item()

total_test_count = decisions.numel()
approved_count = int(decisions.sum().item())
approval_rate = (approved_count / total_test_count) * 100
approved_returns = test_y.squeeze(-1).cpu()[decisions.cpu() == 1]


avg_approved_return = approved_returns.mean().item() * 100


# 결과 출력
print("\n" + "*"*20 + " [ 최종 성적표 ] " + "*"*20)
print(f"▶ 전체 테스트 인원      : {total_test_count:,} 명")
print(f"▶ 모델이 승인한 인원    : {approved_count:,} 명")
print(f"▶ 최종 승인율          : {approval_rate:.2f} %")
print("-" * 51)
print(f"▶ 평균 샤프 지수(Sharpe) : {avg_sharpe:.4f}")
print(f"▶ 최고 팀 샤프 지수     : {max_sharpe:.4f}")
print(f"▶ 최저 팀 샤프 지수     : {min_sharpe:.4f}")
print("-" * 51)
print(f"▶ 승인된 대출의 평균 수익률 : {avg_approved_return:.2f} %")
print("▶ 전략 평균수익(mean_ret) 평균:", mean_ret.mean().item())
print("▶ 국채 평균수익(mean_rtsy) 평균:", mean_rtsy.mean().item())
print("▶ 전략 표준편차(std_ret) 평균:", std_ret.mean().item())
print("▶ 샤프(final_sharpe) 평균:", sharpe_results.mean().item())
print("*"*54)




print("승인 표본 수:", approved_returns.numel())
print("승인 평균(원값):", approved_returns.mean().item())
print("승인 평균(%):", approved_returns.mean().item() * 100)


