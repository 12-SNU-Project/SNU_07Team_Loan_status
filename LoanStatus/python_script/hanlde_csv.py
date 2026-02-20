import pandas as pd

file_path = 'Data1.csv'  

# 2. CSV 파일 불러오기
df = pd.read_csv(file_path)

# 3. 'bLoss2', 'bLoss3' 열 제거하기
df.drop(columns=['bLoss2', 'bLoss3'], inplace=True)

# 4. 수정된 데이터를 CSV 파일로 다시 저장 (원본 덮어쓰기)
df.to_csv(file_path, index=False)

print("bLoss2, bLoss3 열이 제거되고 파일이 성공적으로 저장되었습니다.")