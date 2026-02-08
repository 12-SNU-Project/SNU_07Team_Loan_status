#include "pch.h" 
#include "CsvLoader.h"
#include "ExperimentManager.h"

using namespace std::string_literals;

// 데이터 미리보기 (디버깅용)
void PrintDataPreview(const CsvLoader::Dataset& data, const std::vector<std::string>& featureNames, int numRows = 5)
{
    const int PRECISION = 6;
    const int W_FEAT = 12;
    const int W_IDX = 6;
    const int W_LABEL = 8;
    auto rowsToPrint = std::min(data.rows, numRows);
    auto colsToPrint = std::min(data.cols, 10); // 최대 10개 컬럼만 출력

    std::cout.precision(PRECISION);
    std::cout << std::fixed;

    std::cout << "\n" << std::string(50, '=') << " [Data Preview] " << std::string(50, '=') << "\n";

    // Header
    std::cout << " " << std::setw(W_IDX) << "IDX" << " | " << std::setw(W_LABEL) << "Label" << " |";
    for (int j = 0; j < colsToPrint; ++j)
    {
        std::string name = featureNames[j];
        if (name.length() > W_FEAT) name = name.substr(0, W_FEAT);
        std::cout << " " << std::setw(W_FEAT) << name << " |";
    }
    std::cout << "\n" << std::string(100, '-') << "\n";

    // Data
    for (auto i = 0; i < rowsToPrint; ++i)
    {
        std::cout << " " << std::setw(W_IDX) << i << " | "
            << std::setw(W_LABEL) << data.labels[i] << " |"; // 부도여부

        for (auto j = 0; j < colsToPrint; ++j)
        {
            std::cout << " " << std::setw(W_FEAT) << data.features[i * data.cols + j] << " |";
        }
        std::cout << "\n";
    }
    std::cout << std::string(100, '=') << "\n";
}

int main()
{
    // 1. 콘솔 및 I/O 설정
    ::SetConsoleOutputCP(65001); // UTF-8 설정
    std::ios::sync_with_stdio(false);
    std::cin.tie(NULL);

    try
    {
        // ===================================================================================
        // [설정 1] 데이터 로딩 설정 (Dual Model 검증용)
        // ===================================================================================
        // 듀얼 모델 검증 시, 기본 타겟은 'loan_status'(부도여부)여야 합니다.
        // CsvLoader가 'Return' 값은 별도로 dataset.returns에 저장함.
        auto csvFile = "loan_status.csv"s;
        auto targetColumn = "loan_status"s;

        // 제거할 변수 목록 (Data Leakage 방지)
        std::set<std::string> ignoreList =
        {
            "Actual_term",      // 사후 정보
            "total_pymnt",      // 사후 정보
            "last_pymnt_amnt",  // 사후 정보
            (const char*)u8"내부수익률", // 사용 안 함

            "loan_status",      // Target(Y)이므로 Feature(X)에서 제외
            "Return",           // 결과(Y)이므로 Feature(X)에서 제외
        };

        std::cout << ">>> [1/3] Data Loading...\n";

        CsvLoader loader(csvFile, targetColumn, ignoreList);
        auto dataset = loader.Load();

        if (dataset.rows == 0)
        {
            std::cerr << "Error: No data found in file." << std::endl;
            return 1;
        }

        std::cout << " -> Load Complete: " << dataset.rows << " Rows, " << dataset.cols << " Cols(Features)\n";

        // (선택) 데이터 미리보기
        // PrintDataPreview(dataset, loader.GetFeatureNames());

        std::cout << "\nContinuing to Training? (1: Yes / 0: No): ";
        int bKeepGoing = 0;
        std::cin >> bKeepGoing;
        if (bKeepGoing != 1) return 0;

        // ===================================================================================
        // [설정 2] 모델 하이퍼파라미터 설정
        // ===================================================================================
        // 1. 회귀 모델 (수익률 예측)
        ModelConfig regConfig = { 2, 4, 0.05f, 150, "reg:squarederror", "rmse" };

        // 2. 분류 모델 (부도 확률 예측)
        // 불균형 데이터일 경우 scale_pos_weight 등을 고려할 수 있음
        ModelConfig clsConfig = { 1, 6, 0.1f, 100, "binary:logistic", "auc" };


        // ===================================================================================
        // [설정 3] 실험 및 검증 실행 (Dual Model Validation)
        // ===================================================================================
        std::cout << "\n>>> [2/3] Starting Dual Model Validation...\n";

        ExperimentManager manager;

        // 파라미터: 데이터셋, 분류설정, 회귀설정, Split비율(0.8), PD컷오프(0.1), 수익률컷오프(0.05)
        // 상위 80% 데이터로 학습하고, 하위 20% 데이터로 테스트.
        // 부도확률 < 10% 이고, 예측수익률 > 5% 인 대출만 승인했을 때의 샤프지수 계산.
        float sharpe = manager.RunDualModelValidation(
            dataset,
            clsConfig,
            regConfig,
            0.8f,   // Train/Test Split Ratio
            0.1f,   // PD Threshold (10% 미만 부도확률)
            0.05f   // Return Threshold (5% 이상 예상수익)
        );

        std::cout << "\n>>> [3/3] Process Finished.\n";
        std::cout << ">>> Final Experiment Result (Sharpe Ratio): " << sharpe << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Critical Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}