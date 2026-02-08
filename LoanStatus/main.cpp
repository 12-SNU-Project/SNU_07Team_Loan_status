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
    ::SetConsoleOutputCP(65001);
    std::ios::sync_with_stdio(false);
    std::cin.tie(NULL);

    try
    {
        // [설정 1] 데이터 로딩
        auto csvFile = "loan_status.csv"s;
        auto targetColumn = "loan_status"s;
        std::set<std::string> ignoreList =
        {
            "Actual_term", "total_pymnt", "last_pymnt_amnt", (const char*)u8"내부수익률",
            "loan_status", "Return"
        };

        std::cout << ">>> [1/3] Data Loading...\n";
        CsvLoader loader(csvFile, targetColumn, ignoreList);
        auto dataset = loader.Load();

        if (dataset.rows == 0) {
            std::cerr << "Error: No data found.\n"; return 1;
        }
        std::cout << " -> Load Complete: " << dataset.rows << " Rows, " << dataset.cols << " Cols\n";

        std::cout << "\nContinuing to Training? (1: Yes / 0: No): ";
        int bKeepGoing = 0;
        std::cin >> bKeepGoing;
        if (bKeepGoing != 1) return 0;

        // [설정 2 & 3] 실험 및 검증 실행
        std::cout << "\n>>> [2/3] Starting Integrated Grid Search...\n";
        std::cout << "    - Strategy: Dual Model (Classification + Regression)\n";
        std::cout << "    - Optimization: Parameter Grid + Threshold Auto-Tuning\n";

        ExperimentManager manager;

        // Grid Search 실행 (하이퍼파라미터 + 임계값 최적화 자동 수행)
        // 기존의 RunDualModelValidation(고정값 1회 실행) 대신 이걸 호출하는 게 좋습니다.
        ExperimentResult finalResult = manager.RunGridSearchAuto(dataset, 0.8f);

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << " [FINAL BEST STRATEGY REPORT]\n";
        std::cout << std::string(60, '=') << "\n";

        std::cout << " 1. Performance Metrics (Test Set)\n";
        std::cout << "   - Best Sharpe Ratio : " << std::fixed << std::setprecision(5) << finalResult.bestMetrics.sharpeRatio << "\n";
        std::cout << "   - Portfolio Return  : " << std::setprecision(2) << (finalResult.bestMetrics.avgReturn * 100.f) << "% (Avg)\n";
        std::cout << "   - Portfolio PD      : " << std::setprecision(2) << (finalResult.bestMetrics.avgPD * 100.f) << "% (Avg)\n";
        std::cout << "   - Approved Loans    : " << finalResult.bestMetrics.approvedCount << " cases\n\n";
       
        std::cout << " 2. Optimal Thresholds (Action Plan)\n";
        std::cout << "   - PD Cut-off        : < " << std::setprecision(2) << finalResult.bestMetrics.bestPDThreshold << " (Lower is stricter)\n";
        std::cout << "   - Return Cut-off    : > " << std::setprecision(2) << finalResult.bestMetrics.bestReturnThreshold << " (Higher is stricter)\n\n";
        
        std::cout << " 3. Best Model Hyperparameters\n";
        std::cout << "   [Classification Model - Default Prediction]\n";
        std::cout << "     - Max Depth: " << finalResult.bestClsConfig.maxDepth << "\n";
        std::cout << "     - Eta (LR) : " << finalResult.bestClsConfig.eta << "\n";
        std::cout << "     - Rounds   : " << finalResult.bestClsConfig.numRound << "\n";
        std::cout << "   [Regression Model - Return Prediction]\n";
        std::cout << "     - Max Depth: " << finalResult.bestRegConfig.maxDepth << "\n";
        std::cout << "     - Eta (LR) : " << finalResult.bestRegConfig.eta << "\n";
        std::cout << "     - Rounds   : " << finalResult.bestRegConfig.numRound << "\n";
        std::cout << std::string(60, '=') << "\n";

    }
    catch (const std::exception& e)
    {
        std::cerr << "[Critical Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}