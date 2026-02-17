#include "pch.h" 
#include "CsvLoader.h"
#include "ExperimentManager.h"

using namespace std::string_literals;

// 데이터 미리보기 (디버깅용)
void PrintDataPreview(const CsvLoader::DataSet& data, const std::vector<std::string>& featureNames, int numRows = 5)
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
            "loan_status", "Return",
        };

        std::cout << ">>> [1/3] Data Loading...\n";
        CsvLoader loader(csvFile, targetColumn, ignoreList);
        auto dataset = loader.Load();

        if (dataset.rows == 0) 
        {
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
        ModelConfig bestCls;
        bestCls.maxDepth = 7;       // Log: "Best Depth: 7"
        bestCls.eta = 0.05f;        // Log: "Best Eta: 0.05"
        bestCls.numRound = 400;     // Log: "Rounds: 400"
        bestCls.objective = "binary:logistic";
        bestCls.minChildWeight = 1.0f;
        bestCls.scalePosWeight = 1.0f;
        bestCls.subsample = 0.8f;
        bestCls.colsample = 0.8f;
        bestCls.evalMetric = "auc";

        ModelConfig bestReg;
        bestReg.maxDepth = 5;       // Log: "Best Depth: 5"
        bestReg.eta = 0.05f;        // Log: "Best Eta: 0.05"
        bestReg.numRound = 400;     // Log: "Rounds: 400"
        bestReg.objective = "reg:absoluteerror";
        bestReg.minChildWeight = 1.0f;
        bestReg.scalePosWeight = 1.0f; // 회귀는 1.0
        bestReg.subsample = 0.8f;
        bestReg.colsample = 0.8f;
        bestReg.evalMetric = "mae";
        manager.RunFullTestSet_Boostwrap(dataset, bestCls, bestReg, 0.8f);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Critical Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

