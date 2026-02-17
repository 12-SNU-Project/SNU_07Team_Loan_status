#include "pch.h"
#include "CsvLoader.h"
#include "ExperimentManager.h"


// =========================================================
// [Mode 0] Find Best ThreshHold Configuration
// =========================================================

ExperimentResult ExperimentManager::RunGridSearchAuto(const CsvLoader::DataSet& dataset, float splitRatio)
{
    std::ofstream csvFile("grid_search_auto.csv");
    csvFile << "Iter,Cls_Depth,Cls_Eta,Reg_Depth,Reg_Eta," << "Best_PD_Thresh,Best_Ret_Thresh," << "Approved_Cnt,Avg_Return,Avg_PD,Sharpe_Ratio\n";

    std::cout << "\n>>> Automatic Grid Search (Params + Thresholds)\n";
    auto clsConfigs = GenerateGrid(true);
    auto regConfigs = GenerateGrid(false);

    auto totalIter = (int)(clsConfigs.size() * regConfigs.size());
    std::cout << ">>> Total Combinations: " << totalIter << "\n";
    std::cout << ">>> Log File: 'grid_search_auto.csv'\n\n";

    ExperimentResult bestResult;
    bestResult.bestMetrics.sharpeRatio = -999.0f;

    auto curRound = 0;
    std::cout << std::fixed << std::setprecision(4);

    for (const auto& cConf : clsConfigs)
    {
        for (const auto& rConf : regConfigs)
        {
            curRound++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //고정된 임계값 없이 호출 -> 내부에서 최적 임계값을 찾아옴
            auto ctx = PrepareExperiment(dataset, cConf, rConf, splitRatio);

            auto metrics = FindBestThresholds(ctx);

            // 로그 출력 (찾아낸 최적 임계값도 같이 표시)
            std::cout << "[" << curRound << "/" << totalIter << "] "
                << "C(d" << cConf.maxDepth << " e" << std::setprecision(2) << cConf.eta << ") "
                << "R(d" << rConf.maxDepth << " e" << std::setprecision(2) << rConf.eta << ") "
                << "| Th(" << std::setprecision(2) << metrics.bestPDThreshold << "/" << metrics.bestReturnThreshold << ") "
                << "-> Sharpe: " << std::setprecision(5) << metrics.sharpeRatio;

            if (metrics.sharpeRatio > bestResult.bestMetrics.sharpeRatio)
            {
                bestResult.bestMetrics = metrics;
                bestResult.bestClsConfig = cConf;
                bestResult.bestRegConfig = rConf;
                std::cout << " [★ NEW BEST]";
            }
            std::cout << "\n";

            // CSV 저장
            csvFile << curRound << ","
                << cConf.maxDepth << "," << cConf.eta << ","
                << rConf.maxDepth << "," << rConf.eta << ","
                << metrics.bestPDThreshold << "," << metrics.bestReturnThreshold << ","
                << metrics.approvedCount << ","
                << metrics.avgReturn << ","
                << metrics.avgPD << ","
                << metrics.sharpeRatio << "\n";
            csvFile.flush();
        }
    }

    csvFile.close();
    return bestResult;
}

ValidationMetrics ExperimentManager::FindBestThresholds(const ExperimentContext& ctx)
{
    ValidationMetrics bestMetrics;
    bestMetrics.sharpeRatio = -999.0f;
    bestMetrics.approvedCount = 0;

    // 탐색할 후보군
    std::vector<float> pdCandidates = { 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f };
    std::vector<float> retCandidates = { 0.02f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f };

    // PrepareExperiment가 리턴한 ctx에는 이미 Test Set만 들어있음
    auto n = ctx.testSize;
    std::vector<bool> approval(n);

    for (auto pdTh : pdCandidates)
    {
        for (auto retTh : retCandidates)
        {
            auto approvalCount = 0;
            auto sumRet = 0.0;
            auto sumPD = 0.0;

            // 벡터화된 연산이 없으므로 루프로 필터링
            for (auto i = 0; i < n; ++i)
            {
                auto bPass = (ctx.predPD[i] < pdTh) && (ctx.predEstReturn[i] > retTh);
                approval[i] = bPass;

                if (bPass)
                {
                    approvalCount++;
                    sumRet += ctx.actualReturns[i];
                    sumPD += ctx.predPD[i];
                }
            }

            if (approvalCount < 10) continue;

            // 기존 함수 재사용 (정확한 N-1 샤프지수 계산)
            auto currentSharpe = CalculateSharpeRatio(ctx.actualReturns, ctx.bondYields, approval);

            // 최고 기록 갱신
            if (currentSharpe > bestMetrics.sharpeRatio)
            {
                bestMetrics.sharpeRatio = currentSharpe;
                bestMetrics.bestPDThreshold = pdTh;
                bestMetrics.bestReturnThreshold = retTh;
                bestMetrics.approvedCount = approvalCount;
                bestMetrics.approvedRate = (float)approvalCount / n * 100.0f;
                bestMetrics.avgReturn = (float)(sumRet / approvalCount);
                bestMetrics.avgPD = (float)(sumPD / approvalCount);
            }
        }
    }

    return bestMetrics;
}