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

//ValidationMetrics ExperimentManager::RunDualModelValidationAndOptimizeThreshold(const CsvLoader::Dataset& dataset, const ModelConfig& clsConfig, const ModelConfig& regConfig, float splitRatio)
//{
//    //모델 학습
//    size_t totalRows = dataset.rows;
//    size_t splitPoint = static_cast<size_t>(totalRows * splitRatio);
//    size_t testSize = totalRows - splitPoint;
//
//    DMatrixHandle hFullCls, hFullReg;
//    const float nanVal = std::numeric_limits<float>::quiet_NaN();
//    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullCls));
//    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullReg));
//    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullCls, "label", dataset.labels.data(), totalRows));
//    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullReg, "label", dataset.returns.data(), totalRows));
//
//    std::vector<int> allIndices(totalRows);
//    std::iota(allIndices.begin(), allIndices.end(), 0);
//    DMatrixHandle hTrainCls, hTestCls, hTrainReg, hTestReg;
//    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data(), splitPoint, &hTrainCls));
//    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data() + splitPoint, testSize, &hTestCls));
//    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data(), splitPoint, &hTrainReg));
//    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data() + splitPoint, testSize, &hTestReg));
//
//    BoosterHandle hBoosterCls = TrainBooster(hTrainCls, clsConfig);
//    BoosterHandle hBoosterReg = TrainBooster(hTrainReg, regConfig);
//
//    const float* predPD, * predEstReturn;
//    bst_ulong outLen;
//    SAFE_XGBOOST(XGBoosterPredict(hBoosterCls, hTestCls, 0, 0, 0, &outLen, &predPD));
//    SAFE_XGBOOST(XGBoosterPredict(hBoosterReg, hTestReg, 0, 0, 0, &outLen, &predEstReturn));
//
//
//    double sumAbsoluteError = 0.0; // 회귀 모델 오차 계산용
//    int correctCount = 0;         // 분류 모델 정답 개수용
//
//    // 테스트 데이터 전체를 돌면서 채점
//    for (size_t i = 0; i < testSize; ++i)
//    {
//        // 1. 회귀(Return) 모델 채점: RMSE (얼마나 틀렸나?)
//        float actualRet = dataset.returns[splitPoint + i];
//        float diff = predEstReturn[i] - actualRet;
//        sumAbsoluteError += std::abs(diff);;
//
//        // 2. 분류(PD) 모델 채점: Accuracy (부도 여부 맞췄나?)
//        // (PD가 0.5보다 크면 부도(1), 작으면 정상(0)으로 간주하고 채점)
//        float actualLabel = dataset.labels[splitPoint + i];
//        int predictedLabel = (predPD[i] > 0.5f) ? 1 : 0;
//
//        if (predictedLabel == (int)actualLabel)
//        {
//            correctCount++;
//        }
//    }
//
//    double mae = sumAbsoluteError / testSize;
//    double accuracy = (double)correctCount / testSize * 100.0;
//
//    // 콘솔에 성적표 출력 (첫 번째 이터레이션에서만 출력하거나, 필요시 매번 출력)
//    std::cout << "\n>>> [Basic Performance Report] <<<\n";
//    std::cout << "1. Return Model MAE  : " << mae << "\n";
//    std::cout << "2. PD Model Accuracy : " << accuracy << "%\n";
//    std::cout << "--------------------------------------\n";
//
//    double sumPD = 0, minPD = 1.0, maxPD = 0.0;
//    double sumRet = 0, minRet = 1.0, maxRet = -1.0;
//
//   
//    // --- 임계값 최적화 ---
//    // 학습된 모델(예측값)을 고정해두고, 다양한 임계값을 대입해 최적의 조합을 찾음
//
//    ValidationMetrics bestLocalMetrics;
//    bestLocalMetrics.sharpeRatio = -999.0f;
//    bestLocalMetrics.approvedCount = 0;
//
//    // 실제값 데이터 준비 (반복문 밖에서 한 번만 로드)
//    std::vector<float> testActualReturns(dataset.returns.begin() + splitPoint, dataset.returns.end());
//    std::vector<float> testBonds(dataset.bondYields.begin() + splitPoint, dataset.bondYields.end());
//    std::vector<bool> isApproved(testSize);
//
//    // 검색할 범위 설정
//    std::vector<float> pdCandidates = { 0.05f, 0.10f, 0.15f, 0.20f, 0.25f };
//    std::vector<float> pdCandidates = { 0.f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f };
//    std::vector<float> retCandidates = { 0.02f, 0.04f, 0.05f, 0.06f, 0.08f };
//
//    for (float pdTh : pdCandidates)
//    {
//        for (float retTh : retCandidates)
//        {
//            int approvedCount = 0;
//            double sumRet = 0.0;
//            double sumPD = 0.0;
//
//            // 필터링 시뮬레이션
//            for (size_t i = 0; i < testSize; ++i)
//            {
//                bool pass = (predPD[i] < pdTh && predEstReturn[i] > retTh);
//                isApproved[i] = pass;
//                if (pass)
//                {
//                    approvedCount++;
//                    sumRet += dataset.returns[splitPoint + i];
//                    sumPD += predPD[i];
//                }
//            }
//
//            // 너무 적게 승인되면 통계적 의미가 없으므로 스킵 (예: 10개 미만)
//            if (approvedCount < 10) continue;
//
//            float currentSharpe = CalculateSharpeRatio(testActualReturns, testBonds, isApproved);
//
//            // 현재 모델 내에서 최고의 임계값 갱신
//            if (currentSharpe > bestLocalMetrics.sharpeRatio)
//            {
//                bestLocalMetrics.sharpeRatio = currentSharpe;
//                bestLocalMetrics.bestPDThreshold = pdTh;
//                bestLocalMetrics.bestReturnThreshold = retTh;
//                bestLocalMetrics.approvedCount = approvedCount;
//                bestLocalMetrics.avgReturn = (float)(sumRet / approvedCount);
//                bestLocalMetrics.avgPD = (float)(sumPD / approvedCount);
//            }
//        }
//    }
//
//    // 메모리 해제
//    XGDMatrixFree(hFullCls); XGDMatrixFree(hFullReg);
//    XGDMatrixFree(hTrainCls); XGDMatrixFree(hTestCls);
//    XGDMatrixFree(hTrainReg); XGDMatrixFree(hTestReg);
//    XGBoosterFree(hBoosterCls); XGBoosterFree(hBoosterReg);
//
//    // 만약 조건 맞는게 하나도 없었으면 초기값 반환
//    return bestLocalMetrics;
//}


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