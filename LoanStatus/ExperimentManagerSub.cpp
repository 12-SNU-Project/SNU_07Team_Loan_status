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

ExperimentResult ExperimentManager::RunGridSearchAuto(const CsvLoader::DataPack& pack)
{
    std::ofstream csvFile("grid_search_auto.csv");
    csvFile << "Iter,Cls_Depth,Cls_Eta,Reg_Depth,Reg_Eta,"
        << "Best_PD_Thresh,Best_Ret_Thresh,"
        << "Approved_Cnt,Avg_Return,Avg_PD,Sharpe_Ratio\n";

    std::cout << "\n>>> [Mode 0] Automatic Grid Search on Validation Set\n";
    std::cout << ">>> Training on 60% (Train), Evaluating on 20% (Val)\n";

    auto clsConfigs = GenerateGrid(true);
    auto regConfigs = GenerateGrid(false);

    auto totalIter = (int)(clsConfigs.size() * regConfigs.size());
    std::cout << ">>> Total Combinations: " << totalIter << "\n";

    ExperimentResult bestResult;
    bestResult.bestMetrics.sharpeRatio = -999.0f;

    auto curRound = 0;
    std::cout << std::fixed << std::setprecision(4);

    for (const auto& cConf : clsConfigs)
    {
        for (const auto& rConf : regConfigs)
        {
            curRound++;

            // 1. Train Set (60%)으로 모델 학습

            auto hCls = TrainModelOnSet(pack.train, cConf);
            auto hReg = TrainModelOnSet(pack.train, rConf);

            // 2. Validation Set (20%)으로 예측 수행
            auto valCtx = PredictOnSet(hCls, hReg, pack.val);


            // 3. Validation Set 기준 최적 임계값 및 샤프지수 계산
            auto metrics = FindBestThresholds(valCtx);

            // 로그 출력
            std::cout << "[" << curRound << "/" << totalIter << "] "
                << "C(d" << cConf.maxDepth << " e" << std::setprecision(2) << cConf.eta << ") "
                << "R(d" << rConf.maxDepth << " e" << std::setprecision(2) << rConf.eta << ") "
                << "| Val-Sharpe: " << std::setprecision(5) << metrics.sharpeRatio;

            // 최고 기록 갱신 (기준은 Validation Score!)
            if (metrics.sharpeRatio > bestResult.bestMetrics.sharpeRatio)
            {
                bestResult.bestMetrics = metrics;
                bestResult.bestClsConfig = cConf;
                bestResult.bestRegConfig = rConf;
                std::cout << " [★ BEST]";
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

            // 메모리 해제 (Loop 내에서 누수 방지)
            XGBoosterFree(hCls);
            XGBoosterFree(hReg);
        }
    }

    csvFile.close();
    std::cout << "\n>>> Grid Search Complete.\n";
    std::cout << ">>> Best Validation Sharpe: " << bestResult.bestMetrics.sharpeRatio << "\n";

    return bestResult;
}

void ExperimentManager::RunStandardValidation(const CsvLoader::DataPack& pack, const ModelConfig& bestCls, const ModelConfig& bestReg)
{
    std::cout << "\n======================================================\n";
    std::cout << ">>> [Mode 3] Standard Validation (6:2:2 Split) <<<\n";
    std::cout << "======================================================\n";

    //Train Set (60%)으로 학습
    std::cout << ">>> [Step 1] Training Models on Train Set (" << pack.train.rows << " rows)...\n";

    // Train Set으로 학습 수행
    auto hClsBooster = TrainModelOnSet(pack.train, bestCls);
    auto hRegBooster = TrainModelOnSet(pack.train, bestReg);

    // Validation Set (20%)으로 최적 임계값 찾기
    std::cout << ">>> [Step 2] Optimizing Thresholds on Validation Set (" << pack.val.rows << " rows)...\n";

    // Val Set 예측
    auto valCtx = PredictOnSet(hClsBooster, hRegBooster, pack.val);

    // 기존 FindBestThresholds 함수 재사용 (Val 데이터로 수행)
    auto bestMetrics = FindBestThresholds(valCtx);

    std::cout << "   -> Found Best Thresholds on Validation Set:\n";
    std::cout << "      PD < " << bestMetrics.bestPDThreshold << ", Return > " << bestMetrics.bestReturnThreshold << "\n";
    std::cout << "      Val Sharpe Ratio: " << bestMetrics.sharpeRatio << "\n";

    // Test Set (20%)으로 최종 검증
    std::cout << ">>> [Step 3] Final Verification on Test Set (" << pack.test.rows << " rows)...\n";

    // Test Set 예측
    auto testCtx = PredictOnSet(hClsBooster, hRegBooster, pack.test);

    // Val에서 찾은 임계값(bestMetrics)을 Test에 그대로 적용
    std::vector<bool> finalApproval;
    finalApproval.reserve(testCtx.testSize);

    auto approvedCount = 0;
    for (auto i = 0; i < testCtx.testSize; ++i)
    {
        auto bPass = (testCtx.predPD[i] < bestMetrics.bestPDThreshold) && (testCtx.predEstReturn[i] > bestMetrics.bestReturnThreshold);
        finalApproval.push_back(bPass);
        if (bPass) approvedCount++;
    }

    // 최종 샤프지수 계산
    auto finalSharpe = 0.f;
    if (approvedCount >= 10)
    {
        // 10개 이상일 때만 정식 계산
        finalSharpe = CalculateSharpeRatio(testCtx.actualReturns, testCtx.bondYields, finalApproval);
    }
    else
    {
        // 10개 미만이면 경고 출력 (0점 처리)
        std::cout << ">>> [WARNING] Not enough approved samples (" << approvedCount << " < 10). Sharpe set to 0.0.\n";
        finalSharpe = 0.0f;
    }

    std::cout << "\n------------------------------------------------------\n";
    std::cout << ">>> [Final Result]\n";
    std::cout << "1. Validation Sharpe : " << bestMetrics.sharpeRatio << "\n";
    std::cout << "2. Test Set Sharpe   : " << finalSharpe << "\n";
    std::cout << "3. Approved Count    : " << approvedCount << " / " << testCtx.testSize
        << " (" << (float)approvedCount / testCtx.testSize * 100.f << "%)\n";

    float diff = std::abs(finalSharpe - bestMetrics.sharpeRatio);
    if (diff < 0.5f) 
    {
        std::cout << ">>> SUCCESS: Model is Robust! (Val & Test scores are similar)\n";
    }
    else 
    {
        std::cout << ">>> WARNING: Large gap detected. Check for Overfitting.\n";
    }
    std::cout << "------------------------------------------------------\n";

    PerformRandomPermutationTest(testCtx.predPD, testCtx.predEstReturn, testCtx.actualReturns, testCtx.bondYields,
        bestMetrics.bestPDThreshold, bestMetrics.bestReturnThreshold, 1000);

    XGBoosterFree(hClsBooster);
    XGBoosterFree(hRegBooster);
}

BoosterHandle ExperimentManager::TrainModelOnSet(const CsvLoader::DataSet& trainSet, const ModelConfig& config)
{
    DMatrixHandle hTrain;
    constexpr const auto nanVal = std::numeric_limits<float>::quiet_NaN();

    // Features -> DMatrix
    SAFE_XGBOOST(XGDMatrixCreateFromMat(trainSet.features.data(), trainSet.rows, trainSet.cols, nanVal, &hTrain));

    // Labels 설정 (분류면 labels, 회귀면 returns 사용)
    if (config.objective.find("binary") != std::string::npos) 
    {
        SAFE_XGBOOST(XGDMatrixSetFloatInfo(hTrain, "label", trainSet.labels.data(), trainSet.rows));
    }
    else 
    {
        SAFE_XGBOOST(XGDMatrixSetFloatInfo(hTrain, "label", trainSet.returns.data(), trainSet.rows));
    }

    BoosterHandle booster = TrainBooster(hTrain, config);
    XGDMatrixFree(hTrain); 
    return booster;
}

ExperimentContext ExperimentManager::PredictOnSet(BoosterHandle hCls, BoosterHandle hReg, const CsvLoader::DataSet& targetSet)
{
    ExperimentContext ctx;
    ctx.testSize = targetSet.rows;
    ctx.actualReturns = targetSet.returns;
    ctx.bondYields = targetSet.bondYields;

    // DMatrix 생성
    DMatrixHandle hMat;
    constexpr const auto nanVal = std::numeric_limits<float>::quiet_NaN();
    SAFE_XGBOOST(XGDMatrixCreateFromMat(targetSet.features.data(), targetSet.rows, targetSet.cols, nanVal, &hMat));

    // 예측 수행
    const float* ptrPredPd;
    const float* ptrPredRet;
    bst_ulong outLen;

    SAFE_XGBOOST(XGBoosterPredict(hCls, hMat, 0, 0, 0, &outLen, &ptrPredPd));
    ctx.predPD.assign(ptrPredPd, ptrPredPd + outLen);

    SAFE_XGBOOST(XGBoosterPredict(hReg, hMat, 0, 0, 0, &outLen, &ptrPredRet));
    ctx.predEstReturn.assign(ptrPredRet, ptrPredRet + outLen);

    XGDMatrixFree(hMat);
    return ctx;
}
