#include "pch.h"
#include "CsvLoader.h"
#include "ExperimentManager.h"

// =========================================================
// 학습 및 예측 데이터 준비
// =========================================================
ExperimentContext ExperimentManager::PrepareExperiment(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio)
{
    ExperimentContext ctx;

    // 1. 데이터 분할 크기 계산
    auto totalRows = dataset.rows;
    auto splitPoint = static_cast<size_t>(totalRows * splitRatio);
    ctx.testSize = totalRows - splitPoint;

    if (ctx.testSize <= 0)
    {
        std::cerr << "Error: Test set size is 0." << std::endl;
        std::exit(1);
    }

    // 2. DMatrix 생성
    DMatrixHandle hFullCls, hFullReg;
    const auto nanVal = std::numeric_limits<float>::quiet_NaN();
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullCls));
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullReg));
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullCls, "label", dataset.labels.data(), totalRows));
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullReg, "label", dataset.returns.data(), totalRows));

    // 3. Slice (Train/Test)
    std::vector<int> allIndices(totalRows);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    DMatrixHandle hTrainCls, hTestCls, hTrainReg, hTestReg;
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data(), splitPoint, &hTrainCls));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data() + splitPoint, ctx.testSize, &hTestCls));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data(), splitPoint, &hTrainReg));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data() + splitPoint, ctx.testSize, &hTestReg));

    // 4. 모델 학습
    std::cout << ">>> [Common] Training Best Models...\n";
    auto hBoosterCls = TrainBooster(hTrainCls, bestClsConfig);
    auto hBoosterReg = TrainBooster(hTrainReg, bestRegConfig);

    // 5. 예측 수행
    std::cout << ">>> [Common] Predicting Test Set (" << ctx.testSize << " rows)...\n";
    const float* predPD_ptr;
    const float* predRet_ptr;
    bst_ulong outLen;

    SAFE_XGBOOST(XGBoosterPredict(hBoosterCls, hTestCls, 0, 0, 0, &outLen, &predPD_ptr));
    SAFE_XGBOOST(XGBoosterPredict(hBoosterReg, hTestReg, 0, 0, 0, &outLen, &predRet_ptr));

    // 6. 결과 데이터를 벡터로 복사
    ctx.predPD.assign(predPD_ptr, predPD_ptr + outLen);
    ctx.predEstReturn.assign(predRet_ptr, predRet_ptr + outLen);

    // 정답 데이터 복사
    ctx.actualReturns.assign(dataset.returns.begin() + splitPoint, dataset.returns.end());
    ctx.bondYields.assign(dataset.bondYields.begin() + splitPoint, dataset.bondYields.end());

    // 7. 메모리 정리
    XGDMatrixFree(hFullCls); XGDMatrixFree(hFullReg);
    XGDMatrixFree(hTrainCls); XGDMatrixFree(hTestCls);
    XGDMatrixFree(hTrainReg); XGDMatrixFree(hTestReg);
    XGBoosterFree(hBoosterCls); XGBoosterFree(hBoosterReg);

    return ctx;
}

ExperimentResult ExperimentManager::RunGridSearchFixed(const CsvLoader::DataSet& dataset, float splitRatio, float pdThreshold, float estReturnThreshold)
{
    return ExperimentResult();
}

// =========================================================
// [Mode 1] Single 10k Sample Heatmap
// =========================================================
void ExperimentManager::RunHeatmap_SingleSample10k(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio)
{
    auto ctx = PrepareExperiment(dataset, bestClsConfig, bestRegConfig, splitRatio);

    std::cout << "\n======================================================\n";
    std::cout << ">>> [Mode 1] Single 10k Sample Heatmap <<<\n";
    std::cout << "======================================================\n";

    // 10k 샘플링
    const size_t targetSampleSize = 10000;
    auto finalSampleSize = std::min(targetSampleSize, ctx.testSize);

    std::vector<int> indices(ctx.testSize);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    indices.resize(finalSampleSize);

    std::vector<float> sampActual, sampBonds;
    sampActual.reserve(finalSampleSize);
    sampBonds.reserve(finalSampleSize);

    for (auto idx : indices)
    {
        sampActual.push_back(ctx.actualReturns[idx]);
        sampBonds.push_back(ctx.bondYields[idx]);
    }

    std::cout << ">>> Generating Heatmap Data...\n";
    std::ofstream outFile("heatmap_data_10k.csv");
    outFile << "PD_Threshold,Return_Threshold,Sharpe_Ratio,Approved_Count,Approved_Rate\n";

    auto totalCombinations = 0;

    for (auto i = 1; i <= 40; ++i)
    {
        auto pdTh = i * 0.01f;
        for (auto j = 1; j <= 20; ++j)
        {
            auto retTh = j * 0.005f;

            std::vector<bool> approval;
            approval.reserve(finalSampleSize);
            auto count = 0;

            for (auto idx : indices)
            {
                auto pass = (ctx.predPD[idx] < pdTh) && (ctx.predEstReturn[idx] > retTh);
                approval.push_back(pass);
                if (pass) count++;
            }

            auto sharpe = 0.0f;
            if (count > 10)
            {
                sharpe = CalculateSharpeRatio(sampActual, sampBonds, approval);
            }

            auto rate = (float)count / finalSampleSize * 100.0f;
            outFile << pdTh << "," << retTh << "," << sharpe << "," << count << "," << rate << "\n";
            totalCombinations++;
        }
    }
    outFile.close();
    std::cout << ">>> Heatmap Saved (" << totalCombinations << " combinations).\n";

    // 순열 검정
    std::vector<float> sampPredPD, sampPredRet;
    sampPredPD.reserve(finalSampleSize);
    sampPredRet.reserve(finalSampleSize);

    for (auto idx : indices)
    {
        sampPredPD.push_back(ctx.predPD[idx]);
        sampPredRet.push_back(ctx.predEstReturn[idx]);
    }
    PerformRandomPermutationTest(sampPredPD, sampPredRet, sampActual, sampBonds, 0.20f, 0.075f, 1000);
}

// =========================================================
// [Mode 2] Reliability Check
// =========================================================
void ExperimentManager::RunReliabilityCheck_Bootstrap(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio)
{
    auto ctx = PrepareExperiment(dataset, bestClsConfig, bestRegConfig, splitRatio);

    std::cout << "\n======================================================\n";
    std::cout << ">>> [Mode 2] Reliability Check (Bootstrapping) <<<\n";
    std::cout << "======================================================\n";

    const auto numSimulations = 1000;
    const auto sampleSize = 10000;
    const auto pdTh = 0.20f;    // 고정 파라미터
    const auto retTh = 0.075f;  // 고정 파라미터

    std::ofstream outFile("bootstrapping_detailed_results.csv");
    outFile << "Sim_ID,Sharpe_Ratio,Mean_Excess_Return,Std_Dev,Approved_Count,Approved_Rate\n";

    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> pool(ctx.testSize);
    std::iota(pool.begin(), pool.end(), 0);

    for (auto s = 1; s <= numSimulations; ++s)
    {
        std::shuffle(pool.begin(), pool.end(), g);

        auto sumExcess = 0.0;
        auto sumSqExcess = 0.0;
        auto approvedCount = 0;

        std::vector<float> subActual, subBonds;
        std::vector<bool> subApproval;
        subActual.reserve(sampleSize); subBonds.reserve(sampleSize); subApproval.reserve(sampleSize);

        for (auto k = 0; k < sampleSize; ++k)
        {
            auto idx = pool[k];
            auto pass = (ctx.predPD[idx] < pdTh) && (ctx.predEstReturn[idx] > retTh);

            subActual.push_back(ctx.actualReturns[idx]);
            subBonds.push_back(ctx.bondYields[idx]);
            subApproval.push_back(pass);

            if (pass)
            {
                auto excess = (double)(ctx.actualReturns[idx] - ctx.bondYields[idx]);
                sumExcess += excess;
                sumSqExcess += excess * excess;
                approvedCount++;
            }
        }

        auto meanExcess = 0.0;
        auto stdDev = 0.0;
        auto sharpe = 0.0f;

        if (approvedCount >= 10)
        {
            // 전체 샘플 사이즈 기준 통계 (포트폴리오 관점)
            meanExcess = sumExcess / (double)sampleSize;
            auto meanSq = sumSqExcess / (double)sampleSize;
            auto var = meanSq - (meanExcess * meanExcess);
            stdDev = (var > 0) ? std::sqrt(var) : 0.0;

            sharpe = CalculateSharpeRatio(subActual, subBonds, subApproval);
        }

        auto rate = (float)approvedCount / sampleSize * 100.0f;
        outFile << s << "," << sharpe << "," << meanExcess << "," << stdDev << "," << approvedCount << "," << rate << "\n";

        if (s % 100 == 0) std::cout << "Sim " << s << " / " << numSimulations << "\n";
    }
    outFile.close();
    std::cout << ">>> Results saved to 'bootstrapping_detailed_results.csv'\n";
}

// =========================================================
// [Mode 3] Robust Heatmap
// =========================================================
void ExperimentManager::RunHeatmap_RobustBootstrap(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio)
{
    auto ctx = PrepareExperiment(dataset, bestClsConfig, bestRegConfig, splitRatio);

    std::cout << "\n======================================================\n";
    std::cout << ">>> [Mode 3] Robust Heatmap (1,000 Bootstraps) <<<\n";
    std::cout << "======================================================\n";

    struct GridStats
    {
        double sumSharpe = 0;
        double sumMean = 0;
        double sumStd = 0;
        double sumCount = 0;
    };
    std::vector<GridStats> accumulator(40 * 20);

    const auto numSimulations = 1000;
    const auto sampleSize = 10000;

    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> pool(ctx.testSize);
    std::iota(pool.begin(), pool.end(), 0);

    for (auto s = 1; s <= numSimulations; ++s)
    {
        std::shuffle(pool.begin(), pool.end(), g);

        std::vector<float> curActual, curBonds, curPD, curRet;
        curActual.reserve(sampleSize); curBonds.reserve(sampleSize);
        curPD.reserve(sampleSize); curRet.reserve(sampleSize);

        for (auto k = 0; k < sampleSize; ++k)
        {
            auto idx = pool[k];
            curActual.push_back(ctx.actualReturns[idx]);
            curBonds.push_back(ctx.bondYields[idx]);
            curPD.push_back(ctx.predPD[idx]);
            curRet.push_back(ctx.predEstReturn[idx]);
        }

        for (auto i = 1; i <= 40; ++i)
        {
            auto pdTh = i * 0.01f;
            for (auto j = 1; j <= 20; ++j)
            {
                auto retTh = j * 0.005f;
                auto gridIdx = (i - 1) * 20 + (j - 1);

                auto sumEx = 0.0;
                auto sumSqEx = 0.0;
                auto count = 0;

                for (auto k = 0; k < sampleSize; ++k)
                {
                    if (curPD[k] < pdTh && curRet[k] > retTh)
                    {
                        auto ex = (double)(curActual[k] - curBonds[k]);
                        sumEx += ex;
                        sumSqEx += ex * ex;
                        count++;
                    }
                }

                if (count >= 10)
                {
                    auto mean = sumEx / (double)sampleSize;
                    auto var = (sumSqEx / (double)sampleSize) - (mean * mean);
                    auto std = (var > 0) ? std::sqrt(var) : 0.0;

                    if (std > 1e-9)
                    {
                        accumulator[gridIdx].sumSharpe += (mean / std);
                        accumulator[gridIdx].sumMean += mean;
                        accumulator[gridIdx].sumStd += std;
                        accumulator[gridIdx].sumCount += count;
                    }
                }
            }
        }
        if (s % 100 == 0) std::cout << "Sim " << s << "\n";
    }

    std::ofstream outFile("robust_heatmap_detailed.csv");
    outFile << "PD_Threshold,Return_Threshold,Avg_Sharpe,Avg_Return,Avg_StdDev,Avg_Approved_Count,Avg_Approved_Rate\n";

    for (auto i = 1; i <= 40; ++i)
    {
        auto pdTh = i * 0.01f;
        for (auto j = 1; j <= 20; ++j)
        {
            auto retTh = j * 0.005f;
            auto idx = (i - 1) * 20 + (j - 1);
            auto& acc = accumulator[idx];

            auto avgCnt = acc.sumCount / numSimulations;
            outFile << pdTh << "," << retTh << ","
                << acc.sumSharpe / numSimulations << ","
                << acc.sumMean / numSimulations << ","
                << acc.sumStd / numSimulations << ","
                << avgCnt << "," << (avgCnt / sampleSize * 100.0) << "\n";
        }
    }
    outFile.close();
    std::cout << ">>> Robust Heatmap saved to 'robust_heatmap_detailed.csv'\n";
}

// =========================================================
// [Mode 4] Full Test Set Heatmap
// =========================================================
void ExperimentManager::RunHeatmap_FullTestSet(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio)
{
    auto ctx = PrepareExperiment(dataset, bestClsConfig, bestRegConfig, splitRatio);

    std::cout << "\n======================================================\n";
    std::cout << ">>> [Mode 4] Full Test Set Heatmap (No Sampling) <<<\n";
    std::cout << "======================================================\n";

    std::ofstream outFile("heatmap_full_testset.csv");
    outFile << "PD_Threshold,Return_Threshold,Sharpe_Ratio,Mean_Excess_Return,Std_Dev,Approved_Count,Approved_Rate\n";

    std::vector<bool> fullApproval;
    fullApproval.reserve(ctx.testSize);

    for (auto i = 1; i <= 40; ++i)
    {
        auto pdTh = i * 0.01f;
        for (auto j = 1; j <= 20; ++j)
        {
            auto retTh = j * 0.005f;

            fullApproval.clear();
            auto sumEx = 0.0;
            auto sumSqEx = 0.0;
            auto count = 0;

            for (size_t k = 0; k < ctx.testSize; ++k)
            {
                auto pass = (ctx.predPD[k] < pdTh) && (ctx.predEstReturn[k] > retTh);
                fullApproval.push_back(pass);

                if (pass)
                {
                    auto ex = (double)(ctx.actualReturns[k] - ctx.bondYields[k]);
                    sumEx += ex;
                    sumSqEx += ex * ex;
                    count++;
                }
            }

            auto mean = 0.0;
            auto std = 0.0;
            auto sharpe = 0.0f;

            if (count >= 10)
            {
                // 분모를 전체 Test Size로 설정
                mean = sumEx / (double)ctx.testSize;
                auto var = (sumSqEx / (double)ctx.testSize) - (mean * mean);
                std = (var > 0) ? std::sqrt(var) : 0.0;

                sharpe = CalculateSharpeRatio(ctx.actualReturns, ctx.bondYields, fullApproval);
            }

            auto rate = (float)count / ctx.testSize * 100.0f;
            outFile << pdTh << "," << retTh << "," << sharpe << "," << mean << "," << std << "," << count << "," << rate << "\n";
        }
        if (i % 10 == 0) std::cout << "PD " << i * 0.01f << " done\n";
    }
    outFile.close();
    std::cout << ">>> Full Test Set Heatmap Saved.\n";

    // 순열 검정
    PerformRandomPermutationTest(ctx.predPD, ctx.predEstReturn, ctx.actualReturns, ctx.bondYields, 0.20f, 0.075f, 1000);
}

BoosterHandle ExperimentManager::TrainBooster(DMatrixHandle hTrain, const ModelConfig& config)
{
    BoosterHandle hBooster;
    SAFE_XGBOOST(XGBoosterCreate(&hTrain, 1, &hBooster));

    // 파라미터 설정
    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "verbosity", "0"));
    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "tree_method", "hist"));
    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "objective", config.objective.c_str()));
    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "eval_metric", config.evalMetric.c_str()));

    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "max_depth", std::to_string(config.maxDepth).c_str()));
    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "eta", std::to_string(config.eta).c_str()));

    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "min_child_weight", std::to_string(config.minChildWeight).c_str()));
    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "scale_pos_weight", std::to_string(config.scalePosWeight).c_str()));
    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "subsample", std::to_string(config.subsample).c_str()));
    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "colsample_bytree", std::to_string(config.colsample).c_str()));

    // 스레드 설정
    auto numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "nthread", std::to_string(numThreads).c_str()));

    // 학습 수행
    for (auto i = 0; i < config.numRound; ++i)
    {
        SAFE_XGBOOST(XGBoosterUpdateOneIter(hBooster, i, hTrain));
    }
    return hBooster;
}

std::vector<ModelConfig> ExperimentManager::GenerateGrid(bool isClassification)
{
    std::vector<ModelConfig> configs;
    int idCounter = 1;

    // 1. 모델 성격(분류 vs 회귀)에 따른 기본 설정
    std::string obj = isClassification ? "binary:logistic" : "reg:absoluteerror";
    std::string metric = isClassification ? "auc" : "mae";

    // 분류(불균형 데이터)는 양성 클래스에 가중치 4배, 회귀는 1배
    float scaleWeight = isClassification ? 1.0f : 1.0f;

    // 분류는 미세한 패턴(1.0)까지, 회귀는 굵직한 패턴(5.0)만 학습
    float minChild = isClassification ? 1.0f : 1.0f;

    // 2. 이중 루프로 조합 생성 (Depth x Eta)
    for (auto depth : candidateDepths)
    {
        for (auto eta : candidateEtas)
        {
            // Eta(학습률)에 반비례하여 Round(반복수) 자동 계산
            // 공식: 20 / eta 
            int rounds = static_cast<int>(20.0f / eta);

            // 안전장치 (너무 적거나 많지 않게 범위 제한)
            if (rounds < 100) rounds = 100;
            if (rounds > 3000) rounds = 3000;

            ModelConfig cfg;
            cfg.id = idCounter++;
            cfg.maxDepth = depth;
            cfg.eta = eta;
            cfg.numRound = rounds; // <-- 자동 계산된 값 적용

            cfg.objective = obj;
            cfg.evalMetric = metric;

            // 모델별 고정/추천 파라미터
            cfg.minChildWeight = minChild;
            cfg.scalePosWeight = scaleWeight;
            cfg.subsample = 0.8f;      // 데이터 행 샘플링 (일반적 추천값)
            cfg.colsample = 0.8f;      // 컬럼 샘플링 (일반적 추천값)

            configs.push_back(cfg);
        }
    }
    return configs;
}

float ExperimentManager::CalculateSharpeRatio(const std::vector<float>& returns, const std::vector<float>& bondTest, const std::vector<bool>& bApprovals)
{
    const auto n = returns.size();
    if (n <= 1) return 0.0f;

    //초과 수익률
    std::vector<float> excessReturns;
    excessReturns.reserve(n);

    auto accExcessVal = (double)0.0; // 정밀도를 위해 double 사용

    for (auto i = 0; i < n; ++i)
    {
        auto excessVal = 0.0f;
        if (bApprovals[i])
        {
            excessVal = returns[i] - bondTest[i];
        }
        excessReturns.push_back(excessVal);
        accExcessVal += excessVal;
    }

    // 1. 평균(Mean) 계산 (분자)
    // 데이터: 초과 수익률 벡터
    auto meanExcess = (double)(accExcessVal / static_cast<double>(n));

    // 2. 표준편차(StdDev) 계산 (분모)
    // 데이터: 위와 동일한 '초과 수익률 벡터' 사용
    auto sqAccDiff = (double)0.0;
    for (auto val : excessReturns)
    {
        auto diff = val - meanExcess;
        sqAccDiff += (diff * diff);
    }

    // 표본 분산 및 표준편차
    auto variance = sqAccDiff / static_cast<double>(n - 1);
    auto stdDev = std::sqrt(variance);

    // 3. 샤프지수 반환
    if (stdDev < 1e-9) return 0.0f; // 0으로 나누기 방지

    return static_cast<float>(meanExcess / stdDev);
}

void ExperimentManager::PerformRandomPermutationTest(const std::vector<float>& predPD, const std::vector<float>& predRet, const std::vector<float>& actualRet, const std::vector<float>& bondYields, float bestPDTh, float bestRetTh, int iterations)
{
    std::cout << "\n======================================================\n";
    std::cout << ">>> [Permutation Test] Verifying Best Strategy Significance <<<\n";
    std::cout << "======================================================\n";

    auto testSize = actualRet.size();
    if (testSize == 0) return;

    // 1. [Optimization] 고정된 승인 마스크(Approved Mask) 미리 계산
    //    순열 검정 중 '예측값'은 변하지 않으므로, 승인 여부는 루프 밖에서 한 번만 판정합니다.
    std::vector<bool> fixedApproval;
    fixedApproval.reserve(testSize);

    auto approvedCount = 0;
    for (auto i = 0; i < testSize; ++i)
    {
        // 아까 찾은 Best Threshold 적용 (PD < 0.20 && Return > 0.075)
        auto pass = (predPD[i] < bestPDTh) && (predRet[i] > bestRetTh);
        fixedApproval.push_back(pass);
        if (pass) approvedCount++;
    }

    // 2. 오리지널(진짜) 샤프지수 계산
    auto originalSharpe = CalculateSharpeRatio(actualRet, bondYields, fixedApproval);

    std::cout << "Best Config: PD < " << bestPDTh << ", Ret > " << bestRetTh << "\n";
    std::cout << "Original Sharpe Ratio: " << originalSharpe << " (Count: " << approvedCount << ")\n";
    std::cout << "Running " << iterations << " permutations...\n";

    // 3. 순열 검정 루프 (Permutation Loop)
    auto betterCount = 0;

    // 셔플을 위한 인덱스 벡터 생성 (0, 1, 2, ... N-1)
    std::vector<int> indices(testSize);
    std::iota(indices.begin(), indices.end(), 0);

    // 난수 생성기 (속도가 빠른 mt19937 사용)
    std::random_device rd;
    std::mt19937 g(rd());

    // 임시 벡터 (셔플된 데이터를 담을 공간) - 메모리 재할당 방지
    std::vector<float> shuffledReturns(testSize);
    std::vector<float> shuffledBonds(testSize);

    for (auto k = 0; k < iterations; ++k)
    {
        // 인덱스 뒤섞기 (데이터의 짝을 유지하기 위함)
        std::shuffle(indices.begin(), indices.end(), g);

        // 섞인 인덱스를 기반으로 가짜 데이터(Null Distribution) 생성
        for (auto i = 0; i < testSize; ++i)
        {
            shuffledReturns[i] = actualRet[indices[i]];
            shuffledBonds[i] = bondYields[indices[i]];
        }

        // '고정된 전략'이 '무작위 데이터'에서 얼마나 버는지 테스트
        auto randomSharpe = CalculateSharpeRatio(shuffledReturns, shuffledBonds, fixedApproval);

        // 만약 무작위 데이터로 얻은 수익이 원본보다 좋거나 같다면 카운트
        if (randomSharpe >= originalSharpe)
        {
            betterCount++;
        }

        // 진행 상황 표시 (10% 단위)
        if ((k + 1) % (iterations / 10) == 0)
        {
            std::cout << ".";
            std::cout.flush();
        }
    }
    std::cout << "\n";

    // 4. p-value 계산 및 판정
    // p-value = (betterCount + 1) / (numPermutations + 1)
    auto pValue = static_cast<double>(betterCount + 1) / (iterations + 1);

    std::cout << "------------------------------------------------------\n";
    std::cout << "Permutation Test Result:\n";
    std::cout << " - Better Random Strategies: " << betterCount << " / " << iterations << "\n";
    std::cout << " - P-Value: " << pValue << "\n"; // 보통 0.05 미만이면 유의미(Significant)

    if (pValue < 0.05)
    {
        std::cout << ">>> SUCCESS: The strategy is Statistically Significant! (Not Luck)\n";
    }
    else 
    {
        std::cout << ">>> WARNING: The strategy might be due to randomness.\n";
    }
    std::cout << "======================================================\n";
}
 