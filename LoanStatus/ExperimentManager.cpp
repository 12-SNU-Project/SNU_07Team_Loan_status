#include "pch.h"
#include <random>
#include "CsvLoader.h"
#include "ExperimentManager.h"

ExperimentResult ExperimentManager::RunGridSearchFixed(const CsvLoader::Dataset& dataset, float splitRatio, float pdThreshold, float estReturnThreshold)
{
    // 1) CSV 파일 준비
    std::ofstream csvFile("grid_serach_fixed.csv");
    // 헤더 작성
    csvFile << "Iter,Cls_ID,Cls_Depth,Cls_Eta,Reg_ID,Reg_Depth,Reg_Eta,"
        << "Approved_Cnt,Avg_Return,Avg_PD,Sharpe_Ratio\n";

    std::cout << "\n>>> [Grid Search] Generating Configurations...\n";

    auto clsConfigs = GenerateGrid(true);
    auto regConfigs = GenerateGrid(false);

    int totalIter = (int)(clsConfigs.size() * regConfigs.size());
    std::cout << ">>> Total Combinations: " << totalIter << "\n";
    std::cout << ">>> Log File: 'grid_search_fixed.csv'\n\n";

    ExperimentResult bestResult;
    bestResult.bestMetrics.sharpeRatio = -999.0f; // 초기값

    int currentIter = 0;
    std::cout << std::fixed << std::setprecision(4);

    for (const auto& cConf : clsConfigs)
    {
        for (const auto& rConf : regConfigs)
        {
            currentIter++;

            // 모델 엔진
            ValidationMetrics metrics = RunDualModelValidation(dataset, cConf, rConf, splitRatio, pdThreshold, estReturnThreshold);

            // [하이퍼파라미터] [추정수익률] [부도확률] [샤프레이시오]
            std::cout << "[" << currentIter << "/" << totalIter << "]\n"
                << "\n>>> C:d" << cConf.maxDepth << " e" << std::setprecision(2) << cConf.eta 
                << "\n>>> R:d" << rConf.maxDepth << " e" << std::setprecision(2) << rConf.eta
                << "\n>>> Ret: " << std::setprecision(4) << metrics.avgReturn
                << "\n>>> PD: " << std::setprecision(4) << metrics.avgPD
                << "\n>>> Sharpe: " << std::setprecision(5) << metrics.sharpeRatio;

            // 갱신 여부 체크
            if (metrics.sharpeRatio > bestResult.bestMetrics.sharpeRatio)
            {
                bestResult.bestMetrics = metrics;
                bestResult.bestClsConfig = cConf;
                bestResult.bestRegConfig = rConf;
                std::cout << "  [★ NEW BEST]"; // 갱신 알림
            }
            std::cout << "\n";

            // [CSV Output] 파일 저장
            csvFile << currentIter << ","
                << cConf.id << "," << cConf.maxDepth << "," << cConf.eta << ","
                << rConf.id << "," << rConf.maxDepth << "," << rConf.eta << ","
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

ExperimentResult ExperimentManager::RunGridSearchAuto(const CsvLoader::Dataset& dataset, float splitRatio)
{
    std::ofstream csvFile("grid_search_auto.csv");
    csvFile << "Iter,Cls_Depth,Cls_Eta,Reg_Depth,Reg_Eta,"
        << "Best_PD_Thresh,Best_Ret_Thresh,"
        << "Approved_Cnt,Avg_Return,Avg_PD,Sharpe_Ratio\n";

    std::cout << "\n>>> [Grid Search] Generating Configurations...\n";
    auto clsConfigs = GenerateGrid(true);
    auto regConfigs = GenerateGrid(false);

    int totalIter = (int)(clsConfigs.size() * regConfigs.size());
    std::cout << ">>> Total Combinations: " << totalIter << "\n";
    std::cout << ">>> Log File: 'grid_search_auto.csv'\n\n";

    ExperimentResult bestResult;
    bestResult.bestMetrics.sharpeRatio = -999.0f;

    int currentIter = 0;
    std::cout << std::fixed << std::setprecision(4);

    for (const auto& cConf : clsConfigs)
    {
        for (const auto& rConf : regConfigs)
        {
            currentIter++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //고정된 임계값 없이 호출 -> 내부에서 최적 임계값을 찾아옴
            ValidationMetrics metrics = RunDualModelValidationAndOptimizeThreshold(
                dataset, cConf, rConf, splitRatio
            );

            // 로그 출력 (찾아낸 최적 임계값도 같이 표시)
            std::cout << "[" << currentIter << "/" << totalIter << "] "
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
            csvFile << currentIter << ","
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

ValidationMetrics ExperimentManager::RunDualModelValidationAndOptimizeThreshold(const CsvLoader::Dataset& dataset, const ModelConfig& clsConfig, const ModelConfig& regConfig, float splitRatio)
{
    //모델 학습
    size_t totalRows = dataset.rows;
    size_t splitPoint = static_cast<size_t>(totalRows * splitRatio);
    size_t testSize = totalRows - splitPoint;

    DMatrixHandle hFullCls, hFullReg;
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullCls));
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullReg));
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullCls, "label", dataset.labels.data(), totalRows));
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullReg, "label", dataset.returns.data(), totalRows));

    std::vector<int> allIndices(totalRows);
    std::iota(allIndices.begin(), allIndices.end(), 0);
    DMatrixHandle hTrainCls, hTestCls, hTrainReg, hTestReg;
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data(), splitPoint, &hTrainCls));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data() + splitPoint, testSize, &hTestCls));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data(), splitPoint, &hTrainReg));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data() + splitPoint, testSize, &hTestReg));

    BoosterHandle hBoosterCls = TrainBooster(hTrainCls, clsConfig);
    BoosterHandle hBoosterReg = TrainBooster(hTrainReg, regConfig);

    const float* predPD, * predEstReturn;
    bst_ulong outLen;
    SAFE_XGBOOST(XGBoosterPredict(hBoosterCls, hTestCls, 0, 0, 0, &outLen, &predPD));
    SAFE_XGBOOST(XGBoosterPredict(hBoosterReg, hTestReg, 0, 0, 0, &outLen, &predEstReturn));


    double sumAbsoluteError = 0.0; // 회귀 모델 오차 계산용
    int correctCount = 0;         // 분류 모델 정답 개수용

    // 테스트 데이터 전체를 돌면서 채점
    for (size_t i = 0; i < testSize; ++i)
    {
        // 1. 회귀(Return) 모델 채점: RMSE (얼마나 틀렸나?)
        float actualRet = dataset.returns[splitPoint + i];
        float diff = predEstReturn[i] - actualRet;
        sumAbsoluteError += std::abs(diff);;

        // 2. 분류(PD) 모델 채점: Accuracy (부도 여부 맞췄나?)
        // (PD가 0.5보다 크면 부도(1), 작으면 정상(0)으로 간주하고 채점)
        float actualLabel = dataset.labels[splitPoint + i];
        int predictedLabel = (predPD[i] > 0.5f) ? 1 : 0;

        if (predictedLabel == (int)actualLabel) 
        {
            correctCount++;
        }
    }

    double mae = sumAbsoluteError / testSize;
    double accuracy = (double)correctCount / testSize * 100.0;

    // 콘솔에 성적표 출력 (첫 번째 이터레이션에서만 출력하거나, 필요시 매번 출력)
    std::cout << "\n>>> [Basic Performance Report] <<<\n";
    std::cout << "1. Return Model MAE  : " << mae << "\n";
    std::cout << "2. PD Model Accuracy : " << accuracy << "%\n";
    std::cout << "--------------------------------------\n";

    double sumPD = 0, minPD = 1.0, maxPD = 0.0;
    double sumRet = 0, minRet = 1.0, maxRet = -1.0;

    // 샘플링
    /*for (size_t i = 0; i < testSize; ++i) 
    {
        if (predPD[i] < minPD) minPD = predPD[i];
        if (predPD[i] > maxPD) maxPD = predPD[i];
        sumPD += predPD[i];

        if (predEstReturn[i] < minRet) minRet = predEstReturn[i];
        if (predEstReturn[i] > maxRet) maxRet = predEstReturn[i];
        sumRet += predEstReturn[i];
    }*/

    //// 첫 번째 이터레이션에서만 로그 출력
    //static bool bPrintedDebug = false;
    //if (!bPrintedDebug) 
    //{
    //    std::cout << "\n>>>[DEBUG MODEL OUTPUT]\n";
    //    std::cout << " - PD Range     : " << minPD << " ~ " << maxPD << " (Avg: " << (double)(sumPD / testSize) << ")\n";
    //    std::cout << " - Return Range : " << minRet << " ~ " << maxRet << " (Avg: " << (double)(sumRet / testSize) << ")\n";
    //    bPrintedDebug = true;
    //}


    // --- 임계값 최적화 ---
    // 학습된 모델(예측값)을 고정해두고, 다양한 임계값을 대입해 최적의 조합을 찾음

    ValidationMetrics bestLocalMetrics;
    bestLocalMetrics.sharpeRatio = -999.0f;
    bestLocalMetrics.approvedCount = 0;

    // 실제값 데이터 준비 (반복문 밖에서 한 번만 로드)
    std::vector<float> testActualReturns(dataset.returns.begin() + splitPoint, dataset.returns.end());
    std::vector<float> testBonds(dataset.bondYields.begin() + splitPoint, dataset.bondYields.end());
    std::vector<bool> isApproved(testSize);

    // 검색할 범위 설정
    std::vector<float> pdCandidates =  { 0.f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f};
    std::vector<float> retCandidates = { 0.02f, 0.04f, 0.05f, 0.06f, 0.08f };

    for (float pdTh : pdCandidates)
    {
        for (float retTh : retCandidates)
        {
            int approvedCount = 0;
            double sumRet = 0.0;
            double sumPD = 0.0;

            // 필터링 시뮬레이션
            for (size_t i = 0; i < testSize; ++i)
            {
                bool pass = (predPD[i] < pdTh && predEstReturn[i] > retTh);
                isApproved[i] = pass;
                if (pass)
                {
                    approvedCount++;
                    sumRet += dataset.returns[splitPoint + i];
                    sumPD += predPD[i];
                }
            }

            // 너무 적게 승인되면 통계적 의미가 없으므로 스킵 (예: 10개 미만)
            if (approvedCount < 10) continue;

            float currentSharpe = CalculateSharpeRatioVer2(testActualReturns, testBonds, isApproved);

            // 현재 모델 내에서 최고의 임계값 갱신
            if (currentSharpe > bestLocalMetrics.sharpeRatio)
            {
                bestLocalMetrics.sharpeRatio = currentSharpe;
                bestLocalMetrics.bestPDThreshold = pdTh;
                bestLocalMetrics.bestReturnThreshold = retTh;
                bestLocalMetrics.approvedCount = approvedCount;
                bestLocalMetrics.avgReturn = (float)(sumRet / approvedCount);
                bestLocalMetrics.avgPD = (float)(sumPD / approvedCount);
            }
        }
    }

    // 메모리 해제
    XGDMatrixFree(hFullCls); XGDMatrixFree(hFullReg);
    XGDMatrixFree(hTrainCls); XGDMatrixFree(hTestCls);
    XGDMatrixFree(hTrainReg); XGDMatrixFree(hTestReg);
    XGBoosterFree(hBoosterCls); XGBoosterFree(hBoosterReg);

    // 만약 조건 맞는게 하나도 없었으면 초기값 반환
    return bestLocalMetrics;
}

void ExperimentManager::RunSingleModelValidation(const CsvLoader::Dataset& dataset, const ModelConfig& clsConfig,  float splitRatio, float pdThreshold)
{
    std::cout << "\n>>> [Single Model Validation] Sequential Split & Analysis...\n";

    // 1. 데이터 분할 (Sequential Split)
    size_t totalRows = dataset.rows;
    size_t splitPoint = static_cast<size_t>(totalRows * splitRatio);
    size_t testSize = totalRows - splitPoint;

    // 2. DMatrix 생성(분류용만)
    DMatrixHandle hFullCls;
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullCls));
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullCls, "label", dataset.labels.data(), totalRows));

    // 3. Slice (Train/Test 분리)
    std::vector<int> allIndices(totalRows);
    std::iota(allIndices.begin(), allIndices.end(), 0);
    DMatrixHandle hTrainCls, hTestCls;
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data(), splitPoint, &hTrainCls));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data() + splitPoint, testSize, &hTestCls));


    // 4. 모델 학습
    std::cout << ">>> Training Model (Past Data)...\n";
    BoosterHandle hBoosterCls = TrainBooster(hTrainCls, clsConfig);


    // 5. 예측 수행
    std::cout << ">>> Predicting (Future Data)...\n";
    const float* predPD;
    bst_ulong outLenCls;
    SAFE_XGBOOST(XGBoosterPredict(hBoosterCls, hTestCls, 0, 0, 0, &outLenCls, &predPD));


    // 6. 필터링 적용
    std::vector<bool> isApproved(testSize, false);
    int approvedCount = 0;
    for (size_t i = 0; i < testSize; ++i)
    {
        if (predPD[i] < pdThreshold)
        {
            isApproved[i] = true;
            approvedCount++;
        }
    }

    // 7. Sharpe Ratio 계산 호출
    std::vector<float> testActualReturns(dataset.returns.begin() + splitPoint, dataset.returns.end());
    std::vector<float> testBonds(dataset.bondYields.begin() + splitPoint, dataset.bondYields.end());

    float sharpe = CalculateSharpeRatio(testActualReturns, testBonds, isApproved);


    std::cout << "================ [Dual Model Result] ================\n";
    std::cout << "  - Approved: " << approvedCount << " / " << testSize << "(" << (approvedCount / testSize) * 100.f << ")" << "\n";
    std::cout << "  - PD Threshold: " << pdThreshold << "\n";
    std::cout << "  - FINAL SHARPE RATIO: " << sharpe << "\n";
    std::cout << "=====================================================\n";

    XGDMatrixFree(hFullCls);
    XGDMatrixFree(hTrainCls);
    XGDMatrixFree(hTestCls);
}

// 2-Ver2. 추정수익률 + 부도확률까지 필터링한 후 샤프지수 계산하기
ValidationMetrics ExperimentManager::RunDualModelValidation(
    const CsvLoader::Dataset & dataset, // 전체 데이터셋 객체 전달
    const ModelConfig & clsConfig,     // 분류 모델 설정
    const ModelConfig & regConfig,     // 회귀 모델 설정
    float splitRatio,
    float pdThreshold,         // 부도 확률 임계값
    float estReturnThreshold  // 추정 수익률 임계값
)
{
    std::cout << "\n>>> [Dual Model Validation] Starting Analysis...\n";

    // 1. 데이터 분할 (Sequential Split)
    size_t totalRows = dataset.rows;
    size_t splitPoint = static_cast<size_t>(totalRows * splitRatio);
    size_t testSize = totalRows - splitPoint;

    // 2. DMatrix 생성 (분류용/회귀용 레이블 분리)
    DMatrixHandle hFullCls, hFullReg;
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullCls));
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullReg));

    // 레이블 바인딩: 분류(loan_status), 회귀(Return)
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullCls, "label", dataset.labels.data(), totalRows));
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullReg, "label", dataset.returns.data(), totalRows));

    // Slice (Train/Test 분리)
    std::vector<int> allIndices(totalRows);
    std::iota(allIndices.begin(), allIndices.end(), 0);
    DMatrixHandle hTrainCls, hTestCls, hTrainReg, hTestReg;
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data(), splitPoint, &hTrainCls));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data() + splitPoint, testSize, &hTestCls));

    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data(), splitPoint, &hTrainReg));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data() + splitPoint, testSize, &hTestReg));

    // 3. 모델 학습 (분류 & 회귀)
    BoosterHandle hBoosterCls = TrainBooster(hTrainCls, clsConfig);
    BoosterHandle hBoosterReg = TrainBooster(hTrainReg, regConfig);

    // 4. 예측 수행
    const float* predPD, * predEstReturn;
    bst_ulong outLenCls, outLenReg;
    SAFE_XGBOOST(XGBoosterPredict(hBoosterCls, hTestCls, 0, 0, 0, &outLenCls, &predPD));
    SAFE_XGBOOST(XGBoosterPredict(hBoosterReg, hTestReg, 0, 0, 0, &outLenReg, &predEstReturn));

    // 5. 이중 필터링 적용 (PD < Threshold && Est.Return > Threshold)
    std::vector<bool> isApproved(testSize, false);
    int approvedCount = 0;

    //[추가]
    double sumApprovedReturn = 0.0; // 평균 수익률 계산용
    double sumApprovedPD = 0.0;     // 평균 PD 계산용

    for (size_t i = 0; i < testSize; ++i)
    {
        if (predPD[i] < pdThreshold && predEstReturn[i] > estReturnThreshold)
        {
            isApproved[i] = true;
            approvedCount++;

            //[추가]
            sumApprovedReturn += dataset.returns[splitPoint + i];
            sumApprovedPD += predPD[i];
        }
    }

    // 6. 실제 Test 셋의 Return/Bond 추출 및 Sharpe Ratio 계산
    std::vector<float> testActualReturns(dataset.returns.begin() + splitPoint, dataset.returns.end());
    std::vector<float> testBonds(dataset.bondYields.begin() + splitPoint, dataset.bondYields.end());

    float sharpe = CalculateSharpeRatioVer2(testActualReturns, testBonds, isApproved);


    ValidationMetrics result;
    result.sharpeRatio = sharpe;
    result.approvedCount = approvedCount;

    if (approvedCount > 0) 
    {
        result.avgReturn = (float)(sumApprovedReturn / approvedCount);
        result.avgPD = (float)(sumApprovedPD / approvedCount);
    }
    else 
    {
        result.avgReturn = 0.0f;
        result.avgPD = 0.0f;
    }

    // 7. 결과 요약 출력
    std::cout << "================ [Dual Model Result] ================\n";
    std::cout << "  - Approved: " << approvedCount << " / " << testSize << "(" << ((float)approvedCount / testSize * 100.f) << "%)" << "\n";
    std::cout << "  - PD Threshold: " << pdThreshold << " | Est.Return Threshold: " << estReturnThreshold << "\n";
    std::cout << "  - FINAL SHARPE RATIO: " << sharpe << "\n";
    std::cout << "=====================================================\n";

    //8. 메모리 해제
    XGDMatrixFree(hFullCls); XGDMatrixFree(hFullReg);
    XGDMatrixFree(hTrainCls); XGDMatrixFree(hTestCls);
    XGDMatrixFree(hTrainReg); XGDMatrixFree(hTestReg);
    XGBoosterFree(hBoosterCls); XGBoosterFree(hBoosterReg);

    //[추가]
    return result;
}


void ExperimentManager::PerformRandomPermutationTest(const std::vector<float>& predPD, const std::vector<float>& predEstReturn, const std::vector<float>& testActualReturns, const std::vector<float>& testBonds, float bestPdTh, float bestRetTh, int numPermutations)
{
    std::cout << "\n======================================================\n";
    std::cout << ">>> [Permutation Test] Verifying Best Strategy Significance <<<\n";
    std::cout << "======================================================\n";

    size_t testSize = testActualReturns.size();
    if (testSize == 0) return;

    // 1. [Optimization] 고정된 승인 마스크(Approved Mask) 미리 계산
    //    순열 검정 중 '예측값'은 변하지 않으므로, 승인 여부는 루프 밖에서 한 번만 판정합니다.
    std::vector<bool> fixedApproval;
    fixedApproval.reserve(testSize);

    int approvedCount = 0;
    for (size_t i = 0; i < testSize; ++i)
    {
        // 아까 찾은 Best Threshold 적용 (PD < 0.20 && Return > 0.075)
        bool pass = (predPD[i] < bestPdTh) && (predEstReturn[i] > bestRetTh);
        fixedApproval.push_back(pass);
        if (pass) approvedCount++;
    }

    // 2. 오리지널(진짜) 샤프지수 계산
    float originalSharpe = CalculateSharpeRatioVer2(testActualReturns, testBonds, fixedApproval);

    std::cout << "Best Config: PD < " << bestPdTh << ", Ret > " << bestRetTh << "\n";
    std::cout << "Original Sharpe Ratio: " << originalSharpe << " (Count: " << approvedCount << ")\n";
    std::cout << "Running " << numPermutations << " permutations...\n";

    // 3. 순열 검정 루프 (Permutation Loop)
    int betterCount = 0;

    // 셔플을 위한 인덱스 벡터 생성 (0, 1, 2, ... N-1)
    std::vector<int> indices(testSize);
    std::iota(indices.begin(), indices.end(), 0);

    // 난수 생성기 (속도가 빠른 mt19937 사용)
    std::random_device rd;
    std::mt19937 g(rd());

    // 임시 벡터 (셔플된 데이터를 담을 공간) - 메모리 재할당 방지
    std::vector<float> shuffledReturns(testSize);
    std::vector<float> shuffledBonds(testSize);

    for (int k = 0; k < numPermutations; ++k)
    {
        // 인덱스 뒤섞기 (데이터의 짝을 유지하기 위함)
        std::shuffle(indices.begin(), indices.end(), g);

        // 섞인 인덱스를 기반으로 가짜 데이터(Null Distribution) 생성
        for (size_t i = 0; i < testSize; ++i)
        {
            shuffledReturns[i] = testActualReturns[indices[i]];
            shuffledBonds[i] = testBonds[indices[i]];
        }

        // '고정된 전략'이 '무작위 데이터'에서 얼마나 버는지 테스트
        float randomSharpe = CalculateSharpeRatioVer2(shuffledReturns, shuffledBonds, fixedApproval);

        // 만약 무작위 데이터로 얻은 수익이 원본보다 좋거나 같다면 카운트
        if (randomSharpe >= originalSharpe)
        {
            betterCount++;
        }

        // 진행 상황 표시 (10% 단위)
        if ((k + 1) % (numPermutations / 10) == 0)
        {
            std::cout << ".";
            std::cout.flush();
        }
    }
    std::cout << "\n";

    // 4. p-value 계산 및 판정
    // p-value = (betterCount + 1) / (numPermutations + 1)
    double pValue = static_cast<double>(betterCount + 1) / (numPermutations + 1);

    std::cout << "------------------------------------------------------\n";
    std::cout << "Permutation Test Result:\n";
    std::cout << " - Better Random Strategies: " << betterCount << " / " << numPermutations << "\n";
    std::cout << " - P-Value: " << pValue << "\n"; // 보통 0.05 미만이면 유의미(Significant)

    if (pValue < 0.05) 
    {
        std::cout << ">>> SUCCESS: The strategy is Statistically Significant! (Not Luck)\n";
    }
    else {
        std::cout << ">>> WARNING: The strategy might be due to randomness.\n";
    }
    std::cout << "======================================================\n";
}

void ExperimentManager::RunBestModelAndExportHeatmap(const CsvLoader::Dataset& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio)
{
    std::cout << "\n======================================================\n";
    std::cout << ">>> [Fast Track] Training BEST Model & Exporting Heatmap <<<\n";
    std::cout << "======================================================\n";

    // 1. 데이터 분할 (Train / Test) - 기존 로직 그대로 사용
    size_t totalRows = dataset.rows;
    size_t splitPoint = static_cast<size_t>(totalRows * splitRatio);
    size_t testSize = totalRows - splitPoint;

    // 2. DMatrix 생성
    DMatrixHandle hFullCls, hFullReg;
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullCls));
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullReg));

    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullCls, "label", dataset.labels.data(), totalRows));
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullReg, "label", dataset.returns.data(), totalRows));

    // Slice
    std::vector<int> allIndices(totalRows);
    std::iota(allIndices.begin(), allIndices.end(), 0);
    DMatrixHandle hTrainCls, hTestCls, hTrainReg, hTestReg;
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data(), splitPoint, &hTrainCls));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data() + splitPoint, testSize, &hTestCls));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data(), splitPoint, &hTrainReg));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullReg, allIndices.data() + splitPoint, testSize, &hTestReg));

    // 3. 모델 학습 (딱 한 번만 수행!)
    std::cout << ">>> Training Best Classification Model...\n";
    BoosterHandle hBoosterCls = TrainBooster(hTrainCls, bestClsConfig);

    std::cout << ">>> Training Best Regression Model...\n";
    BoosterHandle hBoosterReg = TrainBooster(hTrainReg, bestRegConfig);

    // 4. 예측 수행 (Predict)
    std::cout << ">>> Predicting Test Set...\n";
    const float* predPD;
    const float* predEstReturn;
    bst_ulong outLen;
    SAFE_XGBOOST(XGBoosterPredict(hBoosterCls, hTestCls, 0, 0, 0, &outLen, &predPD));
    SAFE_XGBOOST(XGBoosterPredict(hBoosterReg, hTestReg, 0, 0, 0, &outLen, &predEstReturn));

    // 5. 정답 데이터 준비 (Test Set)
    std::vector<float> testActualReturns(dataset.returns.begin() + splitPoint, dataset.returns.end());
    std::vector<float> testBonds(dataset.bondYields.begin() + splitPoint, dataset.bondYields.end());


    // =========================================================
    // [핵심] 히트맵 데이터 생성 (Grid Simulation)
    // =========================================================
    std::cout << ">>> Generating Heatmap Data (No Retraining)...\n";

    std::ofstream outFile("heatmap_data.csv");
    outFile << "PD_Threshold,Return_Threshold,Sharpe_Ratio,Approved_Count,Approved_Rate\n";

    // 검색 범위 설정 (팀원 요청: 촘촘하게)

    int totalCombinations = 0;

    // [최적화 Tip] 루프 밖에서 미리 메모리를 확보하여 재할당 오버헤드 제거
    std::vector<bool> tempApproval;
    tempApproval.reserve(testSize);

    // pdTh: 0.01 ~ 0.40 (0.01 단위 -> 40회 반복)
    for (int i = 1; i <= 40; ++i)
    {
        float pdTh = i * 0.01f;

        // retTh: 0.005 ~ 0.100 (0.005 단위 -> 20회 반복)
        for (int j = 1; j <= 20; ++j)
        {
            float retTh = j * 0.005f;

            // 벡터 초기화 (메모리는 유지하고 크기만 0으로)
            tempApproval.clear();

            // 기존 로직 수행
            int count = 0;
            for (size_t k = 0; k < testSize; ++k)
            {
                // 주의: 실수 비교이므로 데이터 정밀도에 따라 
                // 등호(<=, >=) 처리가 필요한지 확인 필요. 여기선 기존 로직(<, >) 유지.
                bool pass = (predPD[k] < pdTh) && (predEstReturn[k] > retTh);

                tempApproval.push_back(pass);
                if (pass) count++;
            }

            // 샤프지수 계산
            float sharpe = 0.0f;
            if (count > 10)
            {
                sharpe = CalculateSharpeRatioVer2(testActualReturns, testBonds, tempApproval);
            }

            // CSV 저장
            float rate = (testSize > 0) ? ((float)count / testSize * 100.0f) : 0.0f;
            outFile << pdTh << "," << retTh << "," << sharpe << "," << count << "," << rate << "\n";

            totalCombinations++;
        }
    }

    outFile.close();
    std::cout << ">>> Heatmap Data Saved to 'heatmap_data.csv' (" << totalCombinations << " combinations)\n";


    std::vector<float> vecPredPD(predPD, predPD + outLen);
    std::vector<float> vecPredRet(predEstReturn, predEstReturn + outLen);
    
    float bestPdParam = 0.20f;
    float bestRetParam = 0.075f;
    PerformRandomPermutationTest(
        vecPredPD,
        vecPredRet,
        testActualReturns,
        testBonds,
        bestPdParam,
        bestRetParam,
        1000 // 1000번 반복
    );
    // 6. 메모리 정리
    XGDMatrixFree(hFullCls); XGDMatrixFree(hFullReg);
    XGDMatrixFree(hTrainCls); XGDMatrixFree(hTestCls);
    XGDMatrixFree(hTrainReg); XGDMatrixFree(hTestReg);
    XGBoosterFree(hBoosterCls); XGBoosterFree(hBoosterReg);

    std::cout << ">>> Done.\n";
}
