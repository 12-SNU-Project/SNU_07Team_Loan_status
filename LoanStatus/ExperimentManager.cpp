#include "pch.h"
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
    // --- [Part A: 모델 학습 (기존과 동일)] ---
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


    double sumPD = 0, minPD = 1.0, maxPD = 0.0;
    double sumRet = 0, minRet = 1.0, maxRet = -1.0;

    // 샘플링으로 앞쪽 1000개만 보거나 전체 루프 돌림
    for (size_t i = 0; i < testSize; ++i) 
    {
        if (predPD[i] < minPD) minPD = predPD[i];
        if (predPD[i] > maxPD) maxPD = predPD[i];
        sumPD += predPD[i];

        if (predEstReturn[i] < minRet) minRet = predEstReturn[i];
        if (predEstReturn[i] > maxRet) maxRet = predEstReturn[i];
        sumRet += predEstReturn[i];
    }

    // 첫 번째 이터레이션에서만 로그 출력 (너무 많이 뜨면 보기 힘드니까)
    static bool bPrintedDebug = false;
    if (!bPrintedDebug) 
    {
        std::cout << "\n>>>[DEBUG MODEL OUTPUT]\n";
        std::cout << " - PD Range     : " << minPD << " ~ " << maxPD << " (Avg: " << (double)(sumPD / testSize) << ")\n";
        std::cout << " - Return Range : " << minRet << " ~ " << maxRet << " (Avg: " << (double)(sumRet / testSize) << ")\n";
        bPrintedDebug = true;
    }


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
    std::vector<float> pdCandidates = { 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.40f, 0.50f };
    std::vector<float> retCandidates = { 0.0f, 0.02f, 0.04f, 0.05f, 0.06f, 0.08f };

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

            float currentSharpe = CalculateSharpeRatio(testActualReturns, testBonds, isApproved);

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

    float sharpe = CalculateSharpeRatio(testActualReturns, testBonds, isApproved);


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
