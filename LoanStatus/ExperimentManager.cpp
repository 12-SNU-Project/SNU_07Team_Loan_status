#include "pch.h"
#include "CsvLoader.h"
#include "ExperimentManager.h"

float ExperimentManager::CalculateSharpeRatio(const std::vector<float>& y_test, const std::vector<float>& bond_test, const std::vector<bool>& is_approved)
{
    const size_t n = y_test.size();
    if (n <= 1) return 0.0f;

    float sum_excess = 0.0f;
    float sum_std_val = 0.0f;

    // 평균 계산을 위한 데이터 모음
    std::vector<float> accepted_returns;
    accepted_returns.reserve(n);

    for (size_t i = 0; i < n; ++i)
    {
        float val = 0.0f;
        if (is_approved[i])
        {
            // 승인됨: 실제 수익률 - 무위험 이자율(Bond) = 초과 수익
            val = y_test[i];
            sum_excess += (y_test[i] - bond_test[i]);
        }
        else {
            // 거절됨: 채권 수익률로 대체 (기회비용 관점) 또는 0 처리
            // 여기서는 "거절 시 채권 투자"라고 가정하면 초과수익은 0이 됨 (Bond - Bond)
            val = bond_test[i];
        }

        accepted_returns.push_back(val);
        sum_std_val += val;
    }

    // 1. 평균 계산
    float mean_excess = sum_excess / static_cast<float>(n);
    float mean_portfolio = sum_std_val / static_cast<float>(n);

    // 2. 표준편차 계산
    float sq_diff_sum = 0.0f;
    for (float val : accepted_returns)
    {
        float diff = val - mean_portfolio;
        sq_diff_sum += diff * diff;
    }

    float std_dev = std::sqrt(sq_diff_sum / static_cast<float>(n - 1));

    // 3. Sharpe Ratio 반환
    return (std_dev > 1e-6f) ? (mean_excess / std_dev) : 0.0f;
}

void ExperimentManager::RunSingleModelValidation(const CsvLoader::Dataset& dataset, const ModelConfig& clsConfig,  float splitRatio, float pdThreshold)
{
    std::cout << "\n>>> [Single Model Validation] Sequential Split & Analysis...\n";


    // 1. 데이터 분할 (Sequential Split)
    size_t totalRows = dataset.rows;
    size_t splitPoint = static_cast<size_t>(totalRows * splitRatio);
    size_t testSize = totalRows - splitPoint;

    DMatrixHandle hFullCls;
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    SAFE_XGBOOST(XGDMatrixCreateFromMat(dataset.features.data(), totalRows, dataset.cols, nanVal, &hFullCls));
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(hFullCls, "label", dataset.labels.data(), totalRows));

    // Slice (Train/Test 분리)
    std::vector<int> allIndices(totalRows);
    std::iota(allIndices.begin(), allIndices.end(), 0);
    DMatrixHandle hTrainCls, hTestCls;
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data(), splitPoint, &hTrainCls));
    SAFE_XGBOOST(XGDMatrixSliceDMatrix(hFullCls, allIndices.data() + splitPoint, testSize, &hTestCls));


    // 2. 모델 학습
    std::cout << ">>> Training Model (Past Data)...\n";
    BoosterHandle hBoosterCls = TrainBooster(hTrainCls, clsConfig);


    // 3. 예측 수행
    std::cout << ">>> Predicting (Future Data)...\n";
    const float* predPD;
    bst_ulong outLenCls;
    SAFE_XGBOOST(XGBoosterPredict(hBoosterCls, hTestCls, 0, 0, 0, &outLenCls, &predPD));


    // 4. 필터링 적용
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

    // 5. Sharpe Ratio 계산 호출
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
