#pragma once
class CsvLoader;

// XGBoost 에러 체크 매크로
inline void SAFE_XGBOOST(int call)
{
    if (call != 0)
    {
        std::cerr << "XGBoost Error: " << XGBGetLastError() << std::endl;
        std::exit(1);
    }
}

// 1. 모델 설정 구조체 (기존 유지)
struct ModelConfig
{
    int id;                 // id 인덱스
    int maxDepth;           // maxDepth 깊이
    float eta;              // eta 학습률
    int numRound;           // numRound 이터레이션 횟수
    std::string objective;  // objective 목적 함수
    std::string evalMetric; // evalMetric 평가 기준

    float minChildWeight = 1.0f; // 잎사귀 최소 가중치
    float scalePosWeight = 1.0f; // 불균형 데이터 가중치
    float subsample = 1.0f;      // 행 샘플링 비율
    float colsample = 1.0f;      // 열 샘플링 비율
};

// 2. 검증 결과 구조체 (기존 유지)
struct ValidationMetrics
{
    float sharpeRatio;      // 샤프 지수
    float avgReturn;        // 승인된 포트폴리오의 평균 수익률
    float avgPD;            // 승인된 포트폴리오의 평균 부도확률
    int approvedCount;      // 승인된 대출 건수
    float approvedRate;     // 승인율
};

struct ExperimentResult
{
    ModelConfig bestClsConfig;
    ModelConfig bestRegConfig;
    ValidationMetrics bestMetrics;
};

// [신규] 실험 데이터를 담는 컨텍스트 (중복 제거용)
struct ExperimentContext
{
    std::vector<float> predPD;          // 예측된 부도확률
    std::vector<float> predEstReturn;   // 예측된 수익률
    std::vector<float> actualReturns;   // 실제 수익률 (Test Set)
    std::vector<float> bondYields;      // 국채 금리 (Test Set)
    size_t testSize;                    // 테스트셋 크기
};

class ExperimentManager
{
public:
    ExperimentResult RunGridSearchFixed(const CsvLoader::DataSet& dataset, float splitRatio, float pdThreshold, float estReturnThreshold);

    // 모드 1: 10k 1회 샘플링 히트맵
    void RunHeatmap_SingleSample10k(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio);

    // 모드 2: 고정 파라미터 신뢰성 검증 (1,000번 반복)
    void RunReliabilityCheck_Bootstrap(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio);

    // 모드 3: 로버스트 히트맵 (모든 그리드 x 1,000번 반복)
    void RunHeatmap_RobustBootstrap(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio);

    // 모드 4: 전체 데이터셋 전수 조사 히트맵
    void RunHeatmap_FullTestSet(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio);

private:
    // 공통 로직 분리: 데이터 분할, 학습, 예측 결과를 반환
    ExperimentContext PrepareExperiment(const CsvLoader::DataSet& dataset, const ModelConfig& bestClsConfig, const ModelConfig& bestRegConfig, float splitRatio);

    // Helper 함수들
    BoosterHandle TrainBooster(DMatrixHandle hTrain, const ModelConfig& config);
    std::vector<ModelConfig> GenerateGrid(bool isClassification);
    float CalculateSharpeRatio(const std::vector<float>& returns, const std::vector<float>& bondYields, const std::vector<bool>& approval);
    void PerformRandomPermutationTest(const std::vector<float>& predPD, const std::vector<float>& predRet, const std::vector<float>& actualRet, const std::vector<float>& bondYields, float bestPDTh, float bestRetTh, int iterations);
private:

    // 1. 깊이 (Depth) 
    std::vector<int> candidateDepths = { 5, 7 };
    // 2. 학습률 (Eta)
    std::vector<float> candidateEtas = { 0.05f, 0.01f };
};