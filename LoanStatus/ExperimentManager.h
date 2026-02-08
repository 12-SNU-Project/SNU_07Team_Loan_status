#pragma once

class CsvLoader;

inline void SAFE_XGBOOST(int call)
{
    if (call != 0)
    {
        std::cerr << "XGBoost Error: " << XGBGetLastError() << std::endl;
        std::exit(1);
    }
}

// 1. 설정 구조체
struct ModelConfig 
{
    int id;                 // id 인덱스
    int maxDepth;           // maxDepth 깊이
    float eta;              // eta 학습률
    int numRound;           // numRound 이터레이션 횟수
    std::string objective;  // objective 목적 함수 logic:binary
    std::string evalMetric; // evalMetric 평가 기준

    float minChildWeight = 1.0f; // 잎사귀 최소 가중치
    float scalePosWeight = 1.0f; // 불균형 데이터 가중치
    float subsample = 1.0f;      // 행 샘플링 비율 (Row Subsampling)
    float colsample = 1.0f;      // 열 샘플링 비율 (Col Subsampling)
};

// 2. 실험 결과

struct ValidationMetrics
{
    float sharpeRatio;      // 샤프 지수
    float avgReturn;        // 승인된 포트폴리오의 평균 수익률
    float avgPD;            // 승인된 포트폴리오의 평균 부도확률(Predicted PD)
    int approvedCount;      // 승인된 대출 건수

    float bestPDThreshold;     // (Auto 모드에서 사용)
    float bestReturnThreshold; // (Auto 모드에서 사용)
};

struct ExperimentResult
{
    ModelConfig bestClsConfig;
    ModelConfig bestRegConfig;
    ValidationMetrics bestMetrics; // 최고 기록의 상세 지표
};

class ExperimentManager 
{
public:
    // 1. 깊이 (Depth): 
     // - 3: 얕음 (회귀 모델의 과적합 방지에 좋음)
     // - 5: 중간 (가장 일반적인 시작점)
     // - 7: 깊음 (분류 모델의 복잡한 패턴 학습용)
    std::vector<int> candidateDepths = { 5, 7 };

    // 2. 학습률 (Eta): 
    // - 0.1: 빠름 (초반 탐색용)
    // - 0.05: 정교함 (가장 권장됨)
    // - 0.01: 매우 정교함 (시간 오래 걸림, 미세 튜닝용)
    std::vector<float> candidateEtas = { 0.1f, 0.05f, 0.01f };

   
    // 목표 설정 (부도 확률 예측용)
    std::string targetObjective = "binary:logistic";
    std::string targetMetric = "auc";

    // 0. Grid Search 조합 생성
    std::vector<ModelConfig> GenerateGrid(bool isClassification)
    {
        std::vector<ModelConfig> configs;
        int idCounter = 1;

        // 1. 모델 성격(분류 vs 회귀)에 따른 기본 설정
        std::string obj = isClassification ? "binary:logistic" : "reg:squarederror";
        std::string metric = isClassification ? "auc" : "rmse";

        // 분류(불균형 데이터)는 양성 클래스에 가중치 4배, 회귀는 1배
        float scaleWeight = isClassification ? 4.0f : 1.0f;

        // 분류는 미세한 패턴(1.0)까지, 회귀는 굵직한 패턴(5.0)만 학습
        float minChild = isClassification ? 1.0f : 5.0f;

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

    // Ver 1. 고정 스레드홀드 기준
    ExperimentResult RunGridSearchFixed(
        const CsvLoader::Dataset& dataset,
        float splitRatio,
        float pdThreshold,
        float estReturnThreshold);

    // Ver 2. 자동 임계값 최적화(Auto Optimization)로 Grid Search (이제 Threshold 인자가 필요 없음)
    ExperimentResult RunGridSearchAuto(const CsvLoader::Dataset& dataset, float splitRatio);

    // Ver 3. 모델 학습 + 내부 임계값 최적화까지 한 방에 수행하는 함수
    ValidationMetrics RunDualModelValidationAndOptimizeThreshold(
        const CsvLoader::Dataset& dataset,
        const ModelConfig& clsConfig,
        const ModelConfig& regConfig,
        float splitRatio
    );

    // -----------------------------------------------------------
    // [내부 엔진 함수]
    // -----------------------------------------------------------
 
    // 엔진 1: 고정된 임계값으로 검증(Ver 1에서 사용) 단순 부도 확률만 봄
    void RunSingleModelValidation(
        const CsvLoader::Dataset& dataset, // 전체 데이터셋 객체 전달
        const ModelConfig& clsConfig,
        float splitRatio = 0.8f, // 8:2로 분류하기
        float pdThreshold = 0.1f // 10% 미만 부도확률인 애들만 태우겠다.
    );

    // 엔진 2: 내부에서 최적 임계값을 찾으며 검증 (Ver 2에서 사용)
    ValidationMetrics RunDualModelValidation(
        const CsvLoader::Dataset& dataset, // 전체 데이터셋 객체 전달
        const ModelConfig& clsConfig,     // 분류 모델 설정
        const ModelConfig& regConfig,     // 회귀 모델 설정
        float splitRatio = 0.8f,
        float pdThreshold = 0.1f,         // 부도 확률 임계값
        float estReturnThreshold = 0.05f  // 추정 수익률 임계값
    );


    // -----------------------------------------------------------
    // [유틸리티 함수]
    // -----------------------------------------------------------
 
    // XGBoost 모델 구조 분석
    void AnalyzeModel(BoosterHandle hBooster, const std::vector<std::string>& featureNames)
    {
        bst_ulong outLen;
        const char** outDumpArray;
        SAFE_XGBOOST(XGBoosterDumpModel(hBooster, "", 0, &outLen, &outDumpArray));

        if (outLen == 0) return;

        std::cout << "\n================ [Model Analysis Report] ================\n";
        std::string tree0 = outDumpArray[0];
        std::cout << "[Tree 0 Root Analysis]\n";
        auto start = tree0.find("[f");
        if (start != std::string::npos)
        {
            auto end = tree0.find("<", start);
            if (end != std::string::npos)
            {
                auto featKey = tree0.substr(start + 1, end - (start + 1));
                try
                {
                    auto featIdx = std::stoi(featKey.substr(1));
                    if (featIdx < featureNames.size())
                    {
                        std::cout << "  >>> KING FEATURE: '" << featureNames[featIdx] << "'\n";
                    }
                }
                catch (...) {}
            }
        }

        std::ofstream dumpFile("model_dump_pretty.txt");
        dumpFile << "=== XGBoost Model Tree Structure (Pretty Print) ===\n\n";

        for (bst_ulong i = 0; i < outLen; ++i)
        {
            dumpFile << "Tree [" << i << "]:\n";

            // 원본 문자열: "0:[f2<700]..." 또는 "\t1:[f0<5000]..."
            std::string rawTree = outDumpArray[i];
            std::stringstream ss(rawTree);
            std::string line;

            while (std::getline(ss, line))
            {
                if (line.empty()) continue;

                int depth = 0;
                while (depth < line.length() && line[depth] == '\t')
                {
                    depth++;
                }

                std::string prefix = "";
                for (int k = 0; k < depth; ++k) prefix += "   |   "; // 깊이 줄기
                prefix += "|-- "; // 가지


                std::string content = line.substr(depth);

                // (4) 변수명 치환 (f0 -> annual_inc)
                // content 안에 있는 "f숫자"를 찾아서 바꿈
                for (int fIdx = (int)featureNames.size() - 1; fIdx >= 0; --fIdx)
                {
                    std::string key = "f" + std::to_string(fIdx);
                    std::string name = featureNames[fIdx];

                    size_t pos = 0;
                    while ((pos = content.find(key, pos)) != std::string::npos)
                    {
                        char nextChar = (pos + key.length() < content.length()) ? content[pos + key.length()] : ' ';
                        if (!isdigit(nextChar))
                        {
                            content.replace(pos, key.length(), name);
                            pos += name.length();
                        }
                        else
                        {
                            pos++;
                        }
                    }
                }
                dumpFile << prefix << content << "\n";
            }
            dumpFile << "\n"; // 트리 간 간격
        }

        dumpFile.close();
        std::cout << "  >>> Detailed Tree Structure saved to 'model_dump_pretty.txt'\n";
        std::cout << "=========================================================\n";
    }

    // 샤프지수 계산 공식
    float CalculateSharpeRatio(
        const std::vector<float>& y_test,    // 실제 수익률 (Return)
        const std::vector<float>& bond_test, // 채권 수익률 (Bond)
        const std::vector<bool>& is_approved // 승인 여부 (Threshold 통과)
    )
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
    // 피처 타입 설정
    static void SetFeatureTypes(DMatrixHandle hData, const std::vector<std::string>& featureNames)
    {
        std::vector<const char*> types;
        types.reserve(featureNames.size());

        for (const auto& name : featureNames)
        {
            // 정수형 또는 범주형으로 취급할 필드들 정의
            if (name == "Actual_term" || name == "application_type" ||
                name == "sub_grade" || name == "Year" || name == "Month" ||
                name == "term" || name.find("is_") != std::string::npos ||
                name == "num_accts_ever_120_pd" || name == "pub_rec")
            {
                // "c"는 categorical(범주형), "int"는 정수형을 의미합니다.
                // 여기서는 정수 특성이 강하므로 "int"를 사용하거나 범주형일 경우 "c"를 씁니다.
                types.push_back("int");
            }
            else
            {
                types.push_back("float"); // 나머지는 연속형 실수(quantitative)
            }
        }

        // XGBoost에게 각 컬럼의 데이터 타입을 알려줌
        SAFE_XGBOOST(XGDMatrixSetStrFeatureInfo(hData, "feature_type", types.data(), (bst_ulong)types.size()));

        // 추가로 피처 이름도 등록해주면 AnalyzeModel 결과가 더 정확.
        std::vector<const char*> names;
        for (const auto& s : featureNames) names.push_back(s.c_str());
        SAFE_XGBOOST(XGDMatrixSetStrFeatureInfo(hData, "feature_name", names.data(), (bst_ulong)names.size()));

        std::cout << " -> Feature types and names configured successfully.\n";
    }
private:
    
    BoosterHandle TrainBooster(DMatrixHandle hTrain, const ModelConfig& config)
    {
        BoosterHandle hBooster;
        SAFE_XGBOOST(XGBoosterCreate(&hTrain, 1, &hBooster));
        SAFE_XGBOOST(XGBoosterSetParam(hBooster, "tree_method", "hist"));
        SAFE_XGBOOST(XGBoosterSetParam(hBooster, "objective", config.objective.c_str()));
        SAFE_XGBOOST(XGBoosterSetParam(hBooster, "max_depth", std::to_string(config.maxDepth).c_str()));
        SAFE_XGBOOST(XGBoosterSetParam(hBooster, "eta", std::to_string(config.eta).c_str()));


        // [추가 파라미터 바인딩]
        SAFE_XGBOOST(XGBoosterSetParam(hBooster, "min_child_weight", std::to_string(config.minChildWeight).c_str()));
        SAFE_XGBOOST(XGBoosterSetParam(hBooster, "scale_pos_weight", std::to_string(config.scalePosWeight).c_str()));
        SAFE_XGBOOST(XGBoosterSetParam(hBooster, "subsample", std::to_string(config.subsample).c_str()));
        SAFE_XGBOOST(XGBoosterSetParam(hBooster, "colsample_bytree", std::to_string(config.colsample).c_str()));

        // 사용할 스레드 수 계산
        // 사용할 스레드 수 안전하게 계산
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 4;

        // 전체 코어의 80% 정도만 쓰거나, 최소 1개는 보장하도록 설정
        int nThread = (int)(numThreads / 2);
        if (nThread < 1) nThread = 1;

        // nthread 파라미터 할당
        SAFE_XGBOOST(XGBoosterSetParam(hBooster, "nthread", std::to_string(nThread).c_str()));

        for (int i = 0; i < config.numRound; ++i)
        {
            SAFE_XGBOOST(XGBoosterUpdateOneIter(hBooster, i, hTrain));
        }

        return hBooster;
    }
};




