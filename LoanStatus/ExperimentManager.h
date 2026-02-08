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
};

// 2. 실험 결과
struct ExperimentResult 
{
    ModelConfig cls;
    ModelConfig reg;
    float finalScore;
};

class ExperimentManager 
{
public:
    // Grid Search 후보군 (lowerCamelCase 적용)
    std::vector<int> candidateDepths = { 3, 6 };
    std::vector<float> candidateEtas = { 0.1f, 0.05f };
    std::vector<int> candidateRounds = { 100 };

    // 목표 설정 (부도 확률 예측용)
    std::string targetObjective = "binary:logistic";
    std::string targetMetric = "auc";

    // Grid Search 조합 생성
    std::vector<ModelConfig> GenerateGrid() 
    {
        std::vector<ModelConfig> configs;
        auto idCounter = 1;

        for (auto depth : candidateDepths) 
        {
            for (auto eta : candidateEtas) 
            {
                for (auto round : candidateRounds) 
                {
                    configs.push_back({ idCounter++, depth, eta, round, targetObjective, targetMetric });
                }
            }
        }
        return configs;
    }

    // --------------------------------------------------------------------------
    // 1. 모델 구조 분석
    // --------------------------------------------------------------------------
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

 
    // 3. 샤프지수 계산 공식
    float CalculateSharpeRatio(
        const std::vector<float>& y_test,    // 실제 수익률 (Return)
        const std::vector<float>& bond_test, // 채권 수익률 (Bond)
        const std::vector<bool>& is_approved // 승인 여부 (Threshold 통과)
    );

    // 2-Ver1.  단순 부도확률만 구한 후에 샤프지수 계산하기
    // - fullReturns, fullBonds: 전체 데이터셋의 수익률/채권정보
    // - approvalThreshold: 부도 확률이 이 값보다 낮아야 승인 (예: 0.05 = 5% 미만 부도확률만 승인)
    void RunSingleModelValidation(
        const CsvLoader::Dataset& dataset, // 전체 데이터셋 객체 전달
        const ModelConfig& clsConfig,
        float splitRatio = 0.8f, // 8:2로 분류하기
        float pdThreshold = 0.1f // 10% 미만 부도확률인 애들만 태우겠다.
    );

    // 2-Ver2. 추정수익률 + 부도확률까지 필터링한 후 샤프지수 계산하기
    float RunDualModelValidation(
        const CsvLoader::Dataset& dataset, // 전체 데이터셋 객체 전달
        const ModelConfig& clsConfig,     // 분류 모델 설정
        const ModelConfig& regConfig,     // 회귀 모델 설정
        float splitRatio = 0.8f,
        float pdThreshold = 0.1f,         // 부도 확률 임계값
        float estReturnThreshold = 0.05f  // 추정 수익률 임계값
    );

    // 일부 필드값은 intger 취급
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

        // 사용할 스레드 수 계산
        // 사용할 스레드 수 안전하게 계산
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 4;

        // 전체 코어의 80% 정도만 쓰거나, 최소 1개는 보장하도록 설정
        int nThread = (int)numThreads - 1;
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




