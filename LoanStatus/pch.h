#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <thread>
#include <mutex>
#include <set>
#include <algorithm>
#include <numeric>
#include <xgboost/c_api.h>
#include <iomanip>
#include <functional>
#define NOMINMAX 
#include <Windows.h>
#include <cmath>
#include <numeric>


//void RunAll(DMatrixHandle hTrain, const std::string& logFilename = "experiment_results.csv")
//{
//    auto configs = GenerateGrid();
//
//    std::cout << ">>> [Manager] Total " << configs.size() << " experiments queued.\n";
//
//    std::ofstream out(logFilename);
//    out << "ID, Depth, Eta, Rounds, Objective, Score\n";
//
//    for (const auto& conf : configs)
//    {
//        std::cout << "\n[Exp " << conf.id << "] Depth=" << conf.maxDepth
//            << ", Eta=" << conf.eta << " Start...";
//
//        auto score = TrainAndEvaluate(hTrain, conf);
//
//        out << conf.id << "," << conf.maxDepth << "," << conf.eta << ","
//            << conf.numRound << "," << conf.objective << "," << score << "\n";
//
//        std::cout << " -> Score: " << score << '\n';
//    }
//
//    std::cout << "\n>>> [Manager] All done. Saved to '" << logFilename << "'\n";
//    out.close();
//}
//void PredictAll(DMatrixHandle hTrain, const ModelConfig& bestConfig,
//    const std::vector<std::string>& featureNames, // ★ 인자 추가됨
//    const std::string& outFilename = "predictions.csv")
//{
//    std::cout << "\n>>> [Manager] Generating predictions with Best Config (Depth="
//        << bestConfig.maxDepth << ", Eta=" << bestConfig.eta << ")...\n";
//
//    // 1. 모델 재학습
//    BoosterHandle hBooster;
//    SAFE_XGBOOST(XGBoosterCreate(&hTrain, 1, &hBooster));
//
//    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "device", "cpu"));
//    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "tree_method", "hist"));
//    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "objective", bestConfig.objective.c_str()));
//    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "max_depth", std::to_string(bestConfig.maxDepth).c_str()));
//    SAFE_XGBOOST(XGBoosterSetParam(hBooster, "eta", std::to_string(bestConfig.eta).c_str()));
//
//    for (auto i = 0; i < bestConfig.numRound; ++i)
//    {
//        SAFE_XGBOOST(XGBoosterUpdateOneIter(hBooster, i, hTrain));
//    }
//
//    AnalyzeModel(hBooster, featureNames);
//
//    // 2. 전체 데이터 예측
//    bst_ulong outLen;
//    const float* outResult;
//    SAFE_XGBOOST(XGBoosterPredict(hBooster, hTrain, 0, 0, 0, &outLen, &outResult));
//
//    // 3. CSV 저장
//    std::ofstream out(outFilename);
//    out << "RowIndex,Probability(Default)\n";
//
//    for (bst_ulong i = 0; i < outLen; ++i)
//    {
//        out << i << "," << outResult[i] << "\n";
//    }
//
//    std::cout << ">>> [Manager] Predictions saved to '" << outFilename << "' (" << outLen << " rows)\n";
//
//    SAFE_XGBOOST(XGBoosterFree(hBooster));
//}