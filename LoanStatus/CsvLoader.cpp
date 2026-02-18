#include "pch.h"
#include "CsvLoader.h"

CsvLoader::DataSet CsvLoader::Load()
{
    std::ifstream file(fName);
    if (!file.is_open())
    {
        throw std::runtime_error("File not found: " + fName);
    }
    featureNames.clear();

    // 1. 헤더 파싱
    auto headerLine = std::string{};
    std::getline(file, headerLine);
    
    if (!headerLine.empty() && headerLine.back() == '\r')
    {
        headerLine.pop_back();
    }

    auto headers = split(headerLine, ',');

    auto featureIndices = std::vector<int>{};
    int targetIdx = -1;
    int returnIdx = -1;
    int bondIdx = -1;



    std::cout << "[CsvLoader] Configuring columns...\n";
    for (int i = 0; i < headers.size(); ++i)
    {
        auto& col = headers[i];
        col.erase(std::remove(col.begin(), col.end(), '\r'), col.end());

        if (col == targetCol)
        {
            targetIdx = i;
        }
        
        if (col == "Return") //샤프 계산할 때 쓸 Return 값
        {
            returnIdx = i;
        }
        
        if (col == "Bond")
        {
            bondIdx = i;
        }

        // Target 변수는 피처에 넣지 않음. 
        // ignoreCol에 있는 것도 넣지 않음.
        // (Bond는 ignoreCol에 없으므로 여기서 피처로 추가됨)
        if (col != targetCol && ignoreCol.find(col) == ignoreCol.end())
        {
            featureIndices.push_back(i);
            featureNames.push_back(col);
        }
    }

    std::cout << "\n";

    if (targetIdx == -1)
    {
        throw std::runtime_error("Target column not found: " + targetCol);
    }

    // 2. 파일 크기 및 스레드 분할 계산
    auto headerSize = (size_t)file.tellg();
    file.seekg(0, std::ios::end);
    auto fileSize = static_cast<size_t>(file.tellg());
    

    auto numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0)
    {
        numThreads = 4;
    }

    std::cout << "[CsvLoader] Spawning " << numThreads << " threads for parsing...\n";

    std::vector<std::thread> threads;
    std::vector<DataSet> threadResults(numThreads);

    auto chunkSize = (size_t)((fileSize - headerSize) / numThreads);
    auto currentPos = headerSize;

    // 3. 워커 스레드 실행 (생성 루프)
    for (unsigned int i = 0; i < numThreads; ++i)
    {
        auto endPos = (i == numThreads - 1) ? fileSize : (currentPos + chunkSize);

        // 멤버 함수를 스레드로 실행할 땐 'this' 포인터를 넘겨야 함
        threads.emplace_back(&CsvLoader::ParseWorker, this,
            currentPos,
            endPos,
            headerSize,
            std::cref(featureIndices),
            targetIdx,
            returnIdx, bondIdx,
            std::ref(threadResults[i]));

        currentPos = endPos;
    }

    // 4. 스레드 대기 및 결과 병합 (Join 루프)
    DataSet finalData;
    finalData.cols = static_cast<int>(featureIndices.size());

    for (unsigned int i = 0; i < numThreads; ++i)
    {
        if (threads[i].joinable())
        {
            threads[i].join();
        }

        auto& res = threadResults[i];
        // 벡터 이어 붙이기
        finalData.features.insert(finalData.features.end(), res.features.begin(), res.features.end());
        finalData.labels.insert(finalData.labels.end(), res.labels.begin(), res.labels.end());

        // return값 bond값도 병합
        finalData.returns.insert(finalData.returns.end(), res.returns.begin(), res.returns.end());
        finalData.bondYields.insert(finalData.bondYields.end(), res.bondYields.begin(), res.bondYields.end());
        finalData.rows += res.rows;
    }

    return finalData;
}

CsvLoader::DataPack CsvLoader::LoadAndSplit(float trainRatio, float valRatio)
{
    std::cout << ">>> [CsvLoader] Loading raw data...\n";
    DataSet raw = Load();

    std::cout << ">>> [CsvLoader] Shuffling and Splitting (Train: "
        << trainRatio << ", Val: " << valRatio << ")... \n";

    auto totalRows = raw.rows;
    auto numCols = raw.cols;

    // 2. 인덱스 셔플링: 0부터 N-1까지 숫자를 만들고 섞음
    std::vector<int> indices(totalRows);
    std::iota(indices.begin(), indices.end(), 0);

    // fixed random seed
    std::mt19937 g(42);
    std::shuffle(indices.begin(), indices.end(), g);

    auto trainCount = static_cast<int>(totalRows * trainRatio);
    auto validCount = static_cast<int>(totalRows * valRatio);
    auto testCount = totalRows - trainCount - validCount;

    DataPack pack;

    pack.train.features.reserve(trainCount * numCols);
    pack.train.labels.reserve(trainCount);
    pack.train.rows = 0; pack.train.cols = numCols;

    pack.val.features.reserve(validCount * numCols);
    pack.val.labels.reserve(validCount);
    pack.val.rows = 0; pack.val.cols = numCols;

   
    pack.test.features.reserve(testCount * numCols);
    pack.test.labels.reserve(testCount);
    pack.test.rows = 0; pack.test.cols = numCols;

    // 4. 데이터 분배 (셔플된 인덱스 순서대로)
    for (auto i = 0; i < totalRows; ++i)
    {
        auto originalIdx = indices[i]; // 섞인 인덱스

        // 어떤 셋에 넣을지 결정
        DataSet* target = nullptr;
        if (i < trainCount) target = &pack.train;
        else if (i < trainCount + validCount) target = &pack.val;
        else target = &pack.test;

        // Data Copy
        // 4-1. Label, Return, Bond (단일 값)
        target->labels.push_back(raw.labels[originalIdx]);
        target->returns.push_back(raw.returns[originalIdx]);
        target->bondYields.push_back(raw.bondYields[originalIdx]);

        // 4-2. Features (1D 배열에서 해당 행의 구간 복사)
        // features 벡터는 [row0_col0, row0_col1, ..., row1_col0 ...] 순서임
        auto startPos = (size_t)originalIdx * numCols;
        auto endPos = startPos + numCols;

        target->features.insert(
            target->features.end(),
            raw.features.begin() + startPos,
            raw.features.begin() + endPos
        );

        target->rows++;
    }

    std::cout << ">>> Split Complete.\n";
    std::cout << " - Train Rows: " << pack.train.rows << "\n";
    std::cout << " - Val Rows:   " << pack.val.rows << "\n";
    std::cout << " - Test Rows:  " << pack.test.rows << "\n";

    return pack;
}

void CsvLoader::ParseWorker(size_t start, size_t end, size_t headerEndPos, const std::vector<int>& featureIndices, int targetIdx, int returnIdx, int bondIdx, DataSet& outResult)
{
    // 각 스레드는 파일 스트림을 별도로 열어야 안전함 (읽기 전용이라도 seekg 충돌 방지)
    std::ifstream file(fName);
    file.seekg(start);

    auto line = std::string();

    // 첫 번째 청크(0번 스레드)가 아니면, 이전 청크의 마지막 줄이 넘어왔을 수 있으므로 한 줄 버림
    // (단, start 위치가 정확히 개행문자 다음이라는 보장이 없으므로 안전하게 처리)
    // 0번 스레드는 건너뛰지 않음.중간에서 시작하는 1~N번 스레드만 앞의 잘린 라인을 버림.
    if (start > headerEndPos)
    {
        std::getline(file, line);
    }

    auto estRows = (size_t)((end - start) / 100);
    if (estRows > 0)
    {
        outResult.features.reserve(estRows * featureIndices.size());
        outResult.labels.reserve(estRows);

        outResult.returns.reserve(estRows);
        outResult.bondYields.reserve(estRows);
    }


    // Features
    // 예외 발생 시 rollback이 귀찮으니 try 블록 안에서 다 처리
    std::vector<float> rowFeatures;
    rowFeatures.reserve(featureIndices.size());

    // 내 구역(end)까지만 읽기
    while (file.tellg() < (std::streampos)end && std::getline(file, line))
    {
        if (line.empty()) continue;

        if (line.back() == '\r')
        {
            line.pop_back();
        }

        auto tokens = split(line, ',');
        if (tokens.size() <= targetIdx) continue;

        try
        {
            // Label
            auto label = std::stof(tokens[targetIdx]);
            float valReturn = 0.0f;
            float valBond = 0.0f;

            if (returnIdx != -1 && returnIdx < tokens.size() && !tokens[returnIdx].empty())
                valReturn = std::stof(tokens[returnIdx]);

            if (bondIdx != -1 && bondIdx < tokens.size() && !tokens[bondIdx].empty())
                valBond = std::stof(tokens[bondIdx]);

            rowFeatures.clear();
            for (auto idx : featureIndices)
            {
                float val = std::numeric_limits<float>::quiet_NaN();
                if (idx < tokens.size() && !tokens[idx].empty())
                {
                    val = std::stof(tokens[idx]);
                }
                rowFeatures.push_back(val);
            }

            // 여기까지 에러 없이 왔으면 저장
            outResult.features.insert(outResult.features.end(), rowFeatures.begin(), rowFeatures.end());
            
            outResult.labels.push_back(label);
            outResult.returns.push_back(valReturn);
            outResult.bondYields.push_back(valBond);
            outResult.rows++;
        }
        catch (...)
        {
            std::cerr << "Parsing Error Skip this col.\n";
            continue;
        }
    }
}
