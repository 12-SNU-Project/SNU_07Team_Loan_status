#pragma once

class CsvLoader
{
public:
    struct Dataset
    {
        std::vector<float> features;
        std::vector<float> labels;

        // 검증용 값들(샤프 지수 계산용)
        std::vector<float> returns;     // "Return"
        std::vector<float> bondYields;  // "Bond"
       

        int rows = 0;
        int cols = 0;
    };

    CsvLoader(std::string filename, std::string targetColName, std::set<std::string> ignoreColNames)
        : fName(std::move(filename)),
        targetCol(std::move(targetColName)),
        ignoreCol(std::move(ignoreColNames))
    {
    }
    const std::vector<std::string>& GetFeatureNames() const
    {
        return featureNames;
    }
    // 메인 로딩 함수
    Dataset Load();

private:
    std::string fName;
    std::string targetCol;
    std::set<std::string> ignoreCol;
    std::vector<std::string> featureNames;

    static std::vector<std::string> split(const std::string& s, char delimiter)
    {
        std::vector<std::string> tokens;
        tokens.reserve(30);
        size_t start = 0;
        auto end = s.find(delimiter);
        while (end != std::string::npos)
        {
            tokens.push_back(s.substr(start, end - start));
            start = end + 1;
            end = s.find(delimiter, start);
        }
        tokens.push_back(s.substr(start));
        return tokens;
    }

    // 워커 스레드 함수
    void ParseWorker(size_t start, size_t end, size_t headerEndPos, const std::vector<int>& featureIndices, int targetIdx, int returnIdx, int bondIdx, Dataset& outResult);

};


