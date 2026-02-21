cmake_minimum_required(VERSION 3.18)
project(XGBoost_Deploy)

# 1. 경로 정의
set(XGB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/external/xgboost")
set(OUT_INC "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(OUT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# 2. XGBoost 빌드 옵션
set(XGBOOST_BUILD_STATIC_LIB OFF CACHE BOOL "Build Shared (.dll)")
set(USE_OPENMP ON CACHE BOOL "Enable Multi-threading")

# 3. XGBoost 서브프로젝트 포함
add_subdirectory(${XGB_SRC} external/xgboost_build EXCLUDE_FROM_ALL)

# 4. 별도의 배포 타겟 생성 (에러 해결 핵심)
add_custom_target(deploy_xgboost
    COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_INC}/xgboost"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${XGB_SRC}/include/xgboost" "${OUT_INC}/xgboost"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_LIB}"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:xgboost> "${OUT_LIB}/"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_LINKER_FILE:xgboost> "${OUT_LIB}/"
    COMMENT "Deploying XGBoost v3.2.0 to /include and /lib..."
)

# xgboost 빌드가 끝난 후 배포가 실행되도록 의존성 설정
add_dependencies(deploy_xgboost xgboost)