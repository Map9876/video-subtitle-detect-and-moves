cmake_minimum_required(VERSION 3.10)
project(subtitle_shifter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(/storage/emulated/0/detect/json/include) //'nlohmann/json.hpp' 路径

add_executable(subtitle_shifter
    main.cpp
)

target_link_libraries(subtitle_shifter
    PRIVATE
    ${OpenCV_LIBS}
)
