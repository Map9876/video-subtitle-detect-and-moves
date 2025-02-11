#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <mutex>
#include "subtitle_detector.hpp"
#include "pixel_processor.hpp"

using json = nlohmann::json;
std::mutex mtx; // Mutex for thread safety

void processFrameRange(cv::VideoCapture& cap, cv::VideoWriter& writer, PixelProcessor& processor, SubtitleDetector& detector, int startFrame, int endFrame, const std::vector<int>& shiftAmounts, std::map<int, std::pair<int, int>>& subtitleRanges) {
    cap.set(cv::CAP_PROP_POS_FRAMES, startFrame);
    cv::Mat keyFrame;
    cap.read(keyFrame);
    auto whitePixels = processor.getWhitePixels(keyFrame, detector.getROI());

    for (int frame = startFrame; frame <= endFrame; ++frame) {
        cap.set(cv::CAP_PROP_POS_FRAMES, frame);
        cv::Mat currentFrame;
        cap.read(currentFrame);
        int shiftAmount = shiftAmounts[frame % shiftAmounts.size()];
        processor.shiftSubtitles(currentFrame, whitePixels, shiftAmount);
        writer.write(currentFrame);
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Processed frame " << frame << "/" << endFrame << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cout << "Usage: " << argv[0] << " <input_video> <output_video> <shift_amounts> <roi_coordinates> <num_threads>\n";
        std::cout << "Example: " << argv[0] << " input.mp4 output.mp4 70,60 0,720,1280,360 4\n";
        return -1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];

    // Parse shift amounts
    std::vector<int> shiftAmounts;
    std::stringstream ss(argv[3]);
    std::string item;
    while (std::getline(ss, item, ',')) {
        shiftAmounts.push_back(std::stoi(item));
    }

    // Parse ROI coordinates
    std::vector<int> roiCoords;
    std::stringstream ssRoi(argv[4]);
    while (std::getline(ssRoi, item, ',')) {
        roiCoords.push_back(std::stoi(item));
    }

    if (roiCoords.size() != 8) {
        std::cerr << "Error: ROI coordinates should be 8 values (x1, y1, x2, y2, x3, y3, x4, y4)\n";
        return -1;
    }

    cv::Rect roi(std::min(roiCoords[0], roiCoords[6]), 
                 std::min(roiCoords[1], roiCoords[7]), 
                 std::max(roiCoords[2], roiCoords[4]) - std::min(roiCoords[0], roiCoords[6]), 
                 std::max(roiCoords[3], roiCoords[5]) - std::min(roiCoords[1], roiCoords[7]));

    // Open video file
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file\n";
        return -1;
    }

    // Get video properties
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    // Create video writer
    cv::VideoWriter writer(outputPath, 
                         cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                         fps,
                         cv::Size(width, height));

    // Initialize subtitle detector
    SubtitleDetector detector(roi);

    // Save the first frame with ROI drawn
    cv::Mat firstFrame;
    if (cap.read(firstFrame)) {
        cv::polylines(firstFrame, std::vector<cv::Point>{{roiCoords[0], roiCoords[1]}, {roiCoords[2], roiCoords[3]}, {roiCoords[4], roiCoords[5]}, {roiCoords[6], roiCoords[7]}}, true, cv::Scalar(0, 255, 0), 2);
        cv::imwrite("roi_frame.jpg", firstFrame);
    } else {
        std::cerr << "Error reading the first frame\n";
        return -1;
    }

    // Detect subtitle frames
    std::map<int, std::pair<int, int>> subtitleRanges;
    detector.detectSubtitleFrames(cap, 0, totalFrames, subtitleRanges);

    // Convert subtitle ranges to JSON
    json rangesJson;
    for (const auto& range : subtitleRanges) {
        rangesJson[std::to_string(range.second.first)] = std::to_string(range.second.second);
    }

    // Process and shift subtitles
    PixelProcessor processor;
    int numThreads = std::stoi(argv[5]);
    std::vector<std::thread> threads;
    for (const auto& range : subtitleRanges) {
        int startFrame = range.second.first;
        int endFrame = range.second.second;

        threads.emplace_back(processFrameRange, std::ref(cap), std::ref(writer), std::ref(processor), std::ref(detector), startFrame, endFrame, std::ref(shiftAmounts), std::ref(subtitleRanges));
        if (threads.size() >= numThreads) {
            for (auto& thread : threads) {
                thread.join();
            }
            threads.clear();
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Clean up
    cap.release();
    writer.release();

    // Save ranges to JSON file
    std::ofstream o("subtitle_ranges.json");
    o << rangesJson.dump() << std::endl;

    return 0;
}
