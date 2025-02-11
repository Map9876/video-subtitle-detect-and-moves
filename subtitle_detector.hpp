#ifndef SUBTITLE_DETECTOR_HPP
#define SUBTITLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <map>
#include <iostream>

class SubtitleDetector {
public:
    SubtitleDetector(cv::Rect roi) : roi_(roi) {}

    bool isSubtitleFrame(const cv::Mat& frame) {
        cv::Mat roiMat = frame(roi_);
        cv::Mat small;
        cv::resize(roiMat, small, cv::Size(8, 8));
        
        // Convert to grayscale if needed
        cv::Mat gray;
        if (small.channels() == 3) {
            cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = small;
        }

        // Calculate average brightness
        double avg = cv::mean(gray)[0];

        // Print debug information
        std::cout << "ROI average brightness: " << avg << std::endl;

        return avg > 30.0; // Threshold for subtitle detection
    }

    void detectSubtitleFrames(cv::VideoCapture& cap, 
                            int startFrame,
                            int endFrame,
                            std::map<int, std::pair<int, int>>& ranges) {
        cap.set(cv::CAP_PROP_POS_FRAMES, startFrame);
        
        int rangeStart = -1;
        bool inSubtitleRange = false;

        for (int frame = startFrame; frame < endFrame; ++frame) {
            cv::Mat currentFrame;
            if (!cap.read(currentFrame)) break;

            bool hasSubtitle = isSubtitleFrame(currentFrame);

            if (hasSubtitle && !inSubtitleRange) {
                rangeStart = frame;
                inSubtitleRange = true;
            } else if (!hasSubtitle && inSubtitleRange) {
                ranges[rangeStart] = std::make_pair(rangeStart, frame - 1);
                inSubtitleRange = false;
            }
        }

        // Handle case where subtitle extends to end of batch
        if (inSubtitleRange) {
            ranges[rangeStart] = std::make_pair(rangeStart, endFrame - 1);
        }
    }

    cv::Rect getROI() const { return roi_; }

private:
    cv::Rect roi_;
};

#endif // SUBTITLE_DETECTOR_HPP
