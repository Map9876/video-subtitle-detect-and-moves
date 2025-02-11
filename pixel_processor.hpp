#ifndef PIXEL_PROCESSOR_HPP
#define PIXEL_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class PixelProcessor {
public:
    struct Pixel {
        int x;
        int y;
        cv::Vec3b color;
    };

    std::vector<Pixel> getWhitePixels(const cv::Mat& frame, const cv::Rect& roi) {
        std::vector<Pixel> whitePixels;
        cv::Mat roiMat = frame(roi);

        for (int y = 0; y < roiMat.rows; ++y) {
            for (int x = 0; x < roiMat.cols; ++x) {
                cv::Vec3b pixel = roiMat.at<cv::Vec3b>(y, x);
                // Check if pixel is close to white
                if (pixel[0] > 230 && pixel[1] > 230 && pixel[2] > 230) {
                    whitePixels.push_back({x, y + roi.y, pixel});
                }
            }
        }

        return whitePixels;
    }

    void shiftSubtitles(cv::Mat& frame, const std::vector<Pixel>& whitePixels, int shiftAmount) {
        // Create a copy of the frame
        cv::Mat result = frame.clone();

        // First, fill original positions with black
        for (const auto& pixel : whitePixels) {
            if (pixel.y >= 0 && pixel.y < frame.rows) {
                result.at<cv::Vec3b>(pixel.y, pixel.x) = cv::Vec3b(0, 0, 0);
            }
        }

        // Then, draw shifted white pixels
        for (const auto& pixel : whitePixels) {
            int newY = pixel.y - shiftAmount;
            if (newY >= 0 && newY < frame.rows) {
                result.at<cv::Vec3b>(newY, pixel.x) = pixel.color;
            }
        }

        frame = result;
    }
};

#endif // PIXEL_PROCESSOR_HPP
