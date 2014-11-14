//
//  main.cpp
//  Lab 2
//
//  Created by Matt Donnelly on 13/11/2014.
//  Copyright (c) 2014 Matt Donnelly. All rights reserved.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define ESC_KEY 27
#define ACCEPTABLE_MOTION_PERCENTAGE 10.0
#define NUM_POSTBOXES 2
#define MIN_LINE_LENGTH 5
#define MIN_VISIBLE_LINES 7

const int postbox_locations[NUM_POSTBOXES][3] {
    // Start, End, Row
    {22,  98, 133},
    {127, 199, 132}
    //{54,  200, 502},
};

bool containsMotion(cv::Mat current_frame, cv::Mat previous_frame) {
    cv::Mat diff;
    cv::absdiff(current_frame, previous_frame, diff);
    
    cvtColor(diff, diff, CV_RGB2GRAY);
    cv::threshold(diff, diff, 35, 255, CV_THRESH_BINARY);
    
    double percentage = cv::countNonZero(diff) * 100 / (double)diff.total();

    return percentage > ACCEPTABLE_MOTION_PERCENTAGE;
}

cv::Mat findEdges(cv::Mat img) {
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);
    
    cv::Canny(img_gray, img_gray, 150, 200);

    return img_gray;
}

int checkVerticalLineLength(cv::Mat img, int col, int row) {
    int length = 0;
    
    int next_pixel  = (int)img.at<uchar>(row, col);
    int left_pixel, right_pixel;
    
    if (next_pixel == 255) {
        while (true) {
            row--;
            next_pixel  = (int)img.at<uchar>(row, col);
            left_pixel  = (int)img.at<uchar>(row, col-1);
            right_pixel = (int)img.at<uchar>(row, col+1);
            
            if (next_pixel != 255) {
                if (left_pixel == 255) {
                    col--;
                }
                else if (right_pixel == 255) {
                    col++;
                }
                else {
                    break;
                }
            }
            
            length++;
        }
    }
    
    return length;
}

void checkPostboxesForFrame(cv::Mat current_frame) {
    static cv::Mat initial_frame = current_frame.clone();
    
    int line_count[NUM_POSTBOXES];
    
    if (!containsMotion(current_frame, initial_frame)) {
        cv::Mat edges = findEdges(current_frame);

        for (int n = 0; n < NUM_POSTBOXES; n++) {
            const int *location = postbox_locations[n];
            int end = location[1];
            int row = location[2];
            
            cv::Mat color_edges;
            cv::cvtColor(edges, color_edges, CV_GRAY2BGR);

            for (int pixel = location[0]; pixel < end; pixel++) {
                int line_length = checkVerticalLineLength(edges, pixel, row);

                if (line_length > MIN_LINE_LENGTH) {
                    line_count[n]++;
                }
                
                color_edges.at<cv::Vec3b>(row, pixel) = cv::Vec3b(255, 0, 0);
            }

            if (line_count[n] < MIN_VISIBLE_LINES) {
                std::cout << "Post in " << n << std::endl;
            }
            else {
                std::cout << "No Post in " << n << std::endl;
            }
            
            cv::imshow("Edges", color_edges);
        }
    }
    else {
        initial_frame = current_frame.clone();
    }
}

int main(int argc, char **argv) {
    cv::VideoCapture cap("/Users/mattdonnelly/Documents/College/Computer Vision/Lab 2/Media/Postboxes.avi");
    if (!cap.isOpened()) {
        return -1;
    }
    
    const double fps = cap.get(CV_CAP_PROP_FPS);
    
    cv::Mat current_frame;
    while (true) {
        if (!cap.read(current_frame)) {
            break;
        }
        
        checkPostboxesForFrame(current_frame);
        
        if (cv::waitKey(1000.0 / fps) == ESC_KEY) {
            break;
        }
    }
    
    cap.release();
}
