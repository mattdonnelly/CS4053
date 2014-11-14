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

bool containsMotion(cv::Mat current_frame, cv::Mat previous_frame) {
    cv::Mat diff;
    cv::absdiff(current_frame, previous_frame, diff);
    
    cvtColor(diff, diff, CV_RGB2GRAY);
    cv::threshold(diff, diff, 35, 255, CV_THRESH_BINARY);
    
    double percentage = (cv::countNonZero(diff) / (double)diff.total()) * 100;

    return percentage > 2.0;
}

cv::Mat findEdges(cv::Mat frame) {
    cv::Mat frame_gray, canny_output, detected_edges;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::GaussianBlur(frame, frame, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    
    cv::cvtColor(frame, frame_gray, CV_RGB2GRAY);
    
    cv::Canny(frame_gray, canny_output, 150, 200);
    cv::findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    cv::Mat edges = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
    for (int i = 0; i < contours.size(); i++) {
        drawContours(edges, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy, 0, cv::Point());
    }

    return edges;
}

void checkPostboxesForFrame(cv::Mat current_frame) {
    static cv::Mat initial_frame = current_frame.clone();
    
    if (!containsMotion(current_frame, initial_frame)) {
        cv::Mat edges = findEdges(current_frame);
        
        cv::imshow("Edges", edges);
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
