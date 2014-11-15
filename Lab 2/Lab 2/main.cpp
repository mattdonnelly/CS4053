//
//  main.cpp
//  Lab 2
//
//  Created by Matt Donnelly on 13/11/2014.
//  Copyright (c) 2014 Matt Donnelly. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#define ESC_KEY 27
#define ACCEPTABLE_MOTION_PERCENTAGE 10.0
#define NUM_POSTBOXES 6
#define MAX_LINES_FULL 6

// Array containing the starting column, end column and
// row points to scan for vertical lines
const int postbox_line_locations[NUM_POSTBOXES][3] {
    // Start Col, End Col, Row
    {22,  98,  130},
    {127, 199, 130},
    {30,  100, 248},
    {130, 194, 248},
    {30,  100, 360},
    {126, 194, 360}
};

// Array of points to draw the status of each postbox at
const cv::Point status_locations[NUM_POSTBOXES] {
    cv::Point(50,  90),
    cv::Point(150, 90),
    cv::Point(50,  215),
    cv::Point(150, 215),
    cv::Point(50,  325),
    cv::Point(150, 325)
};

// Determines if there's motion in a frame by looking at the previous
// frame and calculating the percentage of change in the image
bool containsMotion(cv::Mat current_frame, cv::Mat previous_frame) {
    cv::Mat diff;
    cv::absdiff(current_frame, previous_frame, diff);
    
    cvtColor(diff, diff, CV_RGB2GRAY);
    cv::threshold(diff, diff, 35, 255, CV_THRESH_BINARY);
    
    double percentage = cv::countNonZero(diff) * 100 / (double)diff.total();

    return percentage >= ACCEPTABLE_MOTION_PERCENTAGE;
}

// Finds the edges of an image by performing Canny edge detection. Returns
// a grayscale image of the detected edges
cv::Mat findEdges(cv::Mat img) {
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);
    
    cv::Canny(img_gray, img_gray, 300, 400);

    return img_gray;
}

// Counts the number of lines in a row between two columns where a line
// is defined as a white pixel where the two pixels next to it are black
int countLinesInRow(cv::Mat img, int start_col, int end_col, int row) {
    int line_count = 0;

    for (int i = start_col; i < end_col; i++) {
        int curr_pixel  = (int)img.at<uchar>(row, i);
        int left_pixel  = (int)img.at<uchar>(row, i-1);
        int right_pixel = (int)img.at<uchar>(row, i+1);
        
        if (curr_pixel == 255 && left_pixel == 0 && right_pixel == 0) {
            line_count++;
        }
    }
    
    return line_count;
}

// Labels a specific postbox based on the number of lines visible
void labelPostbox(cv::Mat img, int num, int line_count) {
    if (line_count > MAX_LINES_FULL) {
        cv::putText(img, "x", status_locations[num], cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 2, CV_AA);
    }
    else {
        cv::putText(img, "o", status_locations[num], cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2, CV_AA);
    }
}

// Checks and labels the status of each postbox in a frame
void checkPostboxesForFrame(cv::Mat current_frame) {
    static cv::Mat last_still_frame = current_frame.clone();
    
    static int line_count[NUM_POSTBOXES];
    
    if (!containsMotion(current_frame, last_still_frame)) {
        cv::Mat edges = findEdges(current_frame);

        for (int n = 0; n < NUM_POSTBOXES; n++) {
            const int *location = postbox_line_locations[n];

            line_count[n] = countLinesInRow(edges, location[0], location[1], location[2]);
            
            labelPostbox(current_frame, n, line_count[n]);
        }
    }
    else {
        cv::putText(current_frame, "Motion", cv::Point(50, 210), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 0, 255), 2, CV_AA);
        last_still_frame = current_frame.clone();
    }
}

int main(int argc, char **argv) {
    cv::VideoCapture cap("/Users/mattdonnelly/Documents/College/Computer Vision/Lab 2/Media/Postboxes.avi");
    
    if (!cap.isOpened()) {
        return -1;
    }
    
    const double fps = cap.get(CV_CAP_PROP_FPS);
    
    cv::Mat current_frame;

    while (cap.read(current_frame)) {
        checkPostboxesForFrame(current_frame);
        
        cv::imshow("Postboxes", current_frame);

        if (cv::waitKey(1000.0 / fps) == ESC_KEY) {
            break;
        }
    }
    
    cap.release();
}
