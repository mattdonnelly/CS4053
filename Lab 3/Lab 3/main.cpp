//
//  main.cpp
//  Lab 3
//
//  Created by Matt Donnelly on 02/12/2014.
//  Copyright (c) 2014 Matt Donnelly. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#define MAX_BLUE_VALUE 45.0f
#define MAX_GREEN_VALUE 45.0f
#define MIN_RED_VALUE 100.0f

#define MIN_SHAPE_AREA 2000.0f

typedef std::vector<cv::Point> Shape;

cv::Mat find_red_areas(cv::Mat img) {
    cv::Mat red, green, blue;
   
    cv::extractChannel(img, blue, 0);
    cv::extractChannel(img, green, 1);
    cv::extractChannel(img, red, 2);
    
    cv::inRange(blue, 0, MAX_BLUE_VALUE, blue);
    cv::inRange(green, 0, MAX_BLUE_VALUE, green);
    cv::threshold(red, red, MIN_RED_VALUE, 255, CV_THRESH_BINARY);

    cv::Mat result;
    cv::bitwise_and(blue, green, result);
    cv::bitwise_and(result, red, result);

    //cv::erode(result, result, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    //cv::dilate(result, result, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    
    cv::dilate(result, result, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::erode(result, result, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    
    return result;
}

bool shape_contains_shape(Shape shape1, Shape shape2) {
    cv::Rect bounding_rect = cv::boundingRect(shape1);

    for (int i = 0; i < shape2.size(); i++) {
        if (!bounding_rect.contains(shape2[i])) {
            return false;
        }
    }
    
    return true;
}

bool shapes_contain_shape(std::vector<Shape> shapes, Shape shape) {
    for (int i = 0; i < shapes.size(); i++) {
        if (shape_contains_shape(shapes[i], shape)) {
            return true;
        }
    }

    return false;
}

std::vector<Shape> find_blob_contours(cv::Mat img){
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> filtered_contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::findContours(img.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    
    for (int i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> contour = contours[i];
        
        double area = cv::contourArea(contour, false);
        if (area > MIN_SHAPE_AREA && !shapes_contain_shape(filtered_contours, contour)) {
            filtered_contours.push_back(contours[i]);
        }
    }

    return filtered_contours;
}

std::vector<cv::RotatedRect> convert_contours_to_rects(std::vector<Shape> contours) {
    std::vector<cv::RotatedRect> rects;
    
    for (int i = 0; i < contours.size(); i++) {
        rects.push_back(cv::minAreaRect(cv::Mat(contours[i])));
    }
    
    return rects;
}

cv::Mat extract_shape(cv::Mat img, Shape contour) {
    cv::Rect bounding_rect = boundingRect(contour);
    
    std::vector<Shape> contour_vec;
    contour_vec.push_back(contour);
    
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::drawContours(mask, contour_vec, 0, cv::Scalar(255), CV_FILLED);

    cv::Mat imageROI;
    img.copyTo(imageROI, mask);
    cv::Mat contour_region = imageROI(bounding_rect);

    return contour_region;
}

void draw_rotated_rects(cv::Mat img, std::vector<cv::RotatedRect> rects) {
    for (int i = 0; i < rects.size(); i++) {
        cv::Scalar color = cv::Scalar(255, 0, 255);
        cv::Point2f rect_points[4];
        rects[i].points( rect_points );
        for (int j = 0; j < 4; j++ ) {
            line(img, rect_points[j], rect_points[(j+1)%4], color, 2, 8);
        }
    }
}

int main(int argc, const char * argv[]) {
    cv::Mat orig = cv::imread("/Users/mattdonnelly/Documents/College/Computer Vision/Lab 3/RoadSignRecognitionUnknownSigns/RoadSigns1.jpg");

    cv::Mat red_area = find_red_areas(orig.clone());
    std::vector<Shape> contours = find_blob_contours(red_area);
    std::vector<cv::RotatedRect> rects = convert_contours_to_rects(contours);
    
    for (int i = 0; i < contours.size(); i++) {
        // Get bounding box for contour
        cv::Mat test = extract_shape(orig, contours[i]);
        cv::imshow("Test", test);
        cv::waitKey(-1);
    }
    
    return 0;
}
