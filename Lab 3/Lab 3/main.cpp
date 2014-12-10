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
    
    cv::bitwise_not(blue, blue);
    cv::threshold(blue, blue, 255 - MAX_BLUE_VALUE, 255, CV_THRESH_BINARY);

    cv::bitwise_not(green, green);
    cv::threshold(green, green, 255 - MAX_GREEN_VALUE, 255, CV_THRESH_BINARY);

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

cv::Mat extract_flat_object(cv::Mat img, cv::RotatedRect rect) {
    float angle = rect.angle;
    float width = rect.size.width;
    float height = rect.size.height;
    if (rect.angle < -45) {
        angle += 90.0;
        float temp = width;
        width = height;
        height = temp;
    }

    cv::Mat result;
    cv::Mat rotation_mat = getRotationMatrix2D(rect.center, angle, 1.0);
    warpAffine(img, result, rotation_mat, img.size(), cv::INTER_CUBIC);
    result = result(rect.boundingRect());
    
    return result;
}

void draw_shape_rects(cv::Mat img, std::vector<cv::RotatedRect> rects) {
    for (int i = 0; i < rects.size(); i++) {
        static cv::RNG rng(12345);
        cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        cv::Point2f rect_points[4]; rects[i].points( rect_points );
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
    
    draw_shape_rects(orig, rects);
    
    cv::imshow("Test", orig);
    cv::waitKey(-1);
    
    return 0;
}
