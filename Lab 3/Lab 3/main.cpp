//
//  main.cpp
//  Lab 3
//
//  Created by Matt Donnelly on 02/12/2014.
//  Copyright (c) 2014 Matt Donnelly. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#define MIN_HUE_RANGE 0.0f
#define MAX_HUE_RANGE 55.0f

#define MIN_SAT_RANGE 150.0f
#define MAX_SAT_RANGE 255.0f

#define MAX_BLUE_VALUE 90.0f
#define MAX_GREEN_VALUE 68.0f
#define MIN_RED_VALUE 57.0f

#define CLOSE_AMOUNT 5.0f
#define OPEN_AMOUNT 7.0f

#define MIN_CONTOUR_AREA 480.0f

typedef std::vector<cv::Point> Contour;

cv::Mat find_red_areas(cv::Mat img) {
    cv::Mat lab;
    cv::cvtColor(img, lab, CV_BGR2Lab);
    
    cv::Mat result;
    cv::inRange(lab, cv::Scalar(0, 137, 100), cv::Scalar(180, 255, 255), result);
    
    cv::erode(result, result, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CLOSE_AMOUNT, CLOSE_AMOUNT)));
    cv::dilate(result, result, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CLOSE_AMOUNT, CLOSE_AMOUNT)));
    
    cv::dilate(result, result, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(OPEN_AMOUNT, OPEN_AMOUNT)));
    cv::erode(result, result, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(OPEN_AMOUNT, OPEN_AMOUNT)));

    return result;
}

bool contours_intersect(Contour shape1, Contour shape2) {
    cv::Rect bounding_rect = cv::boundingRect(shape1);

    for (int i = 0; i < shape2.size(); i++) {
        if (!bounding_rect.contains(shape2[i])) {
            return false;
        }
    }
    
    return true;
}

std::vector<Contour> filter_intersecting_shapes(std::vector<Contour> shapes) {
    std::vector<Contour> filtered_shapes;
    for (int i = 0; i < shapes.size(); i++) {
        bool inside_shape = false;
        
        for (int j = 0; j < shapes.size(); j++) {
            if (i != j && contours_intersect(shapes[j], shapes[i])) {
                inside_shape = true;
                break;
            }
        }
        
        if (!inside_shape) {
            filtered_shapes.push_back(shapes[i]);
        }
    }
    
    return filtered_shapes;
}

std::vector<Contour> find_blob_contours(cv::Mat img){
    std::vector<Contour> contours;
    std::vector<Contour> filtered_contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::findContours(img.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    
    for (int i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> contour = contours[i];
        
        double area = cv::contourArea(contour, false);
        if (area > MIN_CONTOUR_AREA) {
            filtered_contours.push_back(contours[i]);
        }
    }

    return filter_intersecting_shapes(filtered_contours);
}

cv::Mat extract_shape(cv::Mat img, Contour contour) {
    cv::Rect bounding_rect = boundingRect(contour);
    
    std::vector<Contour> contour_vec;
    contour_vec.push_back(contour);
    
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::drawContours(mask, contour_vec, 0, cv::Scalar(255), CV_FILLED);

    cv::Mat imageROI;
    img.copyTo(imageROI, mask);
    cv::Mat contour_region = imageROI(bounding_rect);
    
    cv::RotatedRect rotated_rect = cv::minAreaRect(cv::Mat(contour));

    cv::Point2f src_points[4];
    rotated_rect.points(src_points);
    src_points[0] = cv::Point2f(src_points[0].x - bounding_rect.x, src_points[0].y - bounding_rect.y);
    src_points[1] = cv::Point2f(src_points[1].x - bounding_rect.x, src_points[1].y - bounding_rect.y);
    src_points[2] = cv::Point2f(src_points[2].x - bounding_rect.x, src_points[2].y - bounding_rect.y);
    src_points[3] = cv::Point2f(src_points[3].x - bounding_rect.x, src_points[3].y - bounding_rect.y);
    
    cv::Point2f dst_points[4] = {
        cv::Point2f(bounding_rect.width, bounding_rect.height),
        cv::Point2f(0.0, bounding_rect.height),
        cv::Point2f(0.0),
        cv::Point2f(bounding_rect.width, 0.0)
    };
    
    cv::Mat warp_mat = getAffineTransform(src_points, dst_points);
    
    cv::Mat warped = cv::Mat::zeros(contour_region.rows, contour_region.cols, contour_region.type());
    cv::warpAffine(contour_region, warped, warp_mat, warped.size());

    return warped;
}

cv::Mat matching_sign(cv::Mat img, std::vector<cv::Mat> samples) {
    cv::Mat best_match = cv::Mat::zeros(cvSize(1024, 1024), CV_8UC3);
    
    for (int i = 0; i < samples.size(); i++) {
        /*cv::Mat result;
        matchTemplate(img, samples[i], result, CV_TM_SQDIFF);
        normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        
        cv::imshow("hi", result);
        cv::waitKey(-1);*/
    }
    
    return best_match;
}

void draw_shape_rects(cv::Mat img, std::vector<Contour> contours) {
    for (int i = 0; i < contours.size(); i++) {
        cv::RotatedRect rect = cv::minAreaRect(cv::Mat(contours[i]));
        
        static cv::RNG rng(12345);
        cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        cv::Point2f rect_points[4]; rect.points( rect_points );
        for (int j = 0; j < 4; j++ ) {
            line(img, rect_points[j], rect_points[(j+1)%4], color, 2, 8);
        }
    }
}

int main(int argc, const char * argv[]) {
    const int num_known_files = 9;
    std::string known_file_names[] = {
        "RoadSignGoLeft.JPG",
        "RoadSignGoRight.JPG",
        "RoadSignGoStraight.JPG",
        "RoadSignNoLeftTurn.JPG",
        "RoadSignNoParking.JPG",
        "RoadSignNoRightTurn.JPG",
        "RoadSignNoStraight.JPG",
        "RoadSignParking.JPG",
        "RoadSignYield.JPG"
    };
    
    std::vector<cv::Mat> known_images;

    for (int i = 0; i < num_known_files; i++) {
        std::string name = known_file_names[i];
        cv::Mat img = cv::imread("/Users/mattdonnelly/Documents/College/Computer Vision/Lab 3/RoadSignRecognitionKnownSigns/" + name);
        known_images.push_back(img);
    }
    
    const int num_unknown_files = 5;
    std::string unknown_file_names[] = {
        "RoadSigns1.jpg",
        "RoadSigns2.jpg",
        "RoadSigns3.jpg",
        "RoadSignsComposite1.JPG",
        "RoadSignsComposite2.JPG"
    };
    
    for (int i = 0; i < num_unknown_files; i++) {
        std::string name = unknown_file_names[i];
        std::cout << name << std::endl;
        
        cv::Mat img = cv::imread("/Users/mattdonnelly/Documents/College/Computer Vision/Lab 3/RoadSignRecognitionUnknownSigns/" + name);
        
        cv::Mat red_area = find_red_areas(img);
        std::vector<Contour> contours = find_blob_contours(red_area);
        
        draw_shape_rects(img, contours);
        cv::imshow("Hi", img);
        cv::waitKey(-1);
    }
    
    return 0;
}
