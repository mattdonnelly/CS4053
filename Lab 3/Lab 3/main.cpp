//
//  main.cpp
//  Lab 3
//
//  Created by Matt Donnelly on 02/12/2014.
//  Copyright (c) 2014 Matt Donnelly. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#define MIN_LAB_RANGE 30.0f
#define MAX_LAB_RANGE 180.0f

#define MIN_A_RANGE 137.0f
#define MAX_A_RANGE 255.0f

#define MIN_B_RANGE 100.0f
#define MAX_B_RANGE 255.0f

#define BLUR_AMOUNT 5.0f

#define CLOSE_AMOUNT 5.0f
#define OPEN_AMOUNT 7.0f

#define MIN_CONTOUR_AREA 480.0f

typedef std::vector<cv::Point> Contour;

cv::Mat find_red_areas(cv::Mat img) {
    cv::Mat lab;
    cv::cvtColor(img, lab, CV_BGR2Lab);

    cv::GaussianBlur(lab, lab, cv::Size(BLUR_AMOUNT, BLUR_AMOUNT), 20.0);
    
    cv::Mat result;
    cv::inRange(lab,
                cv::Scalar(MIN_LAB_RANGE, MIN_A_RANGE, MIN_B_RANGE),
                cv::Scalar(MAX_LAB_RANGE, MAX_A_RANGE, MAX_B_RANGE),
                result);

    cv::morphologyEx(result, result, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CLOSE_AMOUNT, CLOSE_AMOUNT)));

    cv::morphologyEx(result, result, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CLOSE_AMOUNT, OPEN_AMOUNT)));

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
    //img.copyTo(imageROI, mask);
    cv::Mat contour_region = img(bounding_rect);
    
    /*cv::RotatedRect rotated_rect = cv::minAreaRect(cv::Mat(contour));

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
    cv::warpAffine(contour_region, warped, warp_mat, warped.size());*/

    return contour_region;
}

void chamfer_matching(cv::Mat &chamfer_image, cv::Mat &model, cv::Mat &matching_image) {
    // Extract the model points (as they are sparse).
    std::vector<cv::Point> model_points;
    int image_channels = model.channels();
    for (int model_row = 0; model_row < model.rows; model_row++) {
        uchar *curr_point = model.ptr<uchar>(model_row);
        
        for (int model_column = 0; model_column < model.cols; model_column++) {
            if (*curr_point > 0) {
                cv::Point new_point = cv::Point(model_column,model_row);
                model_points.push_back(new_point);
            }

            curr_point += image_channels;
        }
    }
    
    size_t num_model_points = model_points.size();
    image_channels = chamfer_image.channels();
    
    // Try the model in every possible position
    matching_image = cv::Mat(chamfer_image.rows-model.rows+1, chamfer_image.cols-model.cols+1, CV_32FC1);
    
    for (int search_row=0; search_row <= chamfer_image.rows-model.rows; search_row++) {
        float *output_point = matching_image.ptr<float>(search_row);
        
        for (int search_column = 0; search_column <= chamfer_image.cols-model.cols; search_column++) {
            float matching_score = 0.0;
            
            for (int point_count = 0; point_count < num_model_points; point_count++) {
                matching_score += (float) *(chamfer_image.ptr<float>(model_points[point_count].y+search_row) + search_column + model_points[point_count].x*image_channels);
            }
            
            *output_point = matching_score;
            output_point++;
        }
    }
}

void show_32bit_image(const char *window_name, cv::Mat &passed_image, double zero_maps_to = 0.0, double passed_scale_factor =-1.0 )
{
    cv::Mat display_image;
    double scale_factor = passed_scale_factor;
    if (passed_scale_factor == -1.0)
    {
        double minimum,maximum;
        cv::minMaxLoc(passed_image,&minimum,&maximum);
        scale_factor = (255.0-zero_maps_to)/cv::max(-minimum,maximum);
    }
    passed_image.convertTo(display_image, CV_8U, scale_factor, zero_maps_to);
    imshow( window_name, display_image );
}

cv::Mat aspect_fit_in_image(cv::Mat img, cv::Size size) {
    double ratios[2] = {
        (double)size.width / (double)img.cols,
        (double)size.height / (double)img.rows
    };
    
    cv::Size resize(img.cols, img.rows);
    if (ratios[0] < ratios[1]) {
        resize.width = (int)(resize.width * ratios[0] + 1);
        resize.height = (int)(resize.height * ratios[0] + 1);
    }
    else {
        resize.width = (int)(resize.width * ratios[1] + 1);
        resize.height = (int)(resize.height * ratios[1] + 1);
    }
    
    cv::Mat result;
    cv::resize(img, result, resize);
    
    return result;
}

void matching_sign(cv::Mat img, std::vector<cv::Mat> samples) {
    cv::Mat best_match = cv::Mat::zeros(cvSize(1024, 1024), CV_8UC3);
    
    cv::imshow("Original", img);
    
    cv::Mat img_edges, chamfer_image;
    cv::cvtColor(img, img_edges, CV_BGR2GRAY);
    cv::Canny(img_edges, img_edges, 200, 300, 3);
    cv::threshold(img_edges, img_edges, 127, 255, cv::THRESH_BINARY);
    cv::distanceTransform(img_edges, chamfer_image, CV_DIST_L2 , 3);

    show_32bit_image("Chamfer image", chamfer_image);
    cv::waitKey(-1);
    
    for (int i = 0; i < samples.size(); i++) {
        cv::Mat sample = samples[i].clone();
        cv::imshow("Matching against", sample);
        
        cv::cvtColor(sample, sample, CV_BGR2GRAY);
        cv::Canny(sample, sample, 200, 400, 3);
        cv::threshold(sample, sample, 127, 255, cv::THRESH_BINARY);

        if (sample.size().width > chamfer_image.size().width ||
               sample.size().height > chamfer_image.size().height) {
            sample = aspect_fit_in_image(sample, chamfer_image.size());
        }
        
        cv::Mat result;
        chamfer_matching(chamfer_image, sample, result);

        std::cout << result.size() << std::endl;
        
        if (result.size().area()) {
            show_32bit_image("Matching image", result);
            cv::waitKey(-1);
        }
    }
}

void draw_shape_rects(cv::Mat img, std::vector<Contour> contours) {
    static cv::RNG rng(12345);
    
    for (int i = 0; i < contours.size(); i++) {
        cv::RotatedRect rect = cv::minAreaRect(cv::Mat(contours[i]));
        
        cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        cv::Point2f rect_points[4];
        rect.points( rect_points );
        
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
        
        for (int j = 0; j < contours.size(); j++) {
            cv::Mat sign = extract_shape(img, contours[j]);
            matching_sign(sign, known_images);
        }
    }
    
    return 0;
}
