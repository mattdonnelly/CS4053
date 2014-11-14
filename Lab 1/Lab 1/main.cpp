//
//  main.cpp
//  Lab 1
//
//  Created by Matt Donnelly on 18/10/2014.
//  Copyright (c) 2014 Matt Donnelly. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    std::string filepath = "/Users/mattdonnelly/Documents/College/Computer Vision/Lab 1/Media/";

    const int num_files = 18;

    std::string filenames[] = {
        "BabyFood-Test1.JPG",  "BabyFood-Test2.JPG",  "BabyFood-Test3.JPG",
        "BabyFood-Test4.JPG",  "BabyFood-Test5.JPG",  "BabyFood-Test6.JPG",
        "BabyFood-Test7.JPG",  "BabyFood-Test8.JPG",  "BabyFood-Test9.JPG",
        "BabyFood-Test10.JPG", "BabyFood-Test11.JPG", "BabyFood-Test12.JPG",
        "BabyFood-Test13.JPG", "BabyFood-Test14.JPG", "BabyFood-Test15.JPG",
        "BabyFood-Test16.JPG", "BabyFood-Test17.JPG", "BabyFood-Test18.JPG"
    };

    cv::Mat sample0 = cv::imread(filepath + "BabyFood-Sample0.JPG");
    cv::Mat sample1 = cv::imread(filepath + "BabyFood-Sample1.JPG");
    cv::Mat sample2 = cv::imread(filepath + "BabyFood-Sample2.JPG");

    cv::cvtColor(sample0, sample0, CV_BGR2HSV);
    cv::cvtColor(sample1, sample1, CV_BGR2HSV);
    cv::cvtColor(sample2, sample2, CV_BGR2HSV);

    cv::extractChannel(sample0, sample0, 1);
    cv::extractChannel(sample1, sample1, 1);
    cv::extractChannel(sample2, sample2, 1);

    double sample0_mean = cv::mean(sample0)[0];
    double sample1_mean = cv::mean(sample1)[0];
    double sample2_mean = cv::mean(sample2)[0];

    for (int i = 0; i < num_files; i++) {
        cv::Mat image = cv::imread(filepath + filenames[i]);

        cv::Mat hsv_image;
        cv::cvtColor(image, hsv_image, CV_BGR2HSV);
        cv::extractChannel(hsv_image, hsv_image, 1);

        double image_mean = cv::mean(hsv_image)[0];

        double diff0 = abs(sample0_mean - image_mean);
        double diff1 = abs(sample1_mean - image_mean);
        double diff2 = abs(sample2_mean - image_mean);

        std::string num_spoons;

        if (diff0 <= diff1 && diff0 <= diff2) {
            num_spoons = "0 Spoons";
        }
        else if (diff1 <= diff0 && diff1 <= diff2) {
            num_spoons = "1 Spoon";
        }
        else {
            num_spoons = ">1 Spoon";
        }

        cv::putText(image, num_spoons, cv::Point(10, 45), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2, CV_AA);
        cv::imshow("Lab 1", image);
        cv::waitKey();
    }

    return 0;
}
