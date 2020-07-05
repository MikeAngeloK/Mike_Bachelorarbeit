#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/sfm.hpp>
#include <ceres/ceres.h>

struct Intrinsics{
    cv::Mat_<double> K;
    cv::Mat_<double> distCoef;
};

struct Features {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat image;
    cv::Mat grayImage;
    cv::Matx34d camera_pose;
};

struct ImagePair {
    int first;
    int second;
};

struct Matches {
    ImagePair image_index;
    std::vector<cv::DMatch> match;
};

struct WorldPoint3D {
    cv::Point3d pt;
    std::map<int, int> views;
    int component_id;
};

typedef std::pair<int, int> IntPair;
typedef std::vector<WorldPoint3D> Map3D;
#endif // UTILS_H
