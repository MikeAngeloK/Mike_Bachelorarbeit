#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

#include <unordered_set>
#include <iostream>
#include <fstream>
#include <Utils.h>
#include <ccomp.hpp>
#include <BundleAdjustment.h>

using namespace std;
using namespace cv;

class imageprocessing
{
public:
    imageprocessing();
    void MatchImageFeatures(const int skip_thresh);
    bool ReadStringList(string& filename);
    bool GetCameraMatrix(const std::string & fileName);
    void ExtractFeatures(Mat img, Features& features);
    void ComputeMatches(Features& features1, Features& features2, Matches& matches);
    void FilterMatch(Features& features1, Features& features2, Matches& matches, int type);
    void InitReconstruction();
    int FindMaxSizeMatch(const bool within_todo_views = false);
    void TriangulatePointsFromViews(const int first_id, const int second_id, Map3D& map);
    void TriangulatePoints(Matx34d& P1, vector<Point2d>& points1, Matx34d& P2, vector<Point2d>& points2, Mat& points3d);
    void CombineMapComponents(Map3D& map, const double max_keep_dist);
    void ReconstructAll();
    int GetNextBestViewByViews();
    void ReconstructNextViewPair(const int first_id, const int second_id);
    void MergeAndCombinePoints(Map3D& map, const Map3D& local_map, const double max_keep_dist);
    void ReconstructNextView(const int next_img_id);
    void GetCameraPose(const int first_id, const int second_id, Matx34d& P1, Matx34d& P2);
    bool FindCameraPosePNP(vector<Point3d>& points3D, vector<Point2d>& points2D, Matx34d& P);
    void Find2D3DMatches(const int next_img_id, const int second_id, vector<Point3d>& points3D, vector<Point2d>& points2D);

    void displayAllImages();
    void keypointsMatchesToPoint2d(vector<DMatch> matches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
                            vector<Point2d> &points1, vector<Point2d> &points2);
    void KeypointsToPoint2d(vector<KeyPoint> keypoints, vector<Point2d>& points2d);
    void WorldPointsToVec(vector<Point3d>& point3d);
    bool CheckCoherentRotation(Mat& R);
    void GenerateAllPairs();
    void OptimizeBundle();
    void ExportPoints(string& filename);
    void Tracking();

private:
    vector<string> imageList;
    Intrinsics cameraMatrix;

    CComponents<IntPair> ccomp_;
    vector<ImagePair> image_pairs_;
    vector<Features> image_features_;
    vector<Matches> image_matches_;
    map<IntPair, int> matches_index_;

    unordered_set<int> used_views_;
    unordered_set<int> todo_views_;
    Map3D map_;

    vector<Matx34d> camera_poses_;

    double max_merge_dist = 1.0;
    double repr_error_thresh = 1.0;
};

#endif // IMAGEPROCESSING_H
