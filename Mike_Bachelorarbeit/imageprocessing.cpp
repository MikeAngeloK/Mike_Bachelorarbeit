
#include "imageprocessing.h"

#include <opencv2/sfm.hpp>


imageprocessing::imageprocessing()
{

}

void imageprocessing::MatchImageFeatures(const int skip_thresh)
{
    cout << "[START] image processing..." << endl;
    for(int i=0; i<imageList.size();i++)
    {
        //load image from path
        Mat img = imread(imageList[i], IMREAD_COLOR);
        Features feature;

        ExtractFeatures(img, feature);
        image_features_.push_back(feature);
        cout << "Feature: " << i  << " Keypoints = " << feature.keypoints.size() << endl;
    }
    cout << "[END] image processing" << endl;

    //assert(image_features_.size() > 1);
    if (image_pairs_.empty())
    {
        // Generating all image pairs so all images will be matched.
        GenerateAllPairs();
    }

    int total_matched_points = 0;
    int skipped_matches = 0;
    int filtered_matches = 0;

    cout << "[START] feature matching..." << endl;
    //match all images
    for(int idx=0; idx< image_pairs_.size(); idx++)
    {
        ImagePair ip = image_pairs_[idx];
        int img_first = ip.first;
        int img_second = ip.second;

        Features& features1 = image_features_[img_first];
        Features& features2 = image_features_[img_second];

        Matches matches;
        matches.image_index.first = img_first;
        matches.image_index.second = img_second;

        ComputeMatches(features1, features2, matches);

        int msize = matches.match.size();
        cout << "Matches before filtering = " << msize << " " ;
        //filter matches depending on type.
        // type   0 = Homography
        //        1 = Fundamental
        //        2 = Essential
        FilterMatch(features1, features2, matches, 1 /*type*/);

        filtered_matches += msize - matches.match.size();

        // Don't add empty or small matches
        if (matches.match.size() < skip_thresh)
        {
            cout << "Match:"
                 << " (" << img_first << "," << img_second << ")"
                 << " " << idx << " out of " << image_pairs_.size()
                 << ": matches.size = " << matches.match.size();
            cout << ", skipped ...\n";
            ++skipped_matches;
            continue;
        }
        image_matches_.push_back(matches);

        // put index into matches_index_
        auto p1 = make_pair(img_first, img_second);
        auto p2 = make_pair(img_second, img_first);
        matches_index_.insert(
                    make_pair(p1, image_matches_.size() - 1));
        matches_index_.insert(
                    make_pair(p2, image_matches_.size() - 1));

        // Connect keypoints and images for a quick retrieval later
        for (int i = 0; i < matches.match.size(); ++i) {
            IntPair p1 = make_pair(
                        matches.image_index.first,
                        matches.match[i].queryIdx);
            IntPair p2 = make_pair(
                        matches.image_index.second,
                        matches.match[i].trainIdx);
            ccomp_.Union(p1, p2);
        }

        total_matched_points += matches.match.size();
        cout << "Match:"
             << " (" << img_first << "," << img_second << ")"
             << " " << idx << " out of " << image_pairs_.size()
             << ": matches.size = " << matches.match.size()
             << ", total_matched_points = " << total_matched_points
             << ", id: " << image_matches_.size() - 1
             << endl;
    }
    cout << "[END] feature matching..." <<endl;
    cout << "total_matched_points = " << total_matched_points << endl;
    cout << "skipped_matches = " << skipped_matches << endl;
    cout << "filtered_matches = " << filtered_matches << endl;
    cout << "image_matches_.size = " << image_matches_.size() << endl;
}

bool imageprocessing::ReadStringList(string& filename)
{
    //reading relative path to images from imagelist.xml
    cout<<"Reading image file..."<<endl;
    imageList.clear();
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
    {
        cout<<"Fehler beim Lesen der Input File!";
        return false;
    }
    FileNode n = fs.getFirstTopLevelNode();
    if(n.type()!= FileNode::SEQ)
    {
        cout<<"Fehler beim Lesen der Input File!";
        return false;
    }
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it!= it_end; ++it)
        imageList.push_back((string)*it);
    return true;
}

bool imageprocessing::GetCameraMatrix(const string & fileName)
{
    //Camera calibration matrix comes from another programm.
    //Check out camera calibration tutorial in opencv.
    cout << "Getting camera matrix..." << endl;
    Mat intrinsics;
    Mat cameraDistCoeffs;

    ///Read camera calibration file
    FileStorage fs(fileName, cv::FileStorage::READ);

    //Get data from tags: camera_matrix and distortion_coefficients
    fs["camera_matrix"] >> intrinsics;
    fs["distortion_coefficients"] >> cameraDistCoeffs;

    if(intrinsics.empty() or intrinsics.at<double>(2,0) !=0){
        cerr << "Error: no found or invalid camera calibration file.xml" << endl;
        return false;
    }

    double fx = intrinsics.at<double>(0,0);
    double fy = intrinsics.at<double>(1,1);
    double cx = intrinsics.at<double>(0,2);
    double cy = intrinsics.at<double>(1,2);

    Mat_<double> cam_matrix = (Mat_<double>(3, 3) << fx, 0, cx,
                               0, fy, cy,
                               0,  0,  1);

    double k1 = cameraDistCoeffs.at<double>(0,0);
    double k2 = cameraDistCoeffs.at<double>(0,1);
    double k3 = cameraDistCoeffs.at<double>(0,2);
    double p1 = cameraDistCoeffs.at<double>(0,3);
    double p2 = cameraDistCoeffs.at<double>(0,4);

    Mat_<double> distortionC = (Mat_<double>(1, 5) << k1, k2, k3, p1, p2);

    //Fill local variables with input data
    cameraMatrix.K = cam_matrix;                  //Matrix K (3x3)
    cameraMatrix.distCoef = distortionC;     //Distortion coefficients (1x5)

    cout << "Camera matrix:" << "\n" << cameraMatrix.K << endl;
    cout << "Distortion coefficients: "<< endl;
    cout << cameraMatrix.distCoef << endl;

    if(cameraMatrix.K.empty()){
        cout << "Could not load local variables with camera calibration file data" << endl;
        return false;
    }

    return true;
}

void imageprocessing::ExtractFeatures(Mat img, Features& features)
{
    //detect and compute keypoints
    Mat feature_mask;
    features.keypoints.clear();
    features.descriptors = Mat();
    cvtColor(img, features.grayImage, COLOR_BGR2GRAY);

    Ptr<AKAZE> detector = AKAZE::create();
    detector->detectAndCompute(img, feature_mask, features.keypoints, features.descriptors);
    features.image = img;
}

void imageprocessing::ComputeMatches(Features& features1, Features& features2, Matches& matches)
{
    //matching descriptors with Brute Force Matcher
    matches.match.clear();

    Mat& descriptors1 = features1.descriptors;
    Mat& descriptors2 = features2.descriptors;

    cv::BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    //filter matches using nearest-neighbor distance ratio test
    const float ratio_thresh = 0.7f;
    for (int m = 0; m < knnMatches.size(); ++m) {
        if (knnMatches[m].size() < 2)
            continue; // no match for the points
        if (knnMatches[m][0].distance < ratio_thresh * knnMatches[m][1].distance) {
            DMatch match = knnMatches[m][0];
            matches.match.push_back(match);
        }
    }
}

void imageprocessing::FilterMatch(Features& features1, Features& features2, Matches& matches, int type)
{
    // Convert keypoints from matches into point2d
    vector<Point2d> points1, points2;
    keypointsMatchesToPoint2d(matches.match, features1.keypoints, features2.keypoints, points1, points2);

    if(points1.size()==0)
    {
        cout << "Skipped filtering. ";
        return;
    }

    //functions outputs each match in "inliers" as 0 = invalid, 1 = valid.
    vector<DMatch> inlineMatches;
    vector<uchar> inliers(points1.size(), 0);
    Mat filter;
    switch(type)
    {
    case 0:
        filter = findHomography(Mat(points1), Mat(points2), // matching points
                                inliers,                    // outputed inliers matches
                                RANSAC,                     // RANSAC method
                                1.);                        // max distance to reprojection point
        break;
    case 1:
        filter = findFundamentalMat(Mat(points1), Mat(points2),
                                    inliers,
                                    FM_RANSAC,
                                    1,                      // distance to epipolar line
                                    0.98);                  // confidence probability
        break;
    case 2:
        filter = findEssentialMat(Mat(points1), Mat(points2), // matching points
                                  cameraMatrix.K,             // camera matrix
                                  RANSAC,
                                  0.99,
                                  1,
                                  inliers);
        break;
    default:
        cout << "No known filter type. Check input" << endl;
    }

    // extract the surviving (inliers) matches
    auto itIn = inliers.begin();
    auto itM = matches.match.begin();
    // for all matches
    for (; itIn != inliers.end(); ++itIn, ++itM)
    {

        if (*itIn)
        { // it is a valid match

            inlineMatches.push_back(*itM);
        }
    }

    matches.match.clear();
    matches.match = inlineMatches;
}

void imageprocessing::InitReconstruction()
{
    cout << "[START] INIT Reconstruction..." << endl;

    assert(image_matches_.size() > 0);

    // Init with all images
    for (size_t i = 0; i < image_features_.size(); ++i)
        todo_views_.insert(i);


    // Find the match with the most points
    int most_match_id = FindMaxSizeMatch();
    cout << "most_match = "
         << image_matches_[most_match_id].image_index.first
         << " - " << image_matches_[most_match_id].image_index.second
         << ", " << image_matches_[most_match_id].match.size()
         << " (id: " << most_match_id << ")"
         << endl;

    int first_id = image_matches_[most_match_id].image_index.first;
    int second_id = image_matches_[most_match_id].image_index.second;

    map_.clear();

    //calculate initial projections
    Matx34d P1 = cv::Matx34d::eye();
    Matx34d P2 = cv::Matx34d::eye();
    GetCameraPose(first_id, second_id, P1, P2);
    image_features_[first_id].camera_pose = P1;
    image_features_[second_id].camera_pose = P2;

    TriangulatePointsFromViews(first_id, second_id, map_);

    CombineMapComponents(map_, max_merge_dist);

    cout << "init map size = " << map_.size() << endl;

    // Add used images
    used_views_.insert(first_id);
    used_views_.insert(second_id);

    // Remove used from todo
    todo_views_.erase(first_id);
    todo_views_.erase(second_id);

    camera_poses_.push_back(P1);
    camera_poses_.push_back(P2);

    //OptimizeBundle();

    cout << "[END] INIT Reconstruction" << endl;
}


int imageprocessing::FindMaxSizeMatch(const bool within_todo_views)
{
    int max_match = 0;
    int max_match_id = -1;
    for (int i = 0; i < image_matches_.size(); ++i)
    {
        const Matches& m = image_matches_[i];
        if (within_todo_views // if it should be in todo list but cant find any of the imageidx in the matches
                && todo_views_.find(m.image_index.first) == todo_views_.end())
        {
            continue;
        }
        if (within_todo_views
                && todo_views_.find(m.image_index.second) == todo_views_.end())
        {
            continue;
        }
        if (m.match.size() > max_match)
        {
            max_match = m.match.size();
            max_match_id = i;
        }
    }

    return max_match_id;

}

void imageprocessing::TriangulatePointsFromViews(const int first_id, const int second_id, Map3D &map)
{
    cout << "Triangulate Points... " << endl;
    auto find = matches_index_.find({first_id, second_id});
    if (find == matches_index_.end()) {
        std::cerr << "No match to triangulate for views (" << first_id << ", "
                  << second_id << ")\n";
    }
    int match_index = find->second;

    vector<Point2d> points1, points2;
    keypointsMatchesToPoint2d(image_matches_[match_index].match,
                              image_features_[first_id].keypoints,
                              image_features_[second_id].keypoints,
                              points1, points2);

    //get points without distortions
    vector<Point2d> normalizedPoints1, normalizedPoints2;

    undistortPoints(points1, normalizedPoints1, cameraMatrix.K, cameraMatrix.distCoef);
    undistortPoints(points2, normalizedPoints2, cameraMatrix.K, cameraMatrix.distCoef);

    Matx34d P1 = image_features_[first_id].camera_pose;
    Matx34d P2 = image_features_[second_id].camera_pose;

    Mat points3d;
    TriangulatePoints(P1, normalizedPoints1, P2, normalizedPoints2, points3d);

    //project triangulated 3D-Points back to 2D to check if calculated 3D-Points are good
    Mat rvecLeft;
    Rodrigues(P1.get_minor<3,3>(0,0),rvecLeft);
    Mat tvecLeft(P1.get_minor<3,1>(0,3).t());

    vector<Point2d> projectedLeft(normalizedPoints1.size());
    projectPoints(points3d,rvecLeft,tvecLeft,cameraMatrix.K,cameraMatrix.distCoef,projectedLeft);

    Mat rvecRight;
    Rodrigues(P2.get_minor<3,3>(0,0),rvecRight);
    Mat tvecRight(P2.get_minor<3,1>(0,3).t());

    vector<Point2d> projectedRight(normalizedPoints2.size());
    cv::projectPoints(points3d,rvecRight,tvecRight,cameraMatrix.K,cameraMatrix.distCoef,projectedRight);

    const float MIN_REPROJECTION_ERROR = 6.0; //Maximum 10-pixel allowed re-projection error

    for (int i = 0; i < points3d.rows; ++i)
    {
        const float queryError = cv::norm(projectedLeft[i]  - points1[i]);
        const float trainError = cv::norm(projectedRight[i] - points2[i]);

        if(MIN_REPROJECTION_ERROR < queryError or
                MIN_REPROJECTION_ERROR < trainError)
        {
            cout << i << " [SKIP] : errs1, errs2 = " << queryError
                 << ", " << trainError
                 << endl;
            continue;
        }

        WorldPoint3D wp;
        wp.pt = Point3d(
                    points3d.at<double>(i, 0),
                    points3d.at<double>(i, 1),
                    points3d.at<double>(i, 2));
        wp.views[image_matches_[match_index].image_index.first]
                = image_matches_[match_index].match[i].queryIdx;
        wp.views[image_matches_[match_index].image_index.second]
                = image_matches_[match_index].match[i].trainIdx;
        std::pair<int, int> vk = std::make_pair(
                    image_matches_[match_index].image_index.first,
                    image_matches_[match_index].match[i].queryIdx);
        wp.component_id = ccomp_.Find(vk);
        map.push_back(wp);
    }
    cout << "map_ = " << map_.size();
}

void imageprocessing::TriangulatePoints(Matx34d& P1, vector<Point2d>& points1, Matx34d& P2, vector<Point2d>& points2, Mat& points3d)
{
    Mat points4dh;
    triangulatePoints(P1, P2, points1, points2, points4dh);
    convertPointsFromHomogeneous(points4dh.t(), points3d);
}

void imageprocessing::CombineMapComponents(Map3D &map, const double max_keep_dist)
{
    auto world_point_comp = [](const WorldPoint3D& wp1,
            const WorldPoint3D& wp2)
    {
        return wp1.component_id != wp2.component_id
                ? wp1.component_id < wp2.component_id
                : cv::norm(wp1.pt) < cv::norm(wp2.pt);
    };

    std::sort(map.begin(), map.end(), world_point_comp);

    auto first = map.begin();
    auto second = std::next(first);
    auto w = map.begin();
    bool discard_first = false;
    while (second != map.end()) {
        // auto second = std::next(first);
        // if (second == map.end()) break;
        discard_first = false;
        if (first->component_id == second->component_id)
        {
            double dist = cv::norm(first->pt - second->pt);
            // cout << "comp: " << first->component_id
            //           << ", dist = " << dist << std::endl;
            if (dist < max_keep_dist)
            {
                // combine second to first
                // cout << "   merge\n";
                first->pt = (first->pt + second->pt) * 0.5;
                // Merge second.views to the first
                for (auto view : second->views) {
                    first->views.insert(view);
                }
                // map.erase(second);
                ++second;
                // ++first;
            } else
            {
                // discard all further with that id
                // cout << "   discard\n";
                int discard_id = first->component_id;
                while (first->component_id == discard_id && second != map.end()) {
                    // first = map.erase(first);
                    first = second;
                    second = std::next(first);
                }
                if (first->component_id == discard_id) {
                    discard_first = true;
                }
            }
        } else
        {
            // keep
            if (first != w) {
                *w = std::move(*first);
            }
            first = second;
            second = std::next(first);
            ++w;
        }
    }

    // keep the last first element)
    if (!discard_first) {
        if (first != w) {
            *w = std::move(*first);
        }
        ++w;
    }

    map.erase(w, map.end());
}

void imageprocessing::ReconstructAll()
{
    cout << "[START] Reconstructing all... " << endl;
    int total_views = 0;
    while (todo_views_.size() > 0)
    {
        int todo_size = todo_views_.size();
        int next_img_id = GetNextBestViewByViews();
        cout << "next_img_id = " << next_img_id << endl;

        if (next_img_id < 0)
        {
            //TODO
            // Didn't find connected views, so proceed with the best pair left
            int most_match_id = FindMaxSizeMatch(true);

            if (most_match_id < 0)
            {
                cerr << "ERROR: matches not found for left imgs ";
                for (auto v : todo_views_) {
                    cerr << v << ",";
                }
                cerr << endl;
                todo_views_.clear();
                break;
            }

            cout << "OTHER::: most_match = "
                 << image_matches_[most_match_id].image_index.first
                 << " - " << image_matches_[most_match_id].image_index.second
                 << ", " << image_matches_[most_match_id].match.size()
                 << " (id: " << most_match_id << ")"
                 << endl;

            int first_id = image_matches_[most_match_id].image_index.first;
            int second_id = image_matches_[most_match_id].image_index.second;

            ReconstructNextViewPair(first_id, second_id);
        }
        else
        {
            ReconstructNextView(next_img_id);
        }
        total_views += todo_size - todo_views_.size();
    }

    //Optimize - Bundle Adjustment

    cout << "[END] Reconstructing all... " << endl;
}

void imageprocessing::ReconstructNextViewPair(const int first_id, const int second_id)
{
    //IsPairInOrder?
    cout << "NEXT PAIR RECONSTRUCT: " << first_id
         << " - " << second_id;

    Map3D view_map;

    TriangulatePointsFromViews(first_id, second_id, view_map);
    cout << ", view_map = " << view_map.size();

    MergeAndCombinePoints(map_, view_map, max_merge_dist);
    cout << ", map = " << map_.size();

    cout << endl;

    used_views_.insert(first_id);
    used_views_.insert(second_id);

    todo_views_.erase(first_id);
    todo_views_.erase(second_id);
}

void imageprocessing::MergeAndCombinePoints(Map3D &map, const Map3D &local_map, const double max_keep_dist)
{
    map.insert(map.end(), local_map.begin(), local_map.end());
    CombineMapComponents(map, max_keep_dist);
}

int imageprocessing::GetNextBestViewByViews()
{
    int view_id = -1;
    int max_match_cnt = 0;
    for (auto it = todo_views_.begin(); it != todo_views_.end(); ++it) {
        int view = (*it);
        int match_sum = 0;
        for (auto itc = used_views_.begin(); itc != used_views_.end(); ++itc) {
            int used_view = (*itc);
            std::pair<int, int> m_ind = std::make_pair(view, used_view);
            auto m = matches_index_.find(m_ind);
            if (m == matches_index_.end()) {
                continue;
            }
            match_sum += image_matches_[m->second].match.size();
        }
        if (max_match_cnt < match_sum) {
            view_id = view;
            max_match_cnt = match_sum;
        }
    }
    return view_id;
}

void imageprocessing::ReconstructNextView(const int next_img_id)
{
    assert(todo_views_.count(next_img_id) > 0);

    cout << "====> Process img_id = " << next_img_id << " (";
    cout << "todo_views.size = " << todo_views_.size();
    cout <<  ")" << endl;

    Map3D view_map;

    // Pairwise triangulate with each used views
    for (auto view_it = used_views_.begin(); view_it != used_views_.end(); ++view_it)
    {
        int view_id = (* view_it);
        cout << "==> Triangulate pair = " << next_img_id << ", "
             << view_id << endl;

        int first_id;
        int second_id;
        //switch idx
        if(next_img_id < view_id)
        {
            first_id = next_img_id;
            second_id = view_id;
        }
        else
        {
            first_id = view_id;
            second_id = next_img_id;
        }

        cout << "first, second = " << first_id << ", "
             << second_id << endl;

        auto m = matches_index_.find(make_pair(first_id, second_id));
        if (m == matches_index_.end()) {
            cout << "skip pair ...\n";
            continue;
        }

        // cout << "found = (" << m->first.first << ", " << m->first.second
        //           << "), -> " << m->second
        //           << endl;

        int matches_id = m->second;

        // for (int i = 0; i < image_matches_.size(); ++i) {
        //   Matches& mm = image_matches_[i];
        //   cout << "m: " << i
        //           << " (" << mm.image_index.first << ", "
        //           << mm.image_index.second << ")"
        //           << ", matches.size = " << mm.match.size() << endl;
        // }

        Matches& matches = image_matches_[matches_id];
        cout << "matches_id = " << matches_id
             << " (" << matches.image_index.first << ", "
             << matches.image_index.second << ")"
             << ", matches.size = " << matches.match.size() << endl;

        // Get the right order
        //if (matches.image_index.first != first_id) {
        //  swap(first_id, second_id);
        // cout << "SWAP: first, second = " << first_id << ", "
        //           << second_id << std::endl;
        // cout << "IMAGE_INDEX: first, second = " << matches.image_index.first << ", "
        //         << matches.image_index.second << endl;
        //   }

        // == Triangulate Points =====
        //find 2D-3D correspondences
        vector<Point3d> points3D;
        vector<Point2d> points2D;
        Find2D3DMatches(next_img_id, view_id, points3D, points2D);
        Matx34d P = Matx34d::eye();
        bool success = FindCameraPosePNP(points3D, points2D, P);
        if (not success)
        {
            cout << "Failed. Could not get a good pose estimation. skip view" << endl;
            continue;
        }
        image_features_[next_img_id].camera_pose = P;
        TriangulatePointsFromViews(first_id, second_id, view_map);
        camera_poses_.push_back(P);

    }
    cout << ", view_map = " << view_map.size();

    MergeAndCombinePoints(map_, view_map, max_merge_dist);
    cout << ", map = " << map_.size();

    used_views_.insert(next_img_id);
    todo_views_.erase(next_img_id);
}

void imageprocessing::GetCameraPose(const int first_id, const int second_id, Matx34d& P1, Matx34d& P2)
{
    auto find = matches_index_.find({first_id, second_id});
    if (find == matches_index_.end()) {
        std::cerr << "Camera Pose Error (" << first_id << ", "
                  << second_id << ")\n";
    }
    int match_index = find->second;
    vector<Point2d> points1, points2;
    keypointsMatchesToPoint2d(image_matches_[match_index].match,
                              image_features_[first_id].keypoints,
                              image_features_[second_id].keypoints,
                              points1, points2);

    Mat mask;
    Mat K = Mat(cameraMatrix.K);
    Mat E = findEssentialMat(points1, points2, K, RANSAC, 0.999, 1.0, mask);

    Mat R, T;
    double fx = cameraMatrix.K.at<double>(0,0);
    double cx = cameraMatrix.K.at<double>(0,2);
    double cy = cameraMatrix.K.at<double>(1,2);
    Point2d pp = Point2d(cx,cy);

    recoverPose(E, points1, points2, R, T, fx, pp, mask);

    bool success = CheckCoherentRotation(R);

    cout << "R:\n" << R << endl;
    cout << "T:\n" << T << endl;

    if(not success)
        cerr << "Bad rotation." << endl;

    P1 = cv::Matx34d::eye();
    P2 = cv::Matx34d(R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),T.at<double>(0),
                     R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),T.at<double>(1),
                     R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),T.at<double>(2));

    cout << "P1 = \n" << P1 << endl;
    cout << "P2 = \n" << P2 << endl;
}

bool imageprocessing::FindCameraPosePNP(vector<Point3d>& points3D, vector<Point2d>& points2D, Matx34d& P)
{
    cout << "Find CameraPosePNP... " << endl;
    if(points3D.size() <= 7 || points2D.size() <= 7 || points3D.size() != points2D.size()) {

        //something went wrong finding enough corresponding points
        cerr << "couldn't find [enough] corresponding cloud points... (only " << points3D.size() << ")"
             << endl;
        return false;
    }
    Mat rvec, T;
    vector<int> inliers;
    double minVal, maxVal;
    minMaxIdx(points3D, &minVal, &maxVal);

    solvePnPRansac(points3D, points2D, cameraMatrix.K, cameraMatrix.distCoef, rvec, T, true,
                   1000, 0.006 * maxVal, 0.99, inliers, SOLVEPNP_EPNP  );

    vector<Point2d> projected3D;
    projectPoints(points3D, rvec, T, cameraMatrix.K, cameraMatrix.distCoef, projected3D);

    /*if(inliers.size() == 0)
    { //get inliers
        for(int i = 0; i < projected3D.size(); i++) {
            if(cv::norm(projected3D[i] - points2D[i]) < 8.0)
            {
                inliers.push_back(i);
            }
        }
    }*/

    if (cv::norm(T) > 200.0)
    {
        // this is bad...
        cerr << "estimated camera movement is too big, skip this camera" << endl;
    }

    Mat R;
    Rodrigues(rvec, R);
    if(!CheckCoherentRotation(R))
    {
        cerr << "rotation is incoherent. we should try a different base view..." << endl;
    }

    cout << "found t = " << "\n"<< T << "\nR = \n" << R << endl;

    P = Matx34d(R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),T.at<double>(0),
                R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),T.at<double>(1),
                R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),T.at<double>(2));

    cout << "new P:\n" << P << endl;
    return true;
}

void imageprocessing::Find2D3DMatches(const int first_id, const int second_id, vector<Point3d>& points3D, vector<Point2d>& points2D)
{
    cout << "Find 2D-3D-Matches..." << endl;
    points3D.clear();
    auto find = matches_index_.find({first_id, second_id});
    if (find == matches_index_.end()) {
        std::cerr << "Find2D3DMatches Error (" << first_id << ", "
                  << second_id << ")\n";
    }
    int match_index = find->second;

    //scan all map Points
    for(const WorldPoint3D& mapPoint : map_)
    {
        bool found2DPoint = false;

        //scan all originating views for that 3D point
        for(const pair<int, int>& origView : mapPoint.views)
        {
            //check for 2D-2D matching
            const int originatingViewIndex      = origView.first;
            const int originatingViewFeatureIndex = origView.second;
            if(originatingViewIndex != second_id)
                continue;

            //scan all 2D-2D matches between originating view and new view
            for(const DMatch& m : image_matches_[match_index].match)
            {
                int matched2DPointInNewView = -1;
                if(originatingViewIndex < first_id)
                { //originating view is 'left'

                    if(m.queryIdx == originatingViewFeatureIndex)
                    {
                        matched2DPointInNewView = m.trainIdx;
                    }
                }
                else
                { //originating view is 'right'

                    if(m.trainIdx == originatingViewFeatureIndex)
                    {
                        matched2DPointInNewView = m.queryIdx;
                    }
                }
                if(matched2DPointInNewView >= 0)
                {
                    //This point is matched in the new view
                    vector<Point2d> newViewFeatures;
                    KeypointsToPoint2d(image_features_[first_id].keypoints, newViewFeatures);
                    points2D.push_back(newViewFeatures.at(matched2DPointInNewView));
                    points3D.push_back(mapPoint.pt);
                    found2DPoint = true;
                    break;
                }
            }

            if(found2DPoint)
            {
                break;
            }
        }
    }
    cout << "points3D = " << points3D.size() << endl;
}

void imageprocessing::displayAllImages()
{
    cv::Mat img_show;
    namedWindow("ORB", WINDOW_NORMAL);
    for(int i=0; i<image_matches_.size();i++)
    {
        drawMatches(image_features_[image_matches_[i].image_index.first].image, image_features_[image_matches_[i].image_index.first].keypoints,
                image_features_[image_matches_[i].image_index.second].image, image_features_[image_matches_[i].image_index.second].keypoints,
                image_matches_[i].match, img_show);
        imshow("ORB", img_show);
        waitKey();
    }
}


void imageprocessing::keypointsMatchesToPoint2d(vector<DMatch> matches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
                                                vector<Point2d> &points1, vector<Point2d> &points2)
{
    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        // Get the position of first image keypoints
        double x = keypoints1[it->queryIdx].pt.x;
        double y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(Point2d(x, y));
        // Get the position of second image keypoints
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back(Point2d(x, y));
    }
}

void imageprocessing::KeypointsToPoint2d(vector<KeyPoint> keypoints, vector<Point2d> &points2d)
{
    for(auto it = keypoints.begin(); it != keypoints.end(); ++it)
        points2d.push_back(it->pt);
}

void imageprocessing::WorldPointsToVec(vector<Point3d> &point3d)
{
    for(int i = 0; i < map_.size(); i++)
    {
        point3d.push_back(map_[i].pt);
    }
}

bool imageprocessing::CheckCoherentRotation(Mat& R){

    if(fabsf(determinant(R))-1.0 > 1e-07) {

        std::cout << "det(R) != +-1.0, this is not a rotation matrix" << std::endl;
        return false;
    }
    return true;
}

void imageprocessing::GenerateAllPairs()
{
    if (!image_pairs_.empty()) return;
    for (int i = 0; i < imageList.size() - 1; ++i) {
        for (int j = i + 1; j < imageList.size(); ++j) {
            image_pairs_.push_back({i, j});
        }
    }
}

void imageprocessing::OptimizeBundle()
{
    cout <<"Bundle adjuster..." << endl;
    BundleAdjustment::adjustBundle(map_, camera_poses_, cameraMatrix, image_features_);
}

void imageprocessing::ExportPoints(string &filename)
{
    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("triangulation.xyz", "wt");
    if (fpts == NULL) cout<<"Error writing triangulation"<<endl;
    fprintf(fpts, "%i\n\n", map_.size());
    for (size_t i = 0; i < map_.size(); i++)
        fprintf(fpts, "H %f %f %f\n", map_[i].pt.x, map_[i].pt.y, map_[i].pt.z); // Format: x, y, z
    fclose(fpts);
}

/*void imageprocessing::Tracking()
{
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap(0);

    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat img_cap;
    Features feature_cap;

    while(1)
    {
        cap.read(img_cap);
        ExtractFeatures(img_cap, feature_cap);
        cout << "feature_cap.keypoints = " << feature_cap.keypoints << endl;

        if(feature_cap.descriptors.empty())
            cout << "feature_cap.descriptor is emtpy!" << endl;



        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
            break;
    }

    cap.release();
    destroyAllWindows();

}*/
