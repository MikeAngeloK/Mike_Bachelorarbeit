//***********************************************
//HEADERS
//***********************************************
#include "Utils.h"

class BundleAdjustment {

  private:

  public:
    BundleAdjustment(){

    }

    ~BundleAdjustment(){

    }

    static void adjustBundle(std::vector<WorldPoint3D>& pointCloud, std::vector<cv::Matx34d>& cameraPoses,
                      Intrinsics& intrinsics,const std::vector<Features>& image2dFeatures);

};







