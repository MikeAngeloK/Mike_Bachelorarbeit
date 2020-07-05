#include <QCoreApplication>
#include "imageprocessing.h"

int main(int argc, char** argv) {

    string inputFile = "Imagelist.xml";
    string outputFile = "FeaturePoints.xml";
    string calibrationFile = "CameraCalibrationFile.xml"; // CameraCalibrationFile.xml - Mike Pocofone F1         camera_calibration_template.xml

    imageprocessing ip;
    ip.ReadStringList(inputFile);
    ip.GetCameraMatrix(calibrationFile);
    ip.MatchImageFeatures(7);
    ip.InitReconstruction();
    ip.ReconstructAll();
    
    ip.OptimizeBundle();
    ip.ExportPoints(outputFile);
    //ip.displayAllImages();
    waitKey();

    return 0;
}
