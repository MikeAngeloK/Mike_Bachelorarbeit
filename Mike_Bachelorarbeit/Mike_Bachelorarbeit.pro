QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        BundleAdjustment.cpp \
        imageprocessing.cpp \
        main.cpp

INCLUDEPATH += D:\msys64\mingw64\include
INCLUDEPATH += D:\msys64\mingw64\include\opencv4
INCLUDEPATH += D:\msys64\mingw64\include\eigen3

LIBS += D:\msys64\mingw64\bin\libopencv_core430.dll
LIBS += D:\msys64\mingw64\bin\libopencv_highgui430.dll
LIBS += D:\msys64\mingw64\bin\libopencv_imgcodecs430.dll
LIBS += D:\msys64\mingw64\bin\libopencv_imgproc430.dll
LIBS += D:\msys64\mingw64\bin\libopencv_features2d430.dll
LIBS += D:\msys64\mingw64\bin\libopencv_calib3d430.dll
LIBS += D:\msys64\mingw64\bin\libopencv_videoio430.dll
LIBS += D:\msys64\mingw64\bin\libopencv_xfeatures2d430.dll
LIBS += D:\msys64\mingw64\bin\libopencv_flann430.dll
LIBS += D:\msys64\mingw64\bin\libceres.dll
LIBS += D:\msys64\mingw64\bin\libglog.dll
LIBS += D:\msys64\mingw64\bin\libgflags.dll
LIBS += D:\msys64\mingw64\bin\libopencv_sfm430.dll

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    BundleAdjustment.h \
    Utils.h \
    ccomp.hpp \
    imageprocessing.h
