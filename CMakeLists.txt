# License Agreement (3-clause BSD License)
# Copyright (c) 2015, Klaus Haag, all rights reserved.
# Third party copyrights and patents are property of their respective owners.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the names of the copyright holders nor the names of the contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall copyright holders or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.

project(CfTracking)

cmake_minimum_required(VERSION 2.8.7)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# add c++11 support
if(CMAKE_COMPILER_IS_GNUCC)
    ADD_DEFINITIONS ( -std=c++11 -Wall )
endif(CMAKE_COMPILER_IS_GNUCC)

# add OpenCV
set(OPENCV_DIR_HINT "")

if(WIN32)
    get_filename_component(OPENCV_DIR_PLATFORM $ENV{OPENCV_DIR} DIRECTORY)
    get_filename_component(OPENCV_DIR_HINT ${OPENCV_DIR_PLATFORM} DIRECTORY)
endif(WIN32)

set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED HINTS ${OPENCV_DIR_HINT})
find_package(TBB REQUIRED)

# add 3rdparty source
set(CF_TCLAP_DIR "src/3rdparty/tclap")
set(CF_CV_EXT_DIR "src/3rdparty/cv_ext")
set(CF_PIOTR_DIR "src/3rdparty/piotr")

include_directories(${OpenCV_DIR}/include)
include_directories(${TBB_INCLUDE_DIRS})
include_directories(${CF_TCLAP_DIR}
                    ${CF_CV_EXT_DIR}
                    ${CF_PIOTR_DIR}
                    ${CF_PIOTR_DIR}/src
                    src/main)

include_directories(src/cf_libs/kcf
                    src/cf_libs/dsst
                    src/cf_libs/common
                    src/cf_libs/dskcf)

set(CF_PIOTR_SOURCES
    ${CF_PIOTR_DIR}/gradientMex.hpp
    ${CF_PIOTR_DIR}/src/gradientMex.cpp
    ${CF_PIOTR_DIR}/src/sse.hpp
    ${CF_PIOTR_DIR}/src/wrappers.hpp)

set(CF_LIB_COMMON_SOURCES
    src/cf_libs/common/feature_channels.hpp
    src/cf_libs/common/mat_consts.hpp
    src/cf_libs/common/math_helper.hpp
    src/cf_libs/common/math_helper.cpp
    src/cf_libs/common/cf_tracker.hpp
    src/cf_libs/common/tracker_debug.hpp
    src/cf_libs/common/cv_ext.hpp
    src/cf_libs/common/optional.hpp
    src/cf_libs/common/Typedefs.hpp
    ${CF_CV_EXT_DIR}/shift.cpp
    ${CF_CV_EXT_DIR}/shift.hpp
    ${CF_CV_EXT_DIR}/math_spectrums.cpp
    ${CF_CV_EXT_DIR}/math_spectrums.hpp
    ${CF_CV_EXT_DIR}/psr.hpp
    ${CF_PIOTR_SOURCES})

set(CF_MAIN_SOURCES
    src/main/image_acquisition.hpp
    src/main/image_acquisition.cpp
    ${CF_CV_EXT_DIR}/init_box_selector.hpp
    ${CF_CV_EXT_DIR}/init_box_selector.cpp
    ${CF_CV_EXT_DIR}/tracker_run.hpp
    ${CF_CV_EXT_DIR}/tracker_run.cpp
    ${CF_TCLAP_DIR}/tclap/CmdLine.h)

add_executable(DSKCFcpp
    src/main/main_dskcf.cpp
    src/cf_libs/kcf/kcf_debug.hpp
    src/cf_libs/kcf/kcf_tracker.hpp
    src/cf_libs/dskcf/dskcf_tracker_run.cpp
    src/cf_libs/dskcf/dskcf_tracker_run.hpp
    src/cf_libs/dskcf/dskcf_tracker.cpp
    src/cf_libs/dskcf/dskcf_tracker.hpp
    src/cf_libs/dskcf/DepthSegmenter.cpp
    src/cf_libs/dskcf/DepthSegmenter.hpp
    src/cf_libs/dskcf/FeatureExtractor.cpp
    src/cf_libs/dskcf/FeatureExtractor.hpp
    src/cf_libs/dskcf/FeatureChannelProcessor.hpp
    src/cf_libs/dskcf/OcclusionHandler.cpp
    src/cf_libs/dskcf/OcclusionHandler.hpp
    src/cf_libs/dskcf/ScaleAnalyser.cpp
    src/cf_libs/dskcf/ScaleAnalyser.hpp
    src/cf_libs/dskcf/ScaleChangeObserver.hpp
    src/cf_libs/kcf/GaussianKernel.cpp
    src/cf_libs/kcf/GaussianKernel.hpp
    src/cf_libs/kcf/HOGFeatureExtractor.cpp
    src/cf_libs/kcf/HOGFeatureExtractor.hpp
    src/cf_libs/kcf/Kernel.hpp
    src/cf_libs/kcf/Kernel.cpp
    src/cf_libs/kcf/kcf_tracker.hpp
    src/cf_libs/kcf/kcf_tracker.cpp
    src/cf_libs/dskcf/LinearFeatureChannelProcessor.cpp
    src/cf_libs/dskcf/LinearFeatureChannelProcessor.h
    src/cf_libs/dskcf/ConcatenateFeatureChannelProcessor.cpp
    src/cf_libs/dskcf/ConcatenateFeatureChannelProcessor.h
    src/cf_libs/dskcf/ColourFeatureChannelProcessor.cpp
    src/cf_libs/dskcf/ColourFeatureChannelProcessor.h
    src/cf_libs/dskcf/DepthFeatureChannelProcessor.cpp
    src/cf_libs/dskcf/DepthFeatureChannelProcessor.h
    src/cf_libs/common/KalmanFilter2D.cpp
    src/cf_libs/common/KalmanFilter2D.h
    src/cf_libs/common/circularbuffer.hpp
    src/cf_libs/common/KalmanFilter1D.cpp
    src/cf_libs/common/KalmanFilter1D.h
    src/cf_libs/common/DepthHistogram.cpp
    src/cf_libs/common/DepthHistogram.h
    src/cf_libs/kcf/DepthWeightKCFTracker.cpp
    src/cf_libs/kcf/DepthWeightKCFTracker.h
    ${CF_MAIN_SOURCES}
    ${CF_LIB_COMMON_SOURCES})
	

#set_property(TARGET ${TBB_LIBRARIES} PROPERTY DBG_POSTFIX _debug)
#message("BUILD TYPE = '${CMAKE_BUILD_TYPE}'")
#target_link_libraries( DSKCFcpp ${OpenCV_LIBS} optimized ${TBB_tbb_LIBRARY_RELEASE} debug ${TBB_tbb_LIBRARY_RELEASE})
#target_link_libraries( DSKCFcpp ${OpenCV_LIBS} optimized ${TBB_tbb_LIBRARY_RELEASE})
#target_link_libraries( DSKCFcpp ${OpenCV_LIBS} debug ${TBB_tbb_LIBRARY_RELEASE})

set(TBB_LIBRARIES optimized ${TBB_LIBRARY} ${TBB_tbb_LIBRARY_RELEASE} debug ${TBB_tbb_LIBRARY_DEBUG})
message("FINALLL TBB_LIBRARIES = '${TBB_LIBRARIES}'")
target_link_libraries(DSKCFcpp ${OpenCV_LIBS} ${TBB_LIBRARIES})
#IF(CMAKE_BUILD_TYPE MATCHES Release)
#	message("RELEASE")
#    target_link_libraries(DSKCFcpp ${OpenCV_LIBS} ${TBB_tbb_LIBRARY_RELEASE})
#ELSE()
#	message("DEBUG")
#    target_link_libraries(DSKCFcpp ${OpenCV_LIBS} ${TBB_tbb_LIBRARY_DEBUG})
#ENDIF(CMAKE_BUILD_TYPE MATCHES Release)

