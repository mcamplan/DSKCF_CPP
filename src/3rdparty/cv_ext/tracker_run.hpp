/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
// License Agreement
// For Open Source Computer Vision Library
// (3-clause BSD License)
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//M*/

/*
// Original file: https://github.com/Itseez/opencv_contrib/blob/292b8fa6aa403fb7ad6d2afadf4484e39d8ca2f1/modules/tracking/samples/tracker.cpp
// Modified by Klaus Haag file: https://github.com/klahaag/cf_tracking/blob/master/src/3rdparty/cv_ext/tracker_run.cpp
// + Authors: Jake Hall, Massimo Camplan, Sion Hannuna
// * Add a variety of additional features to visualize tracker, save results according to RGBD dataset (see details below) and to save processing
//   time as in the DS-KCF paper
//  Princeton RGBD data: Shuran Song and Jianxiong Xiao. Tracking Revisited using RGBD Camera: Baseline and Benchmark. 2013.
*/

#ifndef TRACKER_RUN_HPP_
#define TRACKER_RUN_HPP_

#include <tclap/CmdLine.h>
#include <array>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "cf_tracker.hpp"
#include "tracker_debug.hpp"
#include "image_acquisition.hpp"

struct Parameters{
    std::string sequencePathRGB;
    std::string sequencePathDepth;
    std::string outputFilePath;
    std::string imgExportPath;
    std::string expansionRGB;
    std::string expansionDepth;
    cv::Rect initBb;
    int deviceRGB;
    int deviceDepth;
    int startFrame;
    bool showOutput;
    bool paused;
    bool repeat;
    bool isMockSequenceRGB;
    bool isMockSequenceDepth;
	bool useDepth;
};

class TrackerRun
{
public:
    TrackerRun(std::string windowTitle);
    virtual ~TrackerRun();
    bool start(int argc, const char** argv);
    void setTrackerDebug(TrackerDebug* debug);

private:
    Parameters parseCmdArgs(int argc, const char** argv);
    bool init();
    bool run();
    bool update();
    void printResults(const cv::Rect_<double>& boundingBox, bool isConfident, bool isTracked);
	void printResultsTiming(const std::vector<int64> &singleFrameTiming);

protected:
    virtual CfTracker* parseTrackerParas(TCLAP::CmdLine& cmd, int argc, const char** argv) = 0;
private:
    std::array< cv::Mat, 2 > _image;
    CfTracker* _tracker;
    std::string _windowTitle;
    Parameters _paras;
    cv::Rect_<double> _boundingBox;
    std::array< ImageAcquisition, 2 > _cap;
    std::ofstream _resultsFile,_resultsFileTime;
    TCLAP::CmdLine _cmd;
    TrackerDebug* _debug;
    int _frameIdx;
    bool _isPaused;
    bool _isStep;
    bool _exit;
    bool _hasInitBox;
    bool _isTrackerInitialzed;
    bool _targetOnFrame;
    bool _updateAtPos;
	int _imageIndex;
	std::vector< double > frameTime;
};

#endif
