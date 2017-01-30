/*
// License Agreement (3-clause BSD License)
// Copyright (c) 2015, Klaus Haag, all rights reserved.
// Third party copyrights and patents are property of their respective owners.
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
*/

/*
// Original file: https://github.com/klahaag/cf_tracking/blob/master/src/cf_libs/common/cf_tracker.hpp
// + Authors: Jake Hall, Massimo Camplan, Sion Hannuna
// * We modified the original code of  Klaus Haag by re-organizing the code to be fit within the DS-KCF framework
*/

#ifndef TRACKER_HPP_
#define TRACKER_HPP_

#include "opencv2/core/core.hpp"
#include "tracker_debug.hpp"

#include <array>

class CfTracker
{
public:
  virtual ~CfTracker() {};

  /**
   * Updates the model using the the object located in the given bounding box.
   *
   * @param[in] frame The RGB and depth component of the current frame.
   * @param[in,out] boundingBox The bounding box of the tracked object.
   *
   * @returns True if the model was successfully updated, false otherwise.
   */
  virtual bool update(const std::array< cv::Mat, 2 > & frame, cv::Rect_<double>& boundingBox) = 0;

   /**
   * Updates the model using the the object located in the given bounding box.
   *
   * @param[in] frame The RGB and depth component of the current frame.
   * @param[in,out] boundingBox The bounding box of the tracked object.
   *
   * @returns True if the model was successfully updated, false otherwise.
   */
  virtual bool update(const std::array< cv::Mat, 2 > & frame, cv::Rect_<double>& boundingBox, std::vector<int64> &timePerformanceVector) = 0;

  /**
   * Initialises the model using the the object located in the given bounding box.
   *
   * @param[in] frame The RGB and depth component of the current frame.
   * @param[in,out] boundingBox The bounding box of the tracked object.
   *
   * @returns True if the model was successfully initialised, false otherwise.
   */
  virtual bool reinit(const std::array< cv::Mat, 2 > & frame, cv::Rect_<double>& boundingBox) = 0;

  virtual TrackerDebug* getTrackerDebug() = 0;
  virtual const std::string getId() = 0;
};

#endif
