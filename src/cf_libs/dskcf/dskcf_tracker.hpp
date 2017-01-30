#ifndef _DSKCF_TRACKER_HPP_
#define _DSKCF_TRACKER_HPP_
/*
// License Agreement (3-clause BSD License)
// Copyright (c) 2016, Jake Hall, Massimo Camplan, Sion Hannuna.
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


#include <array>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/core.hpp>

#include "cf_tracker.hpp"
#include "kcf_tracker.hpp"
#include "math_helper.hpp"
#include "DepthSegmenter.hpp"
#include "ScaleAnalyser.hpp"
#include "FeatureExtractor.hpp"
#include "OcclusionHandler.hpp"
#include "ScaleChangeObserver.hpp"

/**
 * DskcfTracker implements a depth scaling kernelised correlation filter as
 * described in \cite DSKCF.
 *
 *  [1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao,
 *  DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
 */
class DskcfTracker : public CfTracker
{
public:
	DskcfTracker();
	virtual ~DskcfTracker();
	virtual bool update(const std::array< cv::Mat, 2 > & frame, cv::Rect_< double > & boundingBox);
	virtual bool update(const std::array< cv::Mat, 2 > & frame, cv::Rect_<double>& boundingBox, std::vector<int64> &timePerformanceVector);
	virtual bool reinit(const std::array< cv::Mat, 2 > & frame, cv::Rect_< double > & boundingBox);

	virtual TrackerDebug* getTrackerDebug();
	virtual const std::string getId();
private:

	/** The occlusion handler associated with this object */
	std::shared_ptr< OcclusionHandler > m_occlusionHandler;
};

#endif
