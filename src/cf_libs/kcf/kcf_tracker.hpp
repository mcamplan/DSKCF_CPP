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
// Original file: https://github.com/klahaag/cf_tracking/blob/master/src/cf_libs/kcf/kcf_tracker.hpp
// + Authors: Jake Hall, Massimo Camplan, Sion Hannuna
// * We modified the original code of  Klaus Haag, such that different classes are used for the different KCF components
//   in a more modular way, to support DS-KCF code

It is implemented closely to the Matlab implementation by the original authors:
http://home.isr.uc.pt/~henriques/circulant/
However, some implementation details differ and some difference in performance
has to be expected.

This specific implementation features the scale adaption, sub-pixel
accuracy for the correlation response evaluation and a more robust
filter update scheme [2] used by Henriques, et al. in the VOT Challenge 2014.

As default scale adaption, the tracker uses the 1D scale filter
presented in [3]. The scale filter can be found in scale_estimator.hpp.
Additionally, target loss detection is implemented according to [4].

Every complex matrix is as default in CCS packed form:
see : https://software.intel.com/en-us/node/504243
and http://docs.opencv.org/modules/core/doc/operations_on_arrays.html

References:
[1] J. Henriques, et al.,
"High-Speed Tracking with Kernelized Correlation Filters,"
PAMI, 2015.

[2] M. Danelljan, et al.,
�Adaptive Color Attributes for Real-Time Visual Tracking,�
in Proc. CVPR, 2014.

[3] M. Danelljan,
"Accurate Scale Estimation for Robust Visual Tracking,"
Proceedings of the British Machine Vision Conference BMVC, 2014.

[4] D. Bolme, et al.,
�Visual Object Tracking using Adaptive Correlation Filters,�
in Proc. CVPR, 2010.
*/

#ifndef KCF_TRACKER_HPP_
#define KCF_TRACKER_HPP_

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <algorithm>
#include <array>
#include <memory>

#include "cv_ext.hpp"
#include "feature_channels.hpp"
#include "gradientMex.hpp"
#include "mat_consts.hpp"
#include "math_helper.hpp"
#include "cf_tracker.hpp"
#include "kcf_debug.hpp"
#include "ScaleAnalyser.hpp"
#include "Kernel.hpp"
#include "ScaleChangeObserver.hpp"
#include "optional.hpp"

struct KcfParameters
{
  double padding;
  double lambda;
  double outputSigmaFactor;
  double interpFactor;
  int cellSize;

  KcfParameters();
};

struct DetectResult { cv::Point_< double > position;double maxResponse; };

class KcfTracker : public ScaleChangeObserver
{
  friend class KCFTest;
public:
  KcfTracker( KcfParameters paras, std::shared_ptr< Kernel > kernel );
  virtual ~KcfTracker();

  void init( const cv::Mat & image, const std::shared_ptr< FC > & features, const cv::Point_< double > & position );
  void update( const cv::Mat & image, const std::shared_ptr< FC > & features, const cv::Point_< double > & position );
  virtual void onScaleChange( const cv::Size_< double > & targetSize, const cv::Size_< double > & windowSize, const cv::Mat2d & yf, const cv::Mat1d & cosineWindow );

  const DetectResult detect( const cv::Mat & image, const std::shared_ptr< FC > & features, const cv::Point_< double > & position ) const;
  std::shared_ptr< KcfTracker > duplicate() const;
private:
  bool m_isInitialized;
  int m_frameID;
  int m_cellSize;
  double m_lambda;
  double m_interpFactor;
  cv::Mat1d m_cosineWindow;
  cv::Mat2d m_alphaNumeratorf;
  cv::Mat2d m_alphaDenominatorf;
  cv::Mat2d m_alphaf;
  cv::Mat2d m_yf;
  std::shared_ptr< FC > m_xf;
  std::shared_ptr< Kernel > m_kernel;

  void updateModel( const cv::Mat & image, const std::shared_ptr< FC > & features );
protected:
  struct Response { cv::Mat1d response; double maxResponse; cv::Point maxResponsePosition; };
  struct TrainingData { std::shared_ptr< FC > xf; cv::Mat numeratorf, denominatorf; };

  const TrainingData getTrainingData( const cv::Mat & image, const std::shared_ptr< FC > & features ) const;
  const DetectResult detectModel( const cv::Mat & image, const std::shared_ptr< FC > & features, const cv::Point_< double > & newPos ) const;
  const Response getResponse( const cv::Mat & image, const std::shared_ptr< FC > & features, const cv::Point_< double > & pos ) const;
  const cv::Mat detectResponse( const cv::Mat & image, const std::shared_ptr< FC > & features, const cv::Point_< double > & pos ) const;
};

#endif /* KCF_TRACKER_H_ */
