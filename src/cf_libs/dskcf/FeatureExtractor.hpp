#ifndef _FEATURE_EXTRACTOR_HPP_
#define _FEATURE_EXTRACTOR_HPP_

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

/*
This class represents a C++ implementation of the DS-KCF Tracker [1]. In particular
this class is a template class with virtual methods only. Other classes as for example
HOGFeatureExtractor are derived from this class

References:
[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao,
DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
*/

#include <array>
#include "feature_channels.hpp"
#include <opencv2/core.hpp>

class FeatureExtractor
{
public:
	FeatureExtractor();
	virtual ~FeatureExtractor();

	virtual std::shared_ptr< FC > getFeatures( const cv::Mat & image, const cv::Rect_< double > & boundingBox ) const = 0;

private:
};

#endif
