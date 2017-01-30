#ifndef _FEATURECHANNELPROCESSOR_HPP_
#define _FEATURECHANNELPROCESSOR_HPP_

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
this class is a template class with virtual methods only. Other classes are derived from this class, such as
ColourFeatureChannelProcessor

References:
[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao,
DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
*/

#include <memory>
#include <vector>
#include <opencv2/core/core.hpp>

#include "feature_channels.hpp"

/**
 * FeatureChannelProcessor is responsible for taking a collection of feature channels and processing them.
 */
class FeatureChannelProcessor
{
public:
  /**
   * Concatenate the feature channels.
   *
   * @param featureChannels The input collection of feature channels to be processed.
   *
   * @returns A new collection of processed feature channels.
   */
  virtual const std::vector< std::shared_ptr< FC > > concatenate(
      const std::vector< std::shared_ptr< FC > > & featureChannels ) const = 0;

  /**
   * Process the frame so that each element of the resulting collection is associated with the element in the feature channel collection.
   *
   * @param frame The collection of images for the current frame.
   *
   * @returns A collection ordered so that each element matches its associated feature channel.
   */
  virtual const std::vector< cv::Mat > concatenate( const std::vector< cv::Mat > & frame ) const = 0;

  /**
   * Combine the maximum responses of each feature channel.
   *
   * @param maxResponses The maximum responses for each feature channel.
   *
   * @returns The combined maximum response.
   */
  virtual const double concatenate( const std::vector< double > & maxResponses ) const = 0;

  /**
   * Combine the positions of each tracker
   *
   * @param positions The positions given my each tracker's detect call
   *
   * @returns The combined position
   */
  virtual const cv::Point_< double > concatenate( const std::vector< cv::Point_< double > > & positions ) const = 0;
};

#endif
