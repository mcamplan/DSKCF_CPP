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
// Original file: https://github.com/klahaag/cf_tracking/blob/master/src/cf_libs/common/feature_channels.hpp
// + Authors: Jake Hall, Massimo Camplan, Sion Hannuna
// * We modified the original code of  Klaus Haag by re-organizing the code to be fit within the DS-KCF framework
*/
#ifndef FHOG_FEATURE_CHANNELS_H_
#define FHOG_FEATURE_CHANNELS_H_

#include "opencv2/core/core.hpp"
#include "math_helper.hpp"
#include <memory>
#include <array>
#include <vector>

#include <tbb/parallel_for_each.h>
#include <tbb/parallel_for.h>

class FeatureChannels_
{
public:
  FeatureChannels_( const size_t channelCount = 31 )
  {
    this->channels.resize( channelCount );
  }

  virtual ~FeatureChannels_()
  {
  }

  static std::shared_ptr< FeatureChannels_ > concatFeatures( const std::shared_ptr< FeatureChannels_ > & left, const std::shared_ptr< FeatureChannels_ > & right )
  {
    CV_Assert( left->numberOfChannels() == right->numberOfChannels() );
    std::shared_ptr< FeatureChannels_ > result = std::make_shared< FeatureChannels_ >( left->numberOfChannels() + right->numberOfChannels() );

    std::copy( left->channels.begin(), left->channels.end(), result->channels.begin() );
    std::copy( right->channels.begin(), right->channels.end(), result->channels.begin() + left->numberOfChannels() );

    return result;
  }

  static void mulValueFeatures(std::shared_ptr<FeatureChannels_>& m, const double & value)
  {
    tbb::parallel_for_each( m->channels.begin(), m->channels.end(),
      [value]( cv::Mat & m_ ) -> void
        {
        m_ *= value;
        }
    );
  }

  static void addFeatures(std::shared_ptr<FeatureChannels_>& A, const std::shared_ptr<FeatureChannels_>& B)
  {
    CV_Assert( A->numberOfChannels() == B->numberOfChannels() );

    tbb::parallel_for< size_t >( 0, A->numberOfChannels(), 1,
      [&A, &B]( size_t index ) -> void
      {
        A->channels[ index ] += B->channels[ index ];
      }
    );
  }

  static cv::Mat sumFeatures(const std::shared_ptr<FeatureChannels_>& x)
  {
    cv::Mat result = x->channels[0].clone();

    for (size_t i = 1; i < x->numberOfChannels(); ++i)
    {
      result += x->channels[i];
    }

    return result;
  }

  static void mulFeatures(std::shared_ptr<FeatureChannels_>& features, const cv::Mat& m)
  {
    tbb::parallel_for_each( features->channels.begin(), features->channels.end(),
      [&m]( cv::Mat & channel ) -> void
      {
        channel = channel.mul( m );
      }
    );
  }

  static std::shared_ptr<FeatureChannels_> dftFeatures( const std::shared_ptr<FeatureChannels_>& features, int flags = 0)
  {
    auto result = std::make_shared<FeatureChannels_>( features->numberOfChannels() );

    tbb::parallel_for< size_t >( 0, result->numberOfChannels(), 1,
      [&features,&result,&flags]( size_t index ) -> void
      {
        cv::dft( features->channels[ index ], result->channels[ index ], flags );
      }
    );

    return result;
  }

  static std::shared_ptr<FeatureChannels_> idftFeatures( const std::shared_ptr<FeatureChannels_>& features)
  {
    auto result = std::make_shared<FeatureChannels_>( features->numberOfChannels() );

    tbb::parallel_for< size_t >( 0, result->numberOfChannels(), 1,
      [&features,&result]( size_t index ) -> void
      {
        cv::idft( features->channels[ index ], result->channels[ index ], cv::DFT_REAL_OUTPUT | cv::DFT_SCALE );
      }
    );

    return result;
  }

  static double squaredNormFeaturesNoCcs(const std::shared_ptr<FeatureChannels_>& Af)
  {
    int n = Af->channels[0].rows * Af->channels[0].cols;
    double sum_ = 0;
    cv::Mat elemMul;

    for (size_t i = 0; i < Af->numberOfChannels(); ++i)
    {
      mulSpectrums(Af->channels[i], Af->channels[i], elemMul, 0, true);
      sum_ += static_cast< double >(cv::sum(elemMul)[0]);
    }

    return sum_ / n;
  }

  static std::shared_ptr<FeatureChannels_> mulSpectrumsFeatures( const std::shared_ptr<FeatureChannels_>& Af, const std::shared_ptr<FeatureChannels_>& Bf, bool conjBf)
  {
    CV_Assert( Af->numberOfChannels() == Bf->numberOfChannels() );

    auto result = std::make_shared<FeatureChannels_>( Af->numberOfChannels() );

    tbb::parallel_for< size_t >( 0, Af->numberOfChannels(), 1,
      [&Af, &Bf, &result, conjBf]( size_t index ) -> void
      {
        mulSpectrums( Af->channels[ index ], Bf->channels[ index ], result->channels[ index ], 0, conjBf );
      }
    );

    return result;
  }

  const size_t numberOfChannels() const
  {
    return this->channels.size();
  }

  std::vector< cv::Mat > channels;
};

typedef FeatureChannels_ FC;

#endif
