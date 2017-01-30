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
the depth histogram presented in [1] is implemented within this class

References:
[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao,
DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
*/
#ifndef CFTRACKING_HISTOGRAM_H
#define CFTRACKING_HISTOGRAM_H

#include <vector>

#include <opencv2/core/mat.hpp>

class DepthHistogram
{
public:
  struct Labels{ std::vector< int > labels; std::vector< float > centers; std::vector< int > labelsC;};

  DepthHistogram();

  const std::vector< int > getPeaks() const;
  const std::vector< int > getPeaks( const int minimumPeakdistance, const double minimumPeakHeight = 0.005 ) const;
  const DepthHistogram::Labels getLabels( const std::vector< int > & peaks ) const;

  const int depthToBin( const double depth ) const;
  const double binToDepth( const float bin ) const;
  const int depthToLabel( const double depth, const std::vector< int > & labels ) const;
  //const double depthToCentroid( const double depth, const Labels & labels ) const;
  const int depthToPeak( const double depth, const std::vector< int > & peaks ) const;
  const int depthToCentroid( const double depth, const std::vector< float > & centroids ) const;

  const bool empty() const;
  const size_t size() const;
  const double minimum() const;
  const double maximum() const;
  const float estStep() const;

  const float operator[]( const uint i ) const;

  static const DepthHistogram createHistogram( const uint step, const cv::Mat & region, const cv::Mat1b & mask );
  void visualise( const std::string & string );
private:
  cv::Mat1f m_bins;
  double m_minimum, m_maximum;
  float estimatedStep;

  const Labels kmeans( const std::vector< float > & centroids ) const;
};

#endif //CFTRACKING_HISTOGRAM_H
