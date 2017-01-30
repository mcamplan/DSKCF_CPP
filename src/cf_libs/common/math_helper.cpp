/*
//  License Agreement (3-clause BSD License)
//  Copyright (c) 2015, Klaus Haag, all rights reserved.
//  Third party copyrights and patents are property of their respective owners.
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * Neither the names of the copyright holders nor the names of the contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall copyright holders or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.
*/

/*
// Original file: https://github.com/klahaag/cf_tracking/blob/master/src/cf_libs/common/math_helper.cpp
// + Authors: Jake Hall, Massimo Camplan, Sion Hannuna
// * We modified the original code of  Klaus Haag, adding functionality to deal with DS-KCF system, especially
//   for depth menaging
*/

#include "math_helper.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/flann.hpp>

int mod(int dividend, int divisor)
{
  // http://stackoverflow.com/questions/12276675/modulus-with-negative-numbers-in-c
  return ((dividend % divisor) + divisor) % divisor;
}

// fftshift implements the MATLAB function fftshift
// This is achieved by switching the diagonal quadrants
//
// @param a : The matrix to be shifted
//
// @returns : The shifted matrix
cv::Mat fftshift( const cv::Mat & a )
{
	cv::Mat inquadrant[ 4 ];
	cv::Mat outquadrant[ 4 ];
  int size[] = { a.rows, a.cols };
  cv::Mat result = cv::Mat::zeros( a.dims, size, a.type() );

	int width = a.cols / 2;
	int height = a.rows / 2;

  inquadrant[ 0 ] = cv::Mat( a, cv::Rect( 0, 0, width, height ) );
  inquadrant[ 1 ] = cv::Mat( a, cv::Rect( width, 0, (a.cols - width), height ) );
  inquadrant[ 2 ] = cv::Mat( a, cv::Rect( 0, height, width, ( a.rows - height ) ) );
  inquadrant[ 3 ] = cv::Mat( a, cv::Rect( width, height, (a.cols - width), ( a.rows - height ) ) );

  outquadrant[ 0 ] = cv::Mat( result, cv::Rect( 0, 0, (a.cols - width), ( a.rows - height ) ) );
  outquadrant[ 1 ] = cv::Mat( result, cv::Rect( (a.cols - width), 0, width, ( a.rows - height ) ) );
  outquadrant[ 2 ] = cv::Mat( result, cv::Rect( 0, ( a.rows - height ), (a.cols - width), height ) );
  outquadrant[ 3 ] = cv::Mat( result, cv::Rect( (a.cols - width), ( a.rows - height ), width, height ) );

  inquadrant[ 0 ].copyTo( outquadrant[ 3 ] );
  inquadrant[ 1 ].copyTo( outquadrant[ 2 ] );
  inquadrant[ 2 ].copyTo( outquadrant[ 1 ] );
  inquadrant[ 3 ].copyTo( outquadrant[ 0 ] );

  return result;
}

cv::Mat ifftshift( const cv::Mat & a )
{
	cv::Mat inquadrant[ 4 ];
	cv::Mat outquadrant[ 4 ];
  int size[] = { a.rows, a.cols };
  cv::Mat result = cv::Mat::zeros( a.dims, size, a.type() );

	int width = a.cols / 2;
	int height = a.rows / 2;

  inquadrant[ 0 ] = cv::Mat( a, cv::Rect( 0, 0, (a.cols - width), ( a.rows - height ) ) );
  inquadrant[ 1 ] = cv::Mat( a, cv::Rect( (a.cols - width), 0, width, ( a.rows - height ) ) );
  inquadrant[ 2 ] = cv::Mat( a, cv::Rect( 0, ( a.rows - height ), (a.cols - width), height ) );
  inquadrant[ 3 ] = cv::Mat( a, cv::Rect( (a.cols - width), ( a.rows - height ), width, height ) );

  outquadrant[ 0 ] = cv::Mat( result, cv::Rect( 0, 0, width, height ) );
  outquadrant[ 1 ] = cv::Mat( result, cv::Rect( width, 0, (a.cols - width), height ) );
  outquadrant[ 2 ] = cv::Mat( result, cv::Rect( 0, height, width, ( a.rows - height ) ) );
  outquadrant[ 3 ] = cv::Mat( result, cv::Rect( width, height, (a.cols - width), ( a.rows - height ) ) );

  inquadrant[ 0 ].copyTo( outquadrant[ 3 ] );
  inquadrant[ 1 ].copyTo( outquadrant[ 2 ] );
  inquadrant[ 2 ].copyTo( outquadrant[ 1 ] );
  inquadrant[ 3 ].copyTo( outquadrant[ 0 ] );

  return result;
}

cv::Mat linSpace( double x1, double x2, int a )
{
	a = a + 1;
	cv::Mat result = cv::Mat::zeros( 1, &a, CV_64F );
	double step = ( x2 - x1 ) / a;

	for( int i = 0; i < a; i++ )
	{
		result.at< double >( i ) = x1 + (i * step);
	}

	return result;
}

double weightDistanceLogisticOnDepth( double targetDepth, double candidateDepth, double targetSTD )
{
	double dist = (
		std::abs( targetDepth - candidateDepth ) /
		( 3 * targetSTD )
	);

	return 1 - sigmFunction( dist, 0, 1, 1, 0.5, 3.2, 1.94 );
}

double sigmFunction( double x, double A, double K, double Q, double ni, double B, double M )
{
	return (
		( A + ( K - A ) ) /
		std::pow(
        1 + ( Q * std::exp( -B * ( x - M ) ) ),
        ( 1.0 / ni )
    )
	);
}

void visualiseFourier( const std::string & windowName, const cv::Mat & image )
{
  if( !image.empty() )
  {
    cv::Mat tmp;
    cv::resize( fftshift( image ), tmp, image.size() * 2, cv::INTER_CUBIC );

    cv::Mat channels[ 2 ];

    cv::split( tmp, channels );

    cv::normalize( channels[ 0 ], channels[ 0 ], 0, 1, CV_MINMAX );
    cv::normalize( channels[ 1 ], channels[ 1 ], 0, 1, CV_MINMAX );

    cv::imshow( windowName + ": Real", channels[ 0 ] );
    cv::imshow( windowName + ": Complex", channels[ 1 ] );
  }
}

void visualise( const std::string windowName, const cv::Mat1b & image )
{
  if( !image.empty() )
  {
    cv::Mat result;

    cv::resize( image, result, image.size() * 2, cv::INTER_CUBIC );
    cv::normalize( result, result, 0, 255, CV_MINMAX );

    cv::imshow( windowName, result );
  }
}

void visualise( const std::string windowName, const cv::Mat3b & image )
{
  if( !image.empty() )
  {
    cv::Mat result;
    cv::resize( image, result, image.size() * 2, cv::INTER_CUBIC );
    cv::imshow( windowName, result );
  }
}

void visualise( const std::string windowName, const cv::Mat1w & image )
{
  if( !image.empty() )
  {
    cv::Mat result;

    cv::resize( image, result, image.size() * 2, cv::INTER_CUBIC );
    cv::normalize( result, result, 0, 65500, CV_MINMAX );

    cv::imshow( windowName, result );
  }
}

void visualise( const std::string windowName, const cv::Mat1d & image )
{
  if( !image.empty() )
  {
    cv::Mat result;

    cv::resize( image, result, image.size() * 2, cv::INTER_CUBIC );
    cv::normalize( result, result, 0, 1, CV_MINMAX );

    cv::imshow( windowName, result );
  }
}

void visualiseHistogram( const std::string & windowName, const cv::Mat1f & histogram )
{
  if( !histogram.empty() )
  {
    int size[] = {512, 1024};
    cv::Mat result = cv::Mat::zeros( 2, size, CV_8UC3 );
    double max = 0;

    cv::minMaxLoc( histogram, nullptr, &max );

    float scaleFactor = 400.0f / static_cast< float >( max );

    int binWidth = 1024 / histogram.rows;

    for( int i = 0; i < histogram.rows; i++ )
    {
      float value = histogram.at< float >( i ) * scaleFactor;

      cv::rectangle( result, cv::Point( i * binWidth, 512.0 ), cv::Point( (i+1) * binWidth, 512.0 - value ), 255, -1 );
    }

    cv::imshow( windowName, result );
  }
}

// modelNoise is a function for calculating the depth noise for a
// given distance according to the quadratic noise model presented in [1]
//
// @param depth : The depth measurement
// @param std   : The standard deviation of the object
//
// @returns     : The maximum of the either std or the calculated noise
//
// [1] M. Camplani, T. Mantecon, and L. Salgado. Depth-color fusion
// strategy for 3-D scene modeling with Kinect. Cybernetics, IEEE
// Transactions on, 43(6):1560�1571, 2013
double modelNoise( const double depth, const double std )
{
  const static double noiseModelVector[ 3 ] = { 2.3, 0.00055, 0.00000235 };

	return std::max< double >(
		2.5 * (
			noiseModelVector[ 0 ] +
			(noiseModelVector[ 1 ] * depth) +
			(noiseModelVector[ 2 ] * depth * depth)
		), std
	);
}
