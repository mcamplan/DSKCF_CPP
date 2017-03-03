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
// Original file: // Original file: https://github.com/klahaag/cf_tracking/blob/master/src/cf_libs/common/math_helper.hpp
// + Authors: Jake Hall, Massimo Camplan, Sion Hannuna
// * We modified the original code of  Klaus Haag, adding functionality to deal with DS-KCF system, especially
//   for depth menaging
*/

#ifndef HELPER_H_
#define HELPER_H_

#include <iostream>
#include <queue>

#include "opencv2/core/utility.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/core/ocl.hpp"

#include <cmath>
#include <string>

#include "cv_ext.hpp"
#include "mat_consts.hpp"
#include "shift.hpp"

int mod(int dividend, int divisor);
cv::Mat fftshift( const cv::Mat & a );

/** ifftshift implements the MATLAB function ifftshift.
 * This is achieved by switching the diagonal quadrants
 *
 * @param a : The matrix to be un-shifted
 *
 * @returns : The un-shifted matrix
 */
cv::Mat ifftshift( const cv::Mat & a );
cv::Mat linSpace( double x1, double x2, int a );
double weightDistanceLogisticOnDepth( double targetDepth, double candidateDepth, double targetSTD );
double sigmFunction( double x, double A, double K, double Q, double ni, double B, double M );
void visualiseFourier( const std::string & windowName, const cv::Mat & image );
void visualiseHistogram( const std::string & windowName, const cv::Mat1f & histogram );
void visualise( const std::string windowName, const cv::Mat1b & image );
void visualise( const std::string windowName, const cv::Mat3b & image );
void visualise( const std::string windowName, const cv::Mat1w & image );
void visualise( const std::string windowName, const cv::Mat1d & image );
double modelNoise( const double depth, const double std );

template< class T, class U >
const cv::Point_< T > pointCast( const cv::Point_< U > & a )
{
  return cv::Point_< T >(
    static_cast< T >( a.x ), static_cast< T >( a.y )
  );
};

template< class T >
const cv::Point3_< T > to3D( const cv::Point_< T > & a, const T z )
{
  return cv::Point3_< T >( a.x, a.y, z );
}

template< class T >
cv::Rect_< T > extremeRect( const cv::Rect_< T > & a, const cv::Rect_< T > & b )
{
	cv::Rect_< T > result;
	cv::Point_< T > br[ 2 ] = { a.br(), b.br() };
	cv::Point_< T > br_;
	br_.x = std::max< T >( br[ 0 ].x, br[ 1 ].x );
	br_.y = std::max< T >( br[ 0 ].y, br[ 1 ].y );

	result.x = std::min< T >( a.x, b.x );
	result.y = std::min< T >( a.y, b.y );
	result.width = br_.x - result.x;
	result.height = br_.y - result.y;

	return result;
}

template< class T >
cv::Point_< T > centerPoint( const cv::Rect_< T > & rect )
{
	cv::Point_< T > result;

	result.x = rect.x + ( rect.width / 2 );
	result.y = rect.y + ( rect.height / 2 );

	return result;
}

template< class T >
cv::Point_< T > pointFloor( const cv::Point_< T > & point )
{
  cv::Point_< T > result;

  result.x = floor( point.x );
  result.y = floor( point.y );

  return result;
}

template< class T >
const cv::Mat1b getRegion( const cv::Mat_< T > & region, const T low, const T high )
{
  cv::Mat1b result = cv::Mat1b::zeros( region.rows, region.cols );

  for( int row = 0; row < region.rows; row++ )
  {
    for( int col = 0; col < region.cols; col++ )
    {
      if( ( region( row, col ) > low ) && ( region( row, col ) < high ) )
      {
        result( row, col ) = 1;
      }
    }
  }

  return result;
}

template< class T >
cv::Rect_< T > boundingBoxFromPointSize( const cv::Point_< T > & center,
  const cv::Size_< T > & size )
{
	cv::Rect_< double > result;

	result.x = center.x - ( size.width / 2.0 );
	result.y = center.y - ( size.height / 2.0 );
	result.width = size.width;
	result.height = size.height;

	return result;
}

template< class T >
cv::Rect_< T > resizeBoundingBox( const cv::Rect_< T > & input,
  const cv::Size_< T > & newSize )
{
  return boundingBoxFromPointSize< T >( centerPoint< T >( input ), newSize );
}

template< class T >
cv::Rect_< int > rectRound( cv::Rect_< T > rect )
{
  return cv::Rect_< int >(
    cvRound( rect.x ), cvRound( rect.y ),
    cvRound( rect.width ), cvRound( rect.height )
  );
}

template< class T >
cv::Rect_< int > rectCeil( cv::Rect_< T > rect )
{
  return cv::Rect_< int >(
      cvRound( rect.x ), cvRound( rect.y ),
      cvRound( rect.width ), cvRound( rect.height )
  );
}

template< class T >
cv::Rect_< int > rectFloor( cv::Rect_< T > rect )
{
  return cv::Rect_< int >(
      cvFloor( rect.x ), cvFloor( rect.y ),
      cvFloor( rect.width ), cvFloor( rect.height )
  );
}

template< class T >
cv::Rect_< int > getSubWindowRounding( cv::Rect_< T > rect )
{
	cv::Size windowSize = rect.size();
	cv::Point windowPosition = centerPoint( rect );

	int width = static_cast<int>(windowSize.width);
	int height = static_cast<int>(windowSize.height);

	int xs = static_cast<int>(std::floor(windowPosition.x) - std::floor(width / 2.0)) + 1;
	int ys = static_cast<int>(std::floor(windowPosition.y) - std::floor(height / 2.0)) + 1;

	return cv::Rect_< int >(xs, ys, width, height);
}

template< class T >
cv::Size_< T > sizeRound( const cv::Size_< T > & rect )
{
  return cv::Size_< T >( cvRound( rect.width ), cvRound( rect.height ) );
}

template< class T >
cv::Point_< T > pointRound( const cv::Point_< T > & rect )
{
  return cv::Point_< T >( cvRound( rect.x ), cvRound( rect.y ) );
}

template<typename T>
cv::Size_<T> sizeFloor(cv::Size_<T> size)
{
  return cv::Size_<T>(floor(size.width), floor(size.height));
}

template <typename T>
cv::Mat numberToRowVector(int n)
{
  cv::Mat_<T> rowVec(n, 1);

  for (int i = 0; i < n; ++i)
  {
    rowVec.template at<T>(i, 0) = static_cast<T>(i + 1);
  }

  return rowVec;
}

template <typename T>
cv::Mat numberToColVector(int n)
{
  cv::Mat_<T> colVec(1, n);

  for (int i = 0; i < n; ++i)
  {
    colVec.template at<T>(0, i) = static_cast<T>(i + 1);
  }

  return colVec;
}

// http://home.isr.uc.pt/~henriques/circulant/
template <typename T>
T subPixelPeak(T* p)
{
  T delta = mat_consts::constants<T>::c0_5 * (p[2] - p[0]) / (2 * p[1] - p[2] - p[0]);
#ifdef _WIN32
	if( !_finite( delta ) )
	{
		return 0;
	}
#elif __linux
  if( !std::isfinite( delta ) )
  {
    return 0;
  }
#endif
  return delta;
}

// http://home.isr.uc.pt/~henriques/circulant/
template <typename T>
cv::Point_<T> subPixelDelta(const cv::Mat& response, const cv::Point2i& delta)
{
  cv::Point_<T> subDelta(static_cast<float>(delta.x), static_cast<float>(delta.y));
  T vNeighbors[3] = {};
  T hNeighbors[3] = {};

  for (int i = -1; i < 2; ++i)
  {
    vNeighbors[i + 1] = response.template at<T>(mod(delta.y + i, response.rows), delta.x);
    hNeighbors[i + 1] = response.template at<T>(delta.y, mod(delta.x + i, response.cols));
  }

  subDelta.y += subPixelPeak(vNeighbors);
  subDelta.x += subPixelPeak(hNeighbors);

  return subDelta;
}

// http://home.isr.uc.pt/~henriques/circulant/
template <typename T>
cv::Mat gaussianShapedLabels2D(T sigma, const cv::Size_<T>& size)
{
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);

  cv::Mat_<T> rs(height, width);

  CV_Assert(rs.isContinuous());

  T lowerBoundX = static_cast<T>(-floor(width * 0.5) + 1);
  T lowerBoundY = static_cast<T>(-floor(height * 0.5) + 1);

  T* colValues = new T[width];
  T* rsd = rs.template ptr<T>(0, 0);
  T rowValue = 0;
  T sigmaMult = static_cast<T>(-0.5 / (sigma*sigma));

  for (int i = 0; i < width; ++i)
  {
    colValues[i] = (i + lowerBoundX) * (i + lowerBoundX);
  }

  for (int row = 0; row < height; ++row)
  {
    rowValue = (row + lowerBoundY) * (row + lowerBoundY);

    for (int col = 0; col < width; ++col)
    {
      rsd[row*width + col] = exp((colValues[col] + rowValue) * sigmaMult);
    }
  }

  delete[] colValues;

  return rs;
}

// http://home.isr.uc.pt/~henriques/circulant/
template <typename T>
cv::Mat gaussianShapedLabelsShifted2D(T sigma, const cv::Size_<T>& size)
{
  cv::Mat y = gaussianShapedLabels2D(sigma, size);
  cv::Point2f delta(
    static_cast<float>(1 - floor(size.width * 0.5)),
    static_cast<float>(1 - floor(size.height * 0.5))
  );

  shift(y, y, delta, cv::BORDER_WRAP);

  CV_Assert(y.at<T>(0, 0) == 1.0);
  return y;
}

template <typename BT, typename ET>
cv::Mat pow(BT base_, const cv::Mat_<ET>& exponent)
{
  cv::Mat dst = cv::Mat(exponent.rows, exponent.cols, exponent.type());
  int widthChannels = exponent.cols * exponent.channels();
  int height = exponent.rows;

  // http://docs.opencv.org/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#the-efficient-way
  if (exponent.isContinuous())
  {
    widthChannels *= height;
    height = 1;
  }

  int row = 0, col = 0;
  const ET* exponentd = 0;
  ET* dstd = 0;

  for (row = 0; row < height; ++row)
  {
    exponentd = exponent.template ptr<ET>(row);
    dstd = dst.template ptr<ET>(row);

    for (col = 0; col < widthChannels; ++col)
    {
      dstd[col] = std::pow(base_, exponentd[col]);
    }
  }

  return dst;
}

// http://en.wikipedia.org/wiki/Hann_function
template<typename T>
cv::Mat hanningWindow(int n)
{
  CV_Assert( n > 0 );
  cv::Mat_<T> w = cv::Mat_<T>( n, 1 );

  if ( n == 1 )
  {
    w.template at< T >( 0, 0 ) = 1;
    return w;
  }
  else
  {
    for ( int i = 0; i < n; ++i )
    {
      w.template at<T>( i, 0 ) = static_cast< T >(
        0.5 * ( 1.0 - cos( 2.0 * 3.14159265358979323846 * i / ( n - 1 ) ) )
      );
    }

    return w;
  }
}

template <typename T>
void divideSpectrumsNoCcs(const cv::Mat& numerator, const cv::Mat& denominator, cv::Mat& dst)
{
  // http://mathworld.wolfram.com/ComplexDivision.html
  // (a,b) / (c,d) = ((ac+bd)/v , (bc-ad)/v)
  // with v = (c^2 + d^2)
  // Performance wise implemented according to
  // http://docs.opencv.org/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv
  // TODO: this is still very slow => vectorize (note that mulSpectrums is not vectorized either...)

  int type = numerator.type();
  int channels = numerator.channels();

  CV_Assert(
    type == denominator.type()&& numerator.size() == denominator.size()&&
    channels == denominator.channels() && channels == 2
  );
  CV_Assert(
    type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2
  );

  dst = cv::Mat(numerator.rows, numerator.cols, type);
  int widthChannels = numerator.cols * channels;
  int height = numerator.rows;

  if (numerator.isContinuous() && denominator.isContinuous())
  {
    widthChannels *= height;
    height = 1;
  }

  int row = 0, col = 0;
  const T* numd, *denomd;
  T* dstd;
  T a, b, c, d, v;

  for (row = 0; row < height; ++row)
  {
    numd = numerator.ptr<T>(row);
    denomd = denominator.ptr<T>(row);
    dstd = dst.ptr<T>(row);

    for (col = 0; col < widthChannels; col += 2)
    {
      a = numd[col];          // real part
      b = numd[col + 1];      // imag part
      c = denomd[col];       // real part
      d = denomd[col + 1];   // imag part

      v = (c * c) + (d * d);

      dstd[col] = (a * c + b * d) / v;
      dstd[col + 1] = (b * c - a * d) / v;
    }
  }
}

// http://home.isr.uc.pt/~henriques/circulant/
template<typename T>
bool getSubWindow(const cv::Mat& image, cv::Mat& patch, const cv::Size_<T>& size,
    const cv::Point_<T>& pos, cv::Point_<T>* posInSubWindow = 0)
{
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);

  int xs = static_cast<int>(std::floor(pos.x) - std::floor(width / 2.0)) + 1;
  int ys = static_cast<int>(std::floor(pos.y) - std::floor(height / 2.0)) + 1;
  T posInSubWindowX = pos.x - xs;
  T posInSubWindowY = pos.y - ys;

  int diffTopX = -xs;
  int diffTopY = -ys;
  int diffBottomX = image.cols - xs - width;
  int diffBottomY = image.rows - ys - height;

  cv::Rect imageRect(0, 0, image.cols, image.rows);
  cv::Rect subRect(xs, ys, width, height);
  subRect &= imageRect;
  cv::Mat subWindow = image(subRect);

  if (subWindow.cols == 0 || subWindow.rows == 0)
  {
    return false;
  }

  if (diffTopX > 0 || diffTopY > 0 || diffBottomX < 0 || diffBottomY < 0)
  {
    diffTopX = std::max(0, diffTopX);
    diffTopY = std::max(0, diffTopY);
    diffBottomX = std::min(0, diffBottomX);
    diffBottomY = std::min(0, diffBottomY);

    copyMakeBorder(subWindow, subWindow, diffTopY, -diffBottomY,
        diffTopX, -diffBottomX, cv::BORDER_REPLICATE);
  }

  // this if can be true if the sub window
  // is completely outside the image
  if (width != subWindow.cols || height != subWindow.rows)
      return false;

  if (posInSubWindow != 0)
  {
      posInSubWindow->x = posInSubWindowX;
      posInSubWindow->y = posInSubWindowY;
  }

  patch = subWindow;

  return true;
}

template< class T >
void labelComponents(
  const cv::Mat_< T > & input, cv::Mat1b & result, unsigned int x,
  unsigned int y, unsigned char label, cv::Rect_< int > & boundingBox )
{
  //Expand the bounding box if necessary
  boundingBox.x = std::min< int >( x, boundingBox.x );
  boundingBox.y = std::min< int >( y, boundingBox.y );
  boundingBox.width = std::max< int >( x, boundingBox.width );
  boundingBox.height = std::max< int >( y, boundingBox.height );

  std::queue< cv::Point_< int > > candidates;
  candidates.push( cv::Point_< int >( x, y ) );

  while( candidates.size() > 0 )
  {
    cv::Point_< int > candidate = candidates.front();
    candidates.pop();

    if( ( result( candidate ) == 0 ) )
    {
      result( candidate ) = label;

      for( int _x = -1; _x <= 1; _x++ )
      {
        for( int _y = -1; _y <= 1; _y++ )
        {
          cv::Point_< int > newCandidate = candidate;
          if( ( _x != 0 ) || ( _y != 0 ) )
          {
            newCandidate.x = std::min( input.cols - 1, std::max( 0, newCandidate.x + _x ) );
            newCandidate.y = std::min( input.rows - 1, std::max( 0, newCandidate.y + _y ) );
          }

          if(
            ( input( newCandidate ) == input( candidate ) )&&
            ( input( newCandidate ) == input( y, x ) )&&
            ( result( newCandidate ) == 0 )
          )
          {
            boundingBox.x = std::min< int >( newCandidate.x, boundingBox.x );
            boundingBox.y = std::min< int >( newCandidate.y, boundingBox.y );
            boundingBox.width = std::max< int >( newCandidate.x, boundingBox.width );
            boundingBox.height = std::max< int >( newCandidate.y, boundingBox.height );

            candidates.push( newCandidate );
          }
          else
          {
            result( newCandidate ) = 2;
          }
        }
      }
    }
  }
}

template< class T >
cv::Rect_< int > floodFill( const cv::Mat_< T > & input, const cv::Point_< int > start )
{
  cv::Mat1b tmp = cv::Mat1b::zeros( input.rows, input.cols );
  cv::Rect_< int > result( start.x, start.y, start.x, start.y );

  labelComponents( input, tmp, start.x, start.y, 1, result );

  result.width -= result.x;
  result.height -= result.y;
  return result;
}

template< class T >
std::vector< cv::Rect_< int > > connectedComponents( const cv::Mat_< T > & input )
{
  //Initial label set to one
  unsigned char label = 1;

  //Fill the result image with zeros
  cv::Mat1b result = cv::Mat1b::zeros( input.rows, input.cols );

  //Bounding boxes of each connected component
  std::vector< cv::Rect_< int > > boundingBoxes;

  //For each pixel in the input image
  for( int x = 0; x < result.cols; x++ )
  {
    for( int y = 0; y < result.rows; y++ )
    {
      if( label != 0 )
      {
        //If the pixel hasn't been labelled
        if( result( y, x ) == 0 )
        {
          //Set the bounding box to be at least the current pixel
          cv::Rect_< int > boundingBox( x, y, x, y );

          //Recursively fill the pixel's neighbours
          //Increment the label
          labelComponents< T >( input, result, x, y, label++, boundingBox );

          //Correct the bounding box to be (x,y,width,height)
          boundingBox.width -= boundingBox.x;
          boundingBox.height -= boundingBox.y;

          //Add this components bounding box to the list of bounding boxes
          boundingBoxes.push_back( boundingBox );
        }
      }
      else
      {
        std::cerr << "ERROR : connectedComponents : label is zero!" << std::endl;
      }
    }
  }

  return boundingBoxes;
}

template< class T >
std::vector< cv::Rect_< int > > connectedComponents( const cv::Mat_< T > & input, cv::Mat1b & result )
{
  //Initial label set to one
  unsigned char label = 1;

  //Fill the result image with zeros
  result.setTo(0);

  //Bounding boxes of each connected component
  std::vector< cv::Rect_< int > > boundingBoxes;

  //For each pixel in the input image
  for( int x = 0; x < result.cols; x++ )
  {
    for( int y = 0; y < result.rows; y++ )
    {
      if( label != 0 )
      {
        //If the pixel hasn't been labelled
        if( result( y, x ) == 0 )
        {
          //Set the bounding box to be at least the current pixel
          cv::Rect_< int > boundingBox( x, y, x, y );

          //Recursively fill the pixel's neighbours
          //Increment the label
          labelComponents< T >( input, result, x, y, label++, boundingBox );

          //Correct the bounding box to be (x,y,width,height)
          boundingBox.width -= boundingBox.x;
          boundingBox.height -= boundingBox.y;

          //Add this components bounding box to the list of bounding boxes
          boundingBoxes.push_back( boundingBox );
        }
      }
      else
      {
        std::cerr << "ERROR : connectedComponents : label is zero!" << std::endl;
      }
    }
  }

  return boundingBoxes;
}


template< class T >
cv::Mat_< uchar > createMask( const cv::Mat_< T > & region, const T value = 0, bool maskOut = true )
{
	cv::Mat_< uchar > mask( region.rows, region.cols );

	for( int x = 0; x < region.cols; x++ )
	{
		for( int y = 0; y < region.rows; y++ )
		{
      if( ( region( y, x ) == value ) == maskOut )
      {
        mask( y, x ) = 0;
      }
      else
      {
        mask( y, x ) = 1;
      }
		}
	}

	return mask;
}

template< class T >
cv::Rect_< int > componentBoundingBox( const cv::Mat_< T > & image )
{
  bool initialised = false;
  cv::Rect_< int > result;

  for( int row = 0; row < image.rows; row++ )
  {
    for( int col = 0; col < image.cols; col++ )
    {
      if( image( row, col ) != 0 )
      {
        if( initialised )
        {
          result.x = std::min( result.x, col );
          result.y = std::min( result.y, row );
          result.width = std::max( result.width, col );
          result.height = std::max( result.height, row );
        }
        else
        {
          result.x = col; result.y = row;
          result.width = col; result.height = row;
        }
      }
    }
  }

  result.width -= result.x;
  result.height -= result.y;

  return result;
}

template< class T >
cv::Point_< int > findPixel( const cv::Mat_< T > & image, T value )
{
  for( int col = 0; col < image.cols; col++ )
  {
    for( int row = 0; row < image.rows; row++ )
    {
      if( image( row, col ) == value )
      {
        return cv::Point_< int >( col, row );
      }
    }
  }

  return cv::Point_< int >( -1, -1 );
}

template< class T, class U >
cv::Rect_< T > rectCast( const cv::Rect_< U > & input )
{
  cv::Rect_< T > result;

  result.x = static_cast< T >( input.x );
  result.y = static_cast< T >( input.y );
  result.width = static_cast< T >( input.width );
  result.height = static_cast< T >( input.height );

  return result;
}

#endif
