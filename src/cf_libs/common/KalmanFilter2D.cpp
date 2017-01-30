#include <opencv2/core.hpp>
#include "KalmanFilter2D.h"

KalmanFilter2D::KalmanFilter2D()
{
}

KalmanFilter2D::~KalmanFilter2D()
{
}

void KalmanFilter2D::initialise( const cv::Point_< double > & position )
{
  this->m_filter.init( 4, 2, 0, CV_64F );
  this->m_filter.transitionMatrix = (cv::Mat_< double >( 4, 4 ) << 1,0,1,0,  0,1,0,1,  0,0,1,0,  0,0,0,1);
  this->m_filter.statePre.at< double >( 0 ) = position.x;
  this->m_filter.statePre.at< double >( 1 ) = position.y;
  this->m_filter.statePre.at< double >( 2 ) = 0;
  this->m_filter.statePre.at< double >( 3 ) = 0;
  this->m_filter.statePost.at< double >( 0 ) = position.x;
  this->m_filter.statePost.at< double >( 1 ) = position.y;
  this->m_filter.statePost.at< double >( 2 ) = 0;
  this->m_filter.statePost.at< double >( 3 ) = 0;
  cv::setIdentity( this->m_filter.measurementMatrix );
  cv::setIdentity( this->m_filter.processNoiseCov, cv::Scalar::all( 1e-4 ) );
  cv::setIdentity( this->m_filter.measurementNoiseCov, cv::Scalar::all( 1e-4 ) );
  cv::setIdentity( this->m_filter.errorCovPost, cv::Scalar::all( 0.1 ) );
}

const cv::Point_< double > KalmanFilter2D::getPrediction()
{
  cv::Point_< double > result;
  cv::Mat1d prediction = this->m_filter.predict();

  result.x = prediction( 0 ); result.y = prediction( 1 );

  return result;
}

const cv::Point_< double > KalmanFilter2D::getEstimate( const cv::Point_< double > & measurement )
{
  cv::Point_< double > result;
  cv::Mat1d measurementMatrix( 2, 1 );

  measurementMatrix( 0 ) = measurement.x; measurementMatrix( 1 ) = measurement.y;

  cv::Mat1d estimate = this->m_filter.correct( measurementMatrix );

  result.x = estimate( 0 ); result.y = estimate( 1 );

  return result;
}