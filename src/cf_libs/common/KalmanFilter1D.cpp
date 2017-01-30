#include "KalmanFilter1D.h"

KalmanFilter1D::KalmanFilter1D()
{
}

KalmanFilter1D::~KalmanFilter1D()
{

}

void KalmanFilter1D::initialise( const double position )
{
  this->m_filter.init( 2, 1, 0, CV_64F );
  this->m_filter.transitionMatrix = (cv::Mat_< double >( 2, 2 ) << 1,1,  0,1);
  this->m_filter.statePre.at< double >( 0 ) = position;
  this->m_filter.statePre.at< double >( 1 ) = 0;
  this->m_filter.statePost.at< double >( 0 ) = position;
  this->m_filter.statePost.at< double >( 1 ) = 0;
  cv::setIdentity( this->m_filter.measurementMatrix );
  cv::setIdentity( this->m_filter.processNoiseCov, cv::Scalar::all( 1e-4 ) );
  cv::setIdentity( this->m_filter.measurementNoiseCov, cv::Scalar::all( 1e-4 ) );
  cv::setIdentity( this->m_filter.errorCovPost, cv::Scalar::all( 0.1 ) );
}

const double KalmanFilter1D::getPrediction()
{
  return this->m_filter.predict().at< double >( 0 );
}

const double KalmanFilter1D::getEstimate( const double & measurement )
{
  cv::Mat1d measurementMatrix( 1, 1 );
  measurementMatrix( 0 ) = measurement;

  return this->m_filter.correct( measurementMatrix ).at< double >( 0 );
}
