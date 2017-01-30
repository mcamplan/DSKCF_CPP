#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <tbb/parallel_for.h>
#include "DepthHistogram.h"
#include "math_helper.hpp"

DepthHistogram::DepthHistogram()
{
  this->m_minimum = 0;
  this->m_maximum = 0;
}

const std::vector< int > DepthHistogram::getPeaks() const
{
  int start = 0;
  int end = this->size() - 1;
  double max = 0.0;

  if( this->size() > 50 )
  {
    cv::minMaxLoc( this->m_bins, nullptr, &max );

    for( int i = 0; i < this->size(); i++ )
    {
      if( this->m_bins( i ) > max * 0.005 )//ddddd
      {
        start = std::min( start, i );
        end = std::max( end, i );
      }
    }
  }

  return this->getPeaks( ( end - start > 50 ? 3 : 1 ) );
}

const std::vector< int > DepthHistogram::getPeaks( const int minimumPeakdistance, const double minimumPeakHeight ) const
{
  if( !this->m_bins.empty() )
  {
    double max = 0.0;
    double current_max = 0.0;
    cv::Point current_max_location;
    cv::Mat1f local_histogram;
    this->m_bins.copyTo( local_histogram );
    std::vector< int > candidates;

    cv::minMaxLoc( this->m_bins, nullptr, &max );
    cv::minMaxLoc( local_histogram, nullptr, &current_max, nullptr, &current_max_location );

    //Check if the first bin is a peak (special case, no left neighbour)
    if( ( this->m_bins( 0 ) > this->m_bins( 1 ) ) && ( this->m_bins( 0 ) > minimumPeakHeight * max ) )
    {
      candidates.push_back( 0 );
    }

    //Check if the last bin is a peak (special case, no right neighbour)
    if( ( this->m_bins( this->m_bins.rows - 1 ) > this->m_bins( this->m_bins.rows - 2 ) ) && ( this->m_bins( this->m_bins.rows - 1 ) > minimumPeakHeight * max ) )
    {
      candidates.push_back( this->m_bins.rows - 1 );
    }

    //Check the rest of the bins to see if they are a peak
    for( int i = 1; i < this->m_bins.rows - 1; i++ )
    {
      float left = this->m_bins( i - 1 );
      float right = this->m_bins( i + 1 );
      float value = this->m_bins( i );

      if( ( value > left ) && ( value > right ) && ( value > minimumPeakHeight * max ) )
      {
        candidates.push_back( i );
      }
    }

    //Filter out the neighbouring peaks that are closer than the minimum peak distance
    if( minimumPeakdistance > 1 )
    {
      while( current_max > minimumPeakHeight * max )
      {
        auto itr = std::find( candidates.begin(), candidates.end(), current_max_location.y );

        if( itr != candidates.end() )
        {
          int index = *itr;

          for( auto itr2 = candidates.begin(); itr2 != candidates.end(); )
          {
            if( ( *itr2 >= index - minimumPeakdistance ) && ( *itr2 <= index + minimumPeakdistance ) &&
                                                            ( *itr2 != index ) )
            {
              itr2 = candidates.erase( itr2 );
            }
            else
            {
              itr2++;
            }
          }
        }

        local_histogram( current_max_location.y ) = 0;

        cv::minMaxLoc( local_histogram, nullptr, &current_max, nullptr, &current_max_location );
      }
    }

    //Sort the candidates so that the nearest is at candiates[ 0 ]
    std::sort( candidates.begin(), candidates.end(), std::less< int >() );

    return candidates;
  }

  return std::vector< int >();
}

const DepthHistogram::Labels DepthHistogram::getLabels( const std::vector< int > & peaks ) const
{
  std::vector< float > centroids( peaks.size() );
  for( uint i = 0; i < peaks.size(); i++ )
  {
    centroids[ i ] = static_cast< float >( peaks[ i ] );
  }

  return this->kmeans( centroids );
}

const DepthHistogram::Labels DepthHistogram::kmeans( const std::vector< float > & centroids ) const
{
  float dC = 1000.0f;
  DepthHistogram::Labels result;
  result.centers = centroids;
  result.labelsC.assign(centroids.size(),0);
  result.labels.resize( this->m_bins.rows );

  while( dC > 1.0f )
  {
    std::vector< float > oldCentroids = result.centers;

    //Assign each label to the nearest centroid
    //TODO: Investigate possible parallelisation
    //for( int i = 0; i < this->m_bins.rows; i++ )
    tbb::parallel_for< uint >( 0, this->m_bins.rows, 1,
    [&result]( const uint i )
    {
      for( uint j = 0; j < result.centers.size(); j++ )
      {
        if( std::abs( result.centers[ j ] - i ) < std::abs( result.centers[ result.labels[ i ] ] - i ) )
        {
          result.labels[ i ] = static_cast< int >( j );
        }
      }
    });

    //Move the centroids to the center of their labels
    //TODO: Investigate possible parallelisation
    //for( uint i = 0; i < result.centers.size(); i++ )
    tbb::parallel_for< uint >( 0, result.centers.size(), 1,
    [this,&result,&oldCentroids]( const uint i )
    {
      float numerator = 0.0;
      float denominator = 0.0;

      for( int j = 0; j < this->m_bins.rows; j++ )
      {
        if( static_cast< uint >( result.labels[ j ] == i ) )
        {
          
          numerator += j * this->m_bins( j );
          denominator += this->m_bins( j );
        }
      }

      result.centers[ i ] = numerator / denominator;

      //Sanity check to ensure that we have real numbers
#ifdef _WIN32
			if( !_finite( result.centers[ i ] ) )
#elif __linux
      if( !std::isfinite( result.centers[ i ] ) )
#endif
      {
        result.centers[ i ] = oldCentroids[ i ];
      }
    });

    //Find the maximum that we moved the centroids
    dC = 0.0;
    for( uint i = 0; i < result.centers.size(); i++ )
    {
      dC = std::max( dC, std::abs( oldCentroids[ i ] - result.centers[ i ] ) );
    }
  }

    //fill the label Center Vector
  for(int i=0; i < result.labelsC.size();i++)
	result.labelsC[i]=i+1;

  return result;
}

const int DepthHistogram::depthToBin( const double depth ) const
{
	float stepH=this->estStep()/2;
	double histDepth = this->m_bins.rows * ( ( depth - this->m_minimum -stepH) / ( this->m_maximum - this->m_minimum ) );

  return std::max( 0, std::min( this->m_bins.rows - 1, cvRound( histDepth ) ) );
}

const double DepthHistogram::binToDepth( const float bin ) const
{

	float stepH=this->estStep()/2;
	return (( bin*	estStep() ) + this->m_minimum + stepH);
}

const int DepthHistogram::depthToLabel( const double depth, const std::vector< int > & labels ) const
{
  int bin = this->depthToBin( depth );

  return labels[ bin ];
}

const int DepthHistogram::depthToPeak( const double depth, const std::vector< int > & peaks ) const
{
  //double histDepth = this->m_bins.rows * ( ( depth - this->m_minimum ) / ( this->m_maximum - this->m_minimum ) );
  double histDepth=this->depthToBin(depth);
  std::vector< double > peakTranslated( peaks.size() );

  for( uint i = 0; i < peaks.size(); i++ )
  {
    peakTranslated[ i ] = std::abs( peaks[ i ] - histDepth );
  }

  return std::min_element( peakTranslated.begin(), peakTranslated.end() ) - peakTranslated.begin();
}

const bool DepthHistogram::empty() const
{
  return this->m_bins.empty();
}

const size_t DepthHistogram::size() const
{
  return this->m_bins.rows;
}

const double DepthHistogram::minimum() const
{
  return this->m_minimum;
}

const float DepthHistogram::estStep() const
{
  return this->estimatedStep;
}


const double DepthHistogram::maximum() const
{
  return this->m_maximum;
}

const float DepthHistogram::operator[]( const uint i ) const
{
  return this->m_bins( i );
}

const DepthHistogram DepthHistogram::createHistogram( const uint step, const cv::Mat & region, const cv::Mat1b & mask )
{
  cv::Mat1f region32f;
  DepthHistogram result;
  int histogramBinCount = 0;

  region.convertTo( region32f, CV_32F );
  cv::minMaxLoc( region32f, &result.m_minimum, &result.m_maximum, nullptr, nullptr, mask );



  if( step == 0 )
  {
	
    histogramBinCount = std::max( 1, cvRound( ( result.m_maximum - result.m_minimum ) / 50.0 )+1 );//modified here
	result.m_minimum-=25;
	result.m_maximum += 25;
  }
  else
  {
    histogramBinCount = std::max( 1, cvRound( ( result.m_maximum - result.m_minimum ) / static_cast< double >( step ) ) +1 );//modified here
	result.m_minimum-=static_cast< double >( step )/2;
	result.m_maximum += static_cast< double >( step )/2;
  }

  //change this bit with incrementing the number of bin
  //result.m_maximum += modelNoise( result.m_maximum, 0 );
  histogramBinCount++;


  int channels = 0;
  float hist_range[] = { static_cast< float >( result.m_minimum ), static_cast< float >( result.m_maximum ) };
  const float * hist_ranges[] = { hist_range };

  cv::calcHist( &region32f, 1, &channels, cv::Mat(), result.m_bins, 1, &histogramBinCount, hist_ranges );

  result.estimatedStep=( 1.0 / (float)result.size() ) * ( result.m_maximum - result.m_minimum );

  return result;
}

void DepthHistogram::visualise( const std::string & string )
{
  visualiseHistogram( string, this->m_bins );
}

const int DepthHistogram::depthToCentroid( const double depth, const std::vector< float > & centroids ) const
{
  //double histDepth = this->m_bins.rows * ( ( depth - this->m_minimum ) / ( this->m_maximum - this->m_minimum ) );
  double histDepth=this->depthToBin(depth);
  std::vector< double > peakTranslated( centroids.size() );

  for( uint i = 0; i < centroids.size(); i++ )
  {
    peakTranslated[ i ] = std::abs( centroids[ i ] - histDepth );
  }

  return std::min_element( peakTranslated.begin(), peakTranslated.end() ) - peakTranslated.begin();
}
