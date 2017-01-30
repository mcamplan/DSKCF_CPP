
#include "LinearFeatureChannelProcessor.h"
#include <numeric>

const std::vector< std::shared_ptr< FC > > LinearFeatureChannelProcessor::concatenate( const std::vector< std::shared_ptr< FC > > & featureChannels ) const
{
  return featureChannels;
}

const std::vector< cv::Mat > LinearFeatureChannelProcessor::concatenate( const std::vector< cv::Mat > & frame ) const
{
  return frame;
}

const double LinearFeatureChannelProcessor::concatenate( const std::vector< double > & maxResponses ) const
{
  return std::accumulate( maxResponses.begin(), maxResponses.end(), 0.0 ) / static_cast< double >( maxResponses.size() );
}

const cv::Point_< double > LinearFeatureChannelProcessor::concatenate( const std::vector< cv::Point_< double > > & positions ) const
{
  return std::accumulate( positions.begin(), positions.end(), cv::Point_< double >( 0, 0 ) ) * ( 1.0 / static_cast< double >( positions.size() ) );
}
