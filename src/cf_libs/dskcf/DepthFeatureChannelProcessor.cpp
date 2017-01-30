#include "DepthFeatureChannelProcessor.h"

const std::vector< std::shared_ptr< FC > > DepthFeatureChannelProcessor::concatenate(
    const std::vector< std::shared_ptr< FC > > & featureChannels ) const
{
#ifdef WIN32
  std::vector< std::shared_ptr< FC > > result( 1 );

  result[ 0 ] = featureChannels[ 1 ];

  return result;
#else
  return { featureChannels[ 1 ] };
#endif
}

const std::vector< cv::Mat > DepthFeatureChannelProcessor::concatenate( const std::vector< cv::Mat > & frame ) const
{
#ifdef WIN32
  std::vector< cv::Mat > result( 1 );

  result[ 0 ] = frame[ 1 ];

  return result;
#else
  return { frame[ 1 ] };
#endif
}

const double DepthFeatureChannelProcessor::concatenate( const std::vector< double > & maxResponses ) const
{
  return maxResponses[ 0 ];
}

const cv::Point_< double > DepthFeatureChannelProcessor::concatenate(
    const std::vector< cv::Point_< double > > & positions ) const
{
  return positions[ 0 ];
}
