
#include "ColourFeatureChannelProcessor.h"

const std::vector< std::shared_ptr< FC > > ColourFeatureChannelProcessor::concatenate(
    const std::vector< std::shared_ptr< FC > > & featureChannels ) const
{
#ifdef WIN32
  std::vector< std::shared_ptr< FC > > result( 1 );

  result[ 0 ] = featureChannels[ 0 ];

  return result;
#else
  return { featureChannels[ 0 ] };
#endif
}

const std::vector< cv::Mat > ColourFeatureChannelProcessor::concatenate( const std::vector< cv::Mat > & frame ) const
{
#ifdef WIN32
	std::vector< cv::Mat > result( 1 );

	result[ 0 ] = frame[ 0 ];

	return result;
#else
  return { frame[ 0 ] };
#endif
}

const double ColourFeatureChannelProcessor::concatenate( const std::vector< double > & maxResponses ) const
{
  return maxResponses[ 0 ];
}

const cv::Point_< double > ColourFeatureChannelProcessor::concatenate( const std::vector< cv::Point_< double > > & positions ) const
{
  return positions[ 0 ];
}
