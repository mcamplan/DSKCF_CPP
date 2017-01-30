
#include "ConcatenateFeatureChannelProcessor.h"
#include <numeric>

const std::vector< std::shared_ptr< FC > > ConcatenateFeatureChannelProcessor::concatenate(
    const std::vector< std::shared_ptr< FC > > & featureChannels ) const
{
#ifdef WIN32
  std::vector< std::shared_ptr< FC > > result( 1 );
  result[ 0 ] = featureChannels[ 0 ];
#else
  std::vector< std::shared_ptr< FC > > result = { featureChannels[ 0 ] };
#endif

  for( uint i = 1; i < featureChannels.size(); i++ )
  {
    result[ 0 ] = FC::concatFeatures( result[ 0 ], featureChannels[ i ] );
  }

  return result;
}

const std::vector< cv::Mat > ConcatenateFeatureChannelProcessor::concatenate(
    const std::vector< cv::Mat > & frame ) const
{
#ifdef WIN32
  std::vector< cv::Mat > result( 1 );

  result[ 0 ] = frame[ 1 ];

  return result;
#else
  return { frame[ 1 ] };
#endif
}

const double ConcatenateFeatureChannelProcessor::concatenate( const std::vector< double > & maxResponses ) const
{
  return std::accumulate( maxResponses.begin(), maxResponses.end(), 0.0 ) / static_cast< double >( maxResponses.size() );
}

const cv::Point_< double > ConcatenateFeatureChannelProcessor::concatenate( const std::vector< cv::Point_< double > > & positions ) const
{
  return positions[ 0 ];
}
