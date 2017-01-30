
#include "DepthWeightKCFTracker.h"
#include "math_helper.hpp"

DepthWeightKCFTracker::DepthWeightKCFTracker( KcfParameters paras, std::shared_ptr< Kernel > kernel ) : KcfTracker( paras, kernel )
{
  this->m_cellSize = paras.cellSize;
}

DepthWeightKCFTracker::~DepthWeightKCFTracker()
{
}

const DetectResult DepthWeightKCFTracker::detect( const cv::Mat & image, const std::shared_ptr< FC > & features,
                                                  const cv::Point_< double > & position, const double depth, const double std ) const
{
   //If this is a depth map, give the depth weighted max response.
  //Otherwise give the default kcf max response.
  if( image.type() == CV_16UC1 )
  {
    DetectResult result;
    Response newResponse = this->getResponse( image, features, cv::Point_< double >( position.x, position.y ) );
    std::vector< cv::MatIterator_< double > > pixels;

    for( cv::MatIterator_< double > itr = newResponse.response.begin(); itr != newResponse.response.end(); itr++ )
    {
      pixels.push_back( itr );
    }

    std::sort( pixels.begin(), pixels.end(),
               []( cv::MatIterator_< double > a, cv::MatIterator_< double > b ) -> bool
                {
                  return (*a) > (*b);
                }
    );

	double absoluteMax=*pixels[ 0 ];
    
    for( int i = 0; i < std::min< int >( 20, pixels.size() ); i++ )
    {
      cv::Point_< double > subDelta = subPixelDelta< double >( newResponse.response, pixels[ i ].pos() );

      if( subDelta.x > newResponse.response.cols / 2 )
      {
        subDelta.x -= newResponse.response.cols;
      }

      if( subDelta.y > newResponse.response.rows / 2 )
      {
        subDelta.y -= newResponse.response.rows;
      }

      cv::Point posImagePlane = position + ( this->m_cellSize * subDelta );
      posImagePlane.x = std::max( 0, std::min( image.cols - 1, posImagePlane.x ) );
      posImagePlane.y = std::max( 0, std::min( image.rows - 1, posImagePlane.y ) );
      double _depth = image.at< ushort >( posImagePlane );

      *pixels[ i ] *= weightDistanceLogisticOnDepth( depth, _depth, std );
    }

    auto itr = std::max_element( pixels.begin(), pixels.begin()+20, []( cv::MatIterator_< double > a, cv::MatIterator_< double > b ) -> bool { return (*a) < (*b); } );
    newResponse.maxResponse = **itr;
    newResponse.maxResponsePosition = itr->pos();
    cv::Point_< double > subDelta = subPixelDelta< double >( newResponse.response, newResponse.maxResponsePosition );

    if( subDelta.x > newResponse.response.cols / 2 )
    {
      subDelta.x -= newResponse.response.cols;
    }

    if( subDelta.y > newResponse.response.rows / 2 )
    {
      subDelta.y -= newResponse.response.rows;
    };


	result.maxResponse = absoluteMax;
	result.position = position + ( this->m_cellSize * subDelta );

    return result;
  }
  else
  {
    return KcfTracker::detect( image, features, position );
  }
}
