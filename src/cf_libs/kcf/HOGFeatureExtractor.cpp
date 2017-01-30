#include <memory>

#include "HOGFeatureExtractor.hpp"
#include "gradientMex.hpp"

HOGFeatureExtractor::HOGFeatureExtractor()
{
	this->m_cellSize = 4;
}

HOGFeatureExtractor::~HOGFeatureExtractor()
{
}

std::shared_ptr< FC > HOGFeatureExtractor::getFeatures( const cv::Mat & image,
																												const cv::Rect_< double > & boundingBox ) const
{
	cv::Mat patch;

	if( getSubWindow< double >( image, patch, boundingBox.size(), centerPoint( boundingBox ) ) )
	{
	  cv::Mat patchResizedFloat;
	  patch.convertTo(patchResizedFloat, CV_32FC(3));

	  auto features = std::make_shared< FC >();
	  piotr::cvFhog< double, FC >(patchResizedFloat, features, this->m_cellSize);

		return features;
	}
	else
	{
		std::cerr << "Error : HOGFeatureExtractor::getFeatures : getSubWindow failed!" << std::endl;
	}

	return nullptr;
}
