#include "dskcf_tracker.hpp"

#include "GaussianKernel.hpp"
#include "HOGFeatureExtractor.hpp"
#include "ConcatenateFeatureChannelProcessor.h"

typedef cv::Rect_< double > Rect;

DskcfTracker::DskcfTracker()
{
	std::shared_ptr< Kernel > kernel = std::make_shared< GaussianKernel >();
	std::shared_ptr< FeatureExtractor > features = std::make_shared< HOGFeatureExtractor >();
	std::shared_ptr< FeatureChannelProcessor > processor = std::make_shared< ConcatenateFeatureChannelProcessor >();

	this->m_occlusionHandler = std::make_shared< OcclusionHandler >(KcfParameters(), kernel, features, processor);
}

DskcfTracker::~DskcfTracker()
{
}

bool DskcfTracker::update(const std::array< cv::Mat, 2 > & frame, Rect & boundingBox)
{
	Point position = centerPoint(boundingBox);

	boundingBox = (this->m_occlusionHandler->detect(frame, position));

	position = centerPoint(boundingBox);

	this->m_occlusionHandler->update(frame, position);

	return !this->m_occlusionHandler->isOccluded();
}

bool DskcfTracker::update(const std::array< cv::Mat, 2 > & frame, cv::Rect_<double>& boundingBox, std::vector<int64> &timePerformanceVector)

{
	int64 tStart = cv::getTickCount();

	Point position = centerPoint(boundingBox);

	boundingBox = (this->m_occlusionHandler->detect(frame, position));

	position = centerPoint(boundingBox);

	this->m_occlusionHandler->update(frame, position);

	int64 tStop = cv::getTickCount();

	int lastElement = (int)timePerformanceVector.size() - 1;
	timePerformanceVector = this->m_occlusionHandler->singleFrameProTime;
	//re-init the vector
	this->m_occlusionHandler->singleFrameProTime = std::vector<int64>(8, 0);
	timePerformanceVector[lastElement] = tStop - tStart;

	return !this->m_occlusionHandler->isOccluded();

}

bool DskcfTracker::reinit(const std::array< cv::Mat, 2 > & frame, Rect & boundingBox)
{
	std::shared_ptr< Kernel > kernel = std::make_shared< GaussianKernel >();
	std::shared_ptr< FeatureExtractor > features = std::make_shared< HOGFeatureExtractor >();
	std::shared_ptr< FeatureChannelProcessor > processor = std::make_shared< ConcatenateFeatureChannelProcessor >();

	this->m_occlusionHandler = std::make_shared< OcclusionHandler >(KcfParameters(), kernel, features, processor);

	this->m_occlusionHandler->init(frame, boundingBox);

	return true;
}

TrackerDebug* DskcfTracker::getTrackerDebug()
{
	return nullptr;
}

const std::string DskcfTracker::getId()
{
	return "DSKCF";
}
