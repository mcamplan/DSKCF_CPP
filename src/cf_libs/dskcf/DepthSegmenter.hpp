#ifndef _DEPTHSEGMENTER_HPP_
#define _DEPTHSEGMENTER_HPP_
/*
// License Agreement (3-clause BSD License)
// Copyright (c) 2016, Jake Hall, Massimo Camplan, Sion Hannuna.
// Third party copyrights and patents are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
*/

/*
This class represents a C++ implementation of the DS-KCF Tracker [1]. In particular
the depth segmentation algorithm presented in [1] is performed within this class

References:
[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao,
DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
*/
#include <array>
#include <tuple>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/core.hpp>
#include <DepthHistogram.h>

//#include "tbb/tick_count.h"
	
/**
 * DepthSegmenter implements the fast depth segmentation described in
 * section 3.1 of \cite DSKCF. Although init and update seem like pure functions,
 * they mutate the internal state of the object. This state can be access
 * via the getters.
 */
class DepthSegmenter
{
public:
	DepthSegmenter();

	/**
	 * Initialise the depth segmenter.
	 *
	 * @param frame The depth map of the current frame
	 * @param boundingBox The bounding box of the target object
	 *
	 * @returns The frame, segmented and labelled.
	 */
	cv::Mat1i init( const cv::Mat & frame, const cv::Rect_< double > & boundingBox );

	/**
	 * Update the depth segmenter.
	 *
	 * @param frame The depth map of the current frame
	 * @param boundingBox The bounding box of the target object
	 *
	 * @returns index of the left-most bin of the histogram
	 */
	int update( const cv::Mat & frame, const cv::Rect_< double > & boundingBox );

	const cv::Mat1b segment( const cv::Mat1w & frame, const cv::Rect_< double > & boundingBox ) const;

	const std::vector< cv::Point_< double > > segmentOccluder( const cv::Mat1w & frame, const cv::Rect_< double > & boundingBox, const int minimumArea,cv::Mat1b &objectMask ) const;

	//overloading function, to pass as input DepthHistogram::Labels labels these information will be used to score the candidates (using the already segmented regions)
	const std::vector< cv::Point_< double > > segmentOccluder( const cv::Mat1w & frame, const cv::Rect_< double > & boundingBox, const int minimumArea,cv::Mat1b &objectMask,
		std::vector< float > &centersCandidate, cv::Rect_<double> &occluderRect) const;

	/** @returns The mean of the target object's depth. */
	double getTargetDepth() const;
	/** @returns The standard deviation of the target object's depth. */
	double getTargetSTD() const;
	/** @returns The non-normalised histogram of depths in the target region. */
	const DepthHistogram & getHistogram() const;
	/** @returns The segmented and labelled depth map. */
	const cv::Mat1i & getLabeledImage() const;
	const bool isOccluded() const;

	const std::vector<int> &getAreaRegions() const;
	
	const DepthHistogram::Labels &getLabelsResults() const;  

	/**
	 * Returns the the index of the center corresponding to the closest object to the camera with the biggest area
	 *
	 * @param centroids The histogram centroids of the depth from the k-means.
	 * @param labelsC The labels corresponding to each the centroids
	 *
	 * @returns return the index (to access to the labelsC vector) corresponding to the closest object with the biggest area
	 */
	const int selectClosestObject( std::vector< float > & centroids) ;

	const int selectClosestObject( std::vector< float > & centroids,std::vector< int > & areaVector) const;

	//debug function to save histogram
	void debugSaveHistogram(std::string filename);
private:
	/** The mean of the target object's depth */
	double m_targetDepth;
	/** The standard deviation of the target object's depth */
	double m_targetSTD;
  /** The non-normalised histogram of depths in the target region */
	DepthHistogram m_histogram;
	/** The segmented and labelled depth map */
	//cv::Mat1b m_labeledImage;
	cv::Mat1i m_labeledImage;
	bool m_occluded;

	float minSTD;

	/** The are of the estimated region in  the image plane*/
	std::vector<int> areaRegions;

	/*internal variable to store segmentation data*/
	DepthHistogram::Labels labelsResults; 

	/**
	 * Produces a segmented and labelled image of the target region.
	 *
	 * @param region The depth map of the target region.
	 * @param mask The mask to be excluded from labelling (e.g. for depth shadows). These pixels are given the maximum label value.
	 * @param centroids The histogram centroids of the depth from the k-means.
	 * @param labels The histogram labels of the depth from the k-means.
	 *
	 * @returns An 8-bit labelled image of the target region. Low labels have lower depth values, except for mask excluded pixels which have the highest label.
	 */
	const cv::Mat1b createLabelImage( const cv::Mat1w & region, const cv::Mat1b mask,
																		const std::vector< float > & centroids, const std::vector< int > & labels ) const;

	/**
	* Produces a segmented and labelled image of the target region.
	*
	* @param region The depth map of the target region.
	* @param mask The mask to be excluded from labelling (e.g. for depth shadows). These pixels are given the maximum label value.
	* @param centroids The histogram centroids of the depth from the k-means.
	* @param labels The histogram labels of the depth from the k-means.
	*
	* @returns An 8-bit labelled image of the target region. Low labels have lower depth values, except for mask excluded pixels which have the highest label.
	* @returns histogram depth Histogram
	*/
	const cv::Mat1b createLabelImage( const cv::Mat1w & region, const cv::Mat1b mask,
																		const std::vector< float > & centroids, const std::vector< int > & labels,
																		const DepthHistogram &histogram) const;

	
	/**
	 * Produces a segmented and labelled image of the target region, adding the connected component analysis.
	 *
	 * @param region The depth map of the target region.
	 * @param mask The mask to be excluded from labelling (e.g. for depth shadows). These pixels are given the maximum label value.
	 * @param centroids The histogram centroids of the depth from the k-means.
	 * @param labels The histogram labels of the depth from the k-means.
	 * @param labelsC The labels corresponding to each the centroids
	 * @param smallAreaFraction fraction of the target area used to define the minimum object size
	 *
	 * @returns An 32-bit labelled image of the target region. Low labels have lower depth values, except for mask excluded pixels which have the zero label.
	 */
	const cv::Mat1i createLabelImageCC( const cv::Mat1w & region, const cv::Mat1b mask,
														std::vector< float > & centroids, const std::vector< int > & labels, std::vector< int > & labelsC,
														float smallAreaFraction=0.09);

 	/**
	 * Produces the list of possible target candidates occluded region,the segmentation and component analysis are then filtered
	 * by considering the occluding object (front one) and removing very small regions
	 *
	 * @param region The depth map of the target region.
	 * @param mask The mask to be excluded from labelling (e.g. for depth shadows). These pixels are given the maximum label value.
	 * @param centroids The histogram centroids of the depth from the k-means.
	 * @param labels The histogram labels of the depth from the k-means.
	 * @param labelsC The labels corresponding to each the centroids
	 * @param smallArea fraction of the target area used to define the minimum object size
	 *
	 * @returns A vector containing the list of candidate points is returned
	 * @returns objectMask segmented mask of the occluding object
	 */
	const std::vector< cv::Point_< double > >  createLabelImageCCOccluder( const cv::Mat1w & region, const cv::Mat1b mask,
														std::vector< float > & centroids, const std::vector< int > & labels, std::vector< int > & labelsC,
														const DepthHistogram &histogram,int smallArea,cv::Mat1b &objectMask) const; 

	/**
	* Produces the list of possible target candidates occluded region,the segmentation and component analysis are then filtered
	* by considering the occluding object (front one) and removing very small regions
	*
	* @param region The depth map of the target region.
	* @param mask The mask to be excluded from labelling (e.g. for depth shadows). These pixels are given the maximum label value.
	* @param centroids The histogram centroids of the depth from the k-means.
	* @param labels The histogram labels of the depth from the k-means.
	* @param labelsC The labels corresponding to each the centroids
	* @param smallArea fraction of the target area used to define the minimum object size
	*
	* @returns A vector containing the list of candidate points is returned
	* @returns objectMask segmented mask of the occluding object
	* @returns centroidsCandidates depth value corresponding to the candidate centroids
	* @returns occluderRect rectangle including the occluder
	*/
	const std::vector< cv::Point_< double > >  createLabelImageCCOccluder( const cv::Mat1w & region, const cv::Mat1b mask,
														std::vector< float > & centroids, const std::vector< int > & labels, std::vector< int > & labelsC,
														const DepthHistogram &histogram,int smallArea,cv::Mat1b &objectMask, std::vector< float > & centroidsCandidates,
														cv::Rect_<double> &occluderRect) const; 


	/**
	 * Conditionally updates the internal state of the DepthSegmenter.
	 *
	 * @param centroids The centroids from the k-means.
	 * @param previousDepth The mean of the target object's depth from the previous frame.
	 * @param previousSTD The standard deviation of the target object's depth from the previous frame.
	 * @param targetDepth The mean of the target object's depth from the current frame.
	 * @param targetSTD The standard deviation of the target object's depth from the current frame.
	 *
	 * @returns The label corresponding to the target object.
	 */
	const int handleOcclusion( const std::vector< float > & centroids, const double previousDepth,
														 const double previousSTD, const double targetDepth, const double targetSTD );

	/**
	 * Conditionally updates the internal state of the DepthSegmenter.
	 *
	 * @param centroids The centroids from the k-means and Connected component.
	 * @param labelsC of the centroids after the connected components.
	 * @param previousDepth The mean of the target object's depth from the previous frame.
	 * @param previousSTD The standard deviation of the target object's depth from the previous frame.
	 * @param targetDepth The mean of the target object's depth from the current frame.
	 * @param targetSTD The standard deviation of the target object's depth from the current frame.
	 *
	 * @returns The label corresponding to the target object.
	 */
	const int handleOcclusion( const cv::Mat& image, const std::vector< float > & centroids,const std::vector< int > & labelsC, const double previousDepth,
														 const double previousSTD, const double targetDepth, const double targetSTD );
};

#endif
