#ifndef _OCCLUSIONHANDLER_HPP_
#define _OCCLUSIONHANDLER_HPP_
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
the main tracking system and occlusion managment presented in [1] is performed within this class

References:
[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao,
DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
*/


#include <array>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/core.hpp>
#include <KalmanFilter2D.h>
#include <KalmanFilter1D.h>
#include <DepthWeightKCFTracker.h>

#include "FeatureExtractor.hpp"
#include "DepthSegmenter.hpp"
#include "FeatureExtractor.hpp"
#include "kcf_tracker.hpp"
#include "ScaleChangeObserver.hpp"
#include "FeatureChannelProcessor.hpp"


struct ThreadResult
{
  double score;
  std::vector< cv::Point_< double > >::iterator value;

  const bool operator<( const ThreadResult & rval ) const;
};

/**
 * A wrapper class around a pair of trackers, one for the target object and another for the occluding object.
 * This class handles the change of state from visible to occluded and calls the appropriate tracker.
 */
class OcclusionHandler : public ScaleChangeObserver
{
public:
  /**
   * @param paras The KCF tracker parameters to be used by both trackers.
   * @param kernel The kernel to be used for the correlation step in the tracker.
   * @param depthSegmenter A non-owning reference to the depth segmenter to be used by this tracker.
   * @param featureExtractor A non-owning reference to the feature extractor to be used by this tracker.
   * @warning None of these parameters should be null.
   */
  OcclusionHandler( KcfParameters paras, std::shared_ptr< Kernel > & kernel, std::shared_ptr< FeatureExtractor > & featureExtractor, std::shared_ptr< FeatureChannelProcessor > & featureProcessor );
  virtual ~OcclusionHandler();

  /**
   * Initialise the tracker.
   *
   * @param frame The RGB and depth maps for the current frame.
   * @param boundingBox The boundingBox of the target object.
   */
  void init( const std::array< cv::Mat, 2 > & frame, const Rect & target );

  /**
   * Detect the object or occluder
   *
   * @param frame The RGB and depth maps for the current frame.
   * @param boundingBox The boundingBox of the target object or occluder.
   *
   * @returns The new bounding box of the target object or the occluder.
   */
  const Rect detect( const std::array< cv::Mat, 2 > & frame, const Point & position );

  /**
   * Update the tracker's model
   *
   * @param frame The RGB and depth maps for the current frame.
   * @param position The position of the target object or occluder
   */
  void update( const std::array< cv::Mat, 2 > & frame, const Point & position );

  virtual void onScaleChange( const Size & targetSize, const Size & windowSize, const cv::Mat2d & yf, const cv::Mat1d & cosineWindow );

  const bool isOccluded() const;

  std::vector<int64> singleFrameProTime;

private:
  bool m_isOccluded;
  std::shared_ptr< FeatureChannelProcessor > m_featureProcessor;
  std::shared_ptr< FeatureExtractor > m_featureExtractor;
  std::shared_ptr< DepthSegmenter > m_depthSegmenter;
  std::shared_ptr< ScaleAnalyser > m_scaleAnalyser;
  KcfParameters m_paras;
  std::shared_ptr< Kernel > m_kernel;
  cv::Size_< double > m_targetSize;
  cv::Size_< double > m_initialSize;
  cv::Size_< double > m_windowSize;
  cv::Size_< double > m_occluderSize;
  cv::Size_< double > m_occluderWindowSize;
  cv::Rect_< double > m_searchWindow;
  cv::Mat m_cosineWindow;
  cv::Mat m_occluderCosineWindow;

  double m_lambdaOcc;
  double m_lambdaR1;
  double m_lambdaR2;
  double m_targetDepthMean;
  double m_targetDepthSTD;

  std::array< std::shared_ptr< DepthWeightKCFTracker >, 2 > m_targetTracker;
  std::shared_ptr< KcfTracker > m_occluderTracker;
  KalmanFilter2D m_filter;

  /**
   * Detect the target object. This method also checks if the target object is occluded.
   *
   * @param frame The RGB and depth maps for the current frame.
   * @param[in,out] boundingBox The boundingBox of the target object.
   *
   * @returns The maximum response of the tracker.
   */
  const Rect visibleDetect( const std::array< cv::Mat, 2 > & frame, const Point & position );

  /**
   * Update the tracker's model
   *
   * @param frame The RGB and depth maps for the current frame.
   * @param[in,out] boundingBox The boundingBox of the target object.
   *
   * @returns True if model was successfully updated
   */
  void visibleUpdate( const std::array< cv::Mat, 2 > & frame, const Point & position );

  /**
   * Detect the occluding object. This method also checks if the target object is visible.
   *
   * @param frame The RGB and depth maps for the current frame.
   * @param[in,out] boundingBox The boundingBox of the occluding object.
   *
   * @returns The maximum response of the tracker.
   */
  const Rect occludedDetect( const std::array< cv::Mat, 2 > & frame, const Point & position,float smallAreaFraction=0.05 );

  /**
   * Update the tracker's model
   *
   * @param frame The RGB and depth maps for the current frame.
   * @param[in,out] boundingBox The boundingBox of the occluding object.
   *
   * @returns True if model was successfully updated
   */
  void occludedUpdate( const std::array< cv::Mat, 2 > & frame, const Point & position );

  /**
   * Called when an occlusion has been detected.
   *
   * @param frame The RGB and depth maps for the current frame.
   * @param features The features of the occluding object.
   * @param[in,out] boundingBox The occluding object's bounding box.
   */
  const Rect onOcclusion( const std::array< cv::Mat, 2 > & frame, std::vector< std::shared_ptr< FC > > & features, const Rect & boundingBox );

   /**
   * Called to select the segmented occluder region
   *
   * @param occluderMask binary mask containing the segmented occluder 
   * @param minimumArea minimun area required by an object in order to be considered as occluder 
   * @param[in,out] boundingBox The occluding object's bounding box.
   */
  const Rect selectOccludingRegion(const cv::Mat1b & occluderMask,int minimumArea,  const Rect &targetBB);


  /**
   * Called when the target object has been found, after occlusion.
   *
   * @param frame The RGB and depth maps for the current frame.
   * @param features The features of the target object.
   * @param[in,out] boundingBox The target object's bounding box.
   */
  void onVisible( const std::array< cv::Mat, 2 > & frame, std::vector< std::shared_ptr< FC > > & features, const Point & position );

  /**
   * Produces a list of regions that potentially contain the target object.
   * @param depth The depth map of the scene.
   * @param targetDepth The mean depth of the target object.
   * @param targetSTD The standard deviation of the target object.
   * @param occluderMask A mask to exclude the occluding object from being a candidate region.
   *
   * @returns A collection of points representing the center of each candidate region.
   */
  std::vector< cv::Point_< double > > findCandidateRegions( const cv::Mat1d & depth, const double targetDepth, const double targetSTD, cv::Mat1b occluderMask );
  std::vector< cv::Point_< double > >::iterator findBestCandidateRegion( const std::array< cv::Mat, 2 > & frame, std::vector< cv::Point_< double > > & candidates );
  std::vector< cv::Point_< double > >::iterator findBestCandidateRegion( const std::array< cv::Mat, 2 > & frame, std::vector< cv::Point_< double > > & candidates, std::vector< float > &centersCandidate);

  /**
   * Calculates the result of equation (11) \f$ (\Phi(\Omega_{obj}) > \lambda_{occ}) \wedge (\widehat{f(z)} < \lambda_{r1}) \f$ in \cite DSKCF
   *
   * @param histogram The histogram of depth frequency
   * @param objectBin The first bin in the histogram which belongs to the object
   * @param maxResponse The maximum response from the KCF tracker
   *
   * @returns True if the target object has been occluded
   */
  bool evaluateOcclusion( const DepthHistogram & histogram, const int objectBin, const double maxResponse );

  /**
  * Calculates the result of equation (11) \f$ (\Phi(\Omega_{obj}) > \lambda_{occ}) \wedge (\widehat{f(z)} < \lambda_{r1}) \f$ in \cite DSKCF
  *
  * @param histogram The histogram of depth frequency
  * @param objectBin The first bin in the histogram which belongs to the object
  * @param maxResponse The maximum response from the KCF tracker
  * @param totalArea area of the tracking patch 
  *
  * @returns True if the target object has been occluded
  */
  bool evaluateOcclusion( const DepthHistogram & histogram, const int objectBin, const double maxResponse, const double totalArea );

  /**
   * Calculates the result of equation (13) \f$ (\Phi(\Omega_{T_{bc}}) < \lambda_{occ}) \wedge (\widehat{f(z)} > \lambda_{r2}) \f$ in \cite DSKCF
   *
   * @param histogram The histogram of depth frequency
   * @param objectBin The first bin in the histogram which belongs to the object
   * @param maxResponse The maximum response from the KCF tracker
   *
   * @returns True if the target object has become visible
   */
  bool evaluateVisibility( const DepthHistogram & histogram, const int objectBin, const double maxResponse )const;


  /**
   * Implements the Φ operator used in equation (11) and (13) in \cite DSKCF
   *
   * @param histogram The histogram of depth frequency
   * @param objectBin The first bin in the histogram which belongs to the object
   *
   * @returns The area of the depth histogram between zero and objectBin
   */
  double phi( const DepthHistogram & histogram, const int objectBin )const;
  double phi( const DepthHistogram & histogram, const int objectBin, const double totalArea )const;
  void initialiseOccluder( const cv::Mat & frame, const cv::Rect_< double > occluderBB );
  ThreadResult scoreCandidate( const std::array< cv::Mat, 2 > & frame, std::vector< cv::Point_< double > >::iterator iterator ) const;
  ThreadResult scoreCandidate( const std::array< cv::Mat, 2 > & frame, std::vector< cv::Point_< double > >::iterator iterator,float &candidateCenterDepth ) const;
};

#endif
