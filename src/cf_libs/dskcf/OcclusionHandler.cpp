#include "OcclusionHandler.hpp"

#include <tbb/concurrent_vector.h>

OcclusionHandler::OcclusionHandler( KcfParameters paras, std::shared_ptr< Kernel > & kernel, std::shared_ptr< FeatureExtractor > & featureExtractor, std::shared_ptr< FeatureChannelProcessor > & featureProcessor )
{
  this->m_paras = paras;
  this->m_kernel = kernel;
  this->m_featureExtractor = featureExtractor;
  this->m_featureProcessor = featureProcessor;
  this->m_depthSegmenter = std::make_shared< DepthSegmenter >();
  this->m_scaleAnalyser = std::make_shared< ScaleAnalyser >( this->m_depthSegmenter.get(), paras.padding );

  for( int i = 0; i < 2; i++ )
  {
    this->m_targetTracker[ i ] = std::make_shared< DepthWeightKCFTracker >( paras, kernel );
  }

  this->m_occluderTracker = std::make_shared< KcfTracker >( paras, kernel );

  this->m_lambdaOcc = 0.35;
  this->m_lambdaR1 = 0.4;
  this->m_lambdaR2 = 0.2;
  this->m_isOccluded = false;

  this->singleFrameProTime = std::vector<int64>(8,0);
}

OcclusionHandler::~OcclusionHandler()
{
  this->m_depthSegmenter = nullptr;
}

void OcclusionHandler::init( const std::array< cv::Mat, 2 > & frame, const Rect & target )
{
  std::vector< std::shared_ptr< FC > > features( 2 );
  this->m_isOccluded = false;
  this->m_initialSize = target.size();

  this->m_scaleAnalyser->clearObservers();
  this->m_scaleAnalyser->registerScaleChangeObserver( this );
  this->m_scaleAnalyser->registerScaleChangeObserver( this->m_targetTracker[ 0 ].get() );
  this->m_scaleAnalyser->registerScaleChangeObserver( this->m_targetTracker[ 1 ].get() );
  this->m_depthSegmenter->init( frame[ 1 ], target );
  this->m_scaleAnalyser->init( frame[ 1 ], target );

  Point position = centerPoint( target );
  Rect window = boundingBoxFromPointSize( position, this->m_windowSize );

  //Extract features
  for( int i = 0; i < 2; i++ )
  {
    features[ i ] = this->m_featureExtractor->getFeatures( frame[ i ], window );
    FC::mulFeatures( features[ i ], this->m_cosineWindow );
  }

  features = this->m_featureProcessor->concatenate( features );

  for( uint i = 0; i < features.size(); i++ )
  {
    this->m_targetTracker[ i ]->init( frame[ i ], features[ i ], position );
  }

  this->m_filter.initialise( position );
}

const Rect OcclusionHandler::detect( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
  if( this->m_isOccluded )
  {
    return this->occludedDetect( frame, position );
  }
  else
  {
    return this->visibleDetect( frame, position );
  }
}

void OcclusionHandler::update( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
  if( this->m_isOccluded )
  {
    return this->occludedUpdate( frame, position );
  }
  else
  {
     return this->visibleUpdate( frame, position );
  }
}

const Rect OcclusionHandler::visibleDetect( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
  
  int64 tStartDetection=cv::getTickCount();
  std::vector< double > responses;
  std::vector< std::shared_ptr< FC > > features( 2 );
  std::vector< Point > positions;
  
  Rect target = boundingBoxFromPointSize( position, this->m_targetSize );
  Rect window = boundingBoxFromPointSize( position, this->m_windowSize );


  tbb::parallel_for< uint >( 0, 2, 1,
	  [this,&frame,&features,&window]( uint index ) -> void
	  {
		  features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
		  FC::mulFeatures( features[ index ], this->m_cosineWindow );
	  }
  );

  features = this->m_featureProcessor->concatenate( features );
  std::vector< cv::Mat > frames_ = this->m_featureProcessor->concatenate( std::vector< cv::Mat >( frame.begin(), frame.end() ) );

  for( uint i = 0; i < features.size(); i++ )
  {
    DetectResult result = this->m_targetTracker[ i ]->detect( frames_[ i ], features[ i ], position, this->m_depthSegmenter->getTargetDepth(), this->m_depthSegmenter->getTargetSTD() );
    positions.push_back( result.position );
    responses.push_back( result.maxResponse );
  }
  //here the maximun response is calculated....
  int64 tStopDetection = cv::getTickCount();  
  this->singleFrameProTime[0]=tStopDetection-tStartDetection;

  int64 tStartSegment = tStopDetection;
  //TO BE CHECKED IN CASE OF MULTIPLE MODELS...LINEAR ETC....WORKS ONLY FOR SINGLE (or concatenate) features
  target = boundingBoxFromPointSize( positions.back(), this->m_targetSize );
  int bin=this->m_depthSegmenter->update( frame[ 1 ], target );

  DepthHistogram histogram = this->m_depthSegmenter->getHistogram();

  double totalArea=target.area()*1.05;


  if( this->evaluateOcclusion( histogram, bin, this->m_featureProcessor->concatenate( responses ),totalArea ) )
  {
	  //here the maximun response is calculated....
	  int64 tStopSegment = cv::getTickCount();  
	  this->singleFrameProTime[1]=tStopSegment-tStartSegment;
	  
	  int64 tStartNewTracker = tStopSegment;  

	  const Rect retRect=this->onOcclusion( frame, features, target );

	  int64 tStopNewTracker = cv::getTickCount();  
	  this->singleFrameProTime[2]=tStopNewTracker-tStartNewTracker;

	  return retRect;
	  
	  
  }
  else
  {
	//here the maximun response is calculated....
	int64 tStopSegment = cv::getTickCount();  
	this->singleFrameProTime[1]=tStopSegment-tStartSegment;
	  //Is the object entirely unoccluded?
    if( !this->m_depthSegmenter->isOccluded() )
    {

    }

    Point estimate = this->m_featureProcessor->concatenate( positions );

	estimate.x=(estimate.x -this->m_targetSize.width/2)<	frame[ 0 ].cols	 ? estimate.x : this->m_targetSize.width; 
	estimate.y=(estimate.y -this->m_targetSize.height/2)<	frame[ 0 ].rows	 ? estimate.y : this->m_targetSize.height; 
	estimate.x=(estimate.x +this->m_targetSize.width/2)>	0	 ? estimate.x : 1; 
	estimate.y=(estimate.y +this->m_targetSize.height/2)>	0	 ? estimate.y : 1; 
    return boundingBoxFromPointSize( estimate, this->m_initialSize * this->m_scaleAnalyser->getScaleFactor() );

  }
}

void OcclusionHandler::visibleUpdate( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
	//EVALUATE CHANGE OF SCALE....
	int64 tStartScaleCheck=cv::getTickCount();
	std::vector< std::shared_ptr< FC > > features( 2 );
	Rect window = boundingBoxFromPointSize( position, this->m_windowSize );

	this->m_scaleAnalyser->update( frame[ 1 ], window );

	
	int64 tStopScaleCheck = cv::getTickCount();  
	this->singleFrameProTime[5]=tStopScaleCheck-tStartScaleCheck;



	int64 tStartModelUpdate=tStopScaleCheck;
	window = boundingBoxFromPointSize( position, this->m_windowSize );

	tbb::parallel_for< uint >( 0, 2, 1,
		[this,&frame,&features,&window]( uint index ) -> void
	{
		features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
		FC::mulFeatures( features[ index ], this->m_cosineWindow );
	}
	);

	features = this->m_featureProcessor->concatenate( features );

	for( size_t i = 0; i < features.size(); i++ )
	{
		this->m_targetTracker[ i ]->update( frame[ i ], features[ i ], position );
	}

	int64 tStopModelUpdate = cv::getTickCount();  
	this->singleFrameProTime[6]=tStopModelUpdate-tStartModelUpdate;

}

const Rect OcclusionHandler::occludedDetect( const std::array< cv::Mat, 2 > & frame, const Point & position, float smallAreaFraction)
{
  int64 tStartTrackOccluder = cv::getTickCount();

  Rect imageRect( 0, 0, frame[ 1 ].cols, frame[ 1 ].rows );
  Rect window = rectRound( boundingBoxFromPointSize( position, this->m_occluderWindowSize ) );
  Rect target = rectRound( boundingBoxFromPointSize( position, this->m_occluderSize ) );//old occluder position
 
  Point prediction = position;

  //Extract features
  auto features = this->m_featureExtractor->getFeatures( frame[ 0 ], window );
  FC::mulFeatures( features, m_occluderCosineWindow );


  Rect oldSearchWindow=this->m_searchWindow;//old occluder search window
  //mix it with the predicted position
  Rect newSearchWindow=extremeRect( boundingBoxFromPointSize( prediction, this->m_targetSize ), extremeRect( oldSearchWindow, target ) );
  
  //move here the response of the tracker....then test the 
  Rect result = boundingBoxFromPointSize( this->m_occluderTracker->detect( frame[ 0 ], features, position ).position, this->m_occluderSize );
  int64 tStopTrackOccluder = cv::getTickCount();  
  this->singleFrameProTime[3]=tStopTrackOccluder-tStartTrackOccluder;


  //START COUNTER FOR SOLVING OCCLUSIONS....
  int64 tStartSolveOcclusions = tStopTrackOccluder;


  this->m_searchWindow = extremeRect( newSearchWindow, result) & imageRect;
  this->m_searchWindow = resizeBoundingBox( this->m_searchWindow, this->m_searchWindow.size() ) & imageRect;
  Rect areaToSegment = resizeBoundingBox( this->m_searchWindow,  this->m_searchWindow.size()* 1.05 ) & imageRect;
  
  //substitute the next three lines with a better segmentation 
  cv::Mat1w area( frame[ 1 ], areaToSegment );

  cv::Size size=this->m_targetSize;
  int tmpWidth=static_cast<int>(size.height);
  int tmpHeight = static_cast<int>(size.height);
  int minimumArea=cvRound(smallAreaFraction*tmpWidth*tmpHeight);
  cv::Mat1b objectMask(frame[ 1 ].rows,frame[ 1 ].cols);
  objectMask.setTo(0);
  cv::Mat1b tmpObjectMask= objectMask(areaToSegment );
  cv::Rect_<double> objMaskRect;

  std::vector< float > centersCandidate;

  std::vector< cv::Point_< double > > candidates = this->m_depthSegmenter->segmentOccluder(area,Rect( 0, 0, area.cols, area.rows ),minimumArea,tmpObjectMask,centersCandidate,objMaskRect);  
  
  //re-update here the window search with the segmented occluder....
  objMaskRect.x+=areaToSegment.x;
  objMaskRect.y+=areaToSegment.y;
  this->m_searchWindow = extremeRect( objMaskRect, this->m_searchWindow) & imageRect;

  std::for_each( candidates.begin(), candidates.end(), [ this, areaToSegment ]( cv::Point_< double > & candidate ) -> void { candidate += areaToSegment.tl(); } );//this->m_searchWindow.tl(); } );


  auto itr = this->findBestCandidateRegion( frame, candidates,centersCandidate);
  if( itr == candidates.end() )
  {
    //MOVED UP!!!!!!

    //Rect result = boundingBoxFromPointSize( this->m_occluderTracker->detect( frame[ 0 ], features, position ).position, this->m_occluderSize );
    //this->m_searchWindow = extremeRect( extremeRect( this->m_searchWindow, result ), boundingBoxFromPointSize( prediction, this->m_targetSize ) ) & imageRect;
  
	int64 tStopSolveOcclusions = cv::getTickCount();  
    this->singleFrameProTime[4]=tStopSolveOcclusions-tStartSolveOcclusions;
    return result;
  }
  else
  {
    //check now if the bounding box belong to the occluder mask or not
    cv::Point shiftedPoint=pointRound(*itr);
	bool onMaskPixel=(objectMask(shiftedPoint)>0);
	objectMask.release();
	if(onMaskPixel){
		this->m_isOccluded = true;
		int64 tStopSolveOcclusions = cv::getTickCount();  
		this->singleFrameProTime[4]=tStopSolveOcclusions-tStartSolveOcclusions;
		return result;
  }
	
	return boundingBoxFromPointSize( *itr, this->m_targetSize );
}
}

void OcclusionHandler::occludedUpdate( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
  int64 tStartTrackOccluder = cv::getTickCount();  

  Size paddedSize = sizeRound( this->m_occluderSize * this->m_paras.padding );
  Rect window = boundingBoxFromPointSize( position, paddedSize );

  auto features = this->m_featureExtractor->getFeatures( frame[ 0 ], window );
  FC::mulFeatures( features, m_occluderCosineWindow );

  this->m_occluderTracker->update( frame[ 0 ], features, position );

  int64 tStopTrackOccluder = cv::getTickCount();  
  int64 newInterval=tStopTrackOccluder-tStartTrackOccluder;
  this->singleFrameProTime[3]+=newInterval;
}

const Rect OcclusionHandler::onOcclusion( const std::array< cv::Mat, 2 > & frame, std::vector< std::shared_ptr< FC > > & features, const Rect & boundingBox )
{
  this->m_isOccluded = true;
  cv::Scalar mean, stddev;
  Rect imageRect( 0, 0, frame[ 0 ].cols, frame[ 0 ].rows );
  Rect boundingBoxModified=getSubWindowRounding( boundingBox );

  Rect window = resizeBoundingBox( boundingBoxModified, this->m_windowSize );
  //Find the region of the window belonging to the occluder
  auto L = this->m_depthSegmenter->getLabeledImage();

  //You must now select the closest and the biggest area!!!!
  DepthHistogram::Labels labels=this->m_depthSegmenter->getLabelsResults();
  int indexCenter=this->m_depthSegmenter->selectClosestObject( labels.centers);
  cv::Mat occluder = createMask< uchar >( L, labels.labelsC[indexCenter], false );
  
  //cv::meanStdDev( cv::Mat( frame[ 1 ], boundingBoxModified & imageRect ), mean, stddev );
  cv::meanStdDev(cv::Mat(frame[1], boundingBoxModified & imageRect), mean, stddev, occluder);
  stddev.val[ 0 ] = modelNoise( mean.val[ 0 ], stddev.val[ 0 ] );
  
  occluder = getRegion< ushort >( cv::Mat( frame[ 1 ], imageRect ), cvFloor( mean.val[ 0 ] - stddev.val[ 0 ] ), cvCeil( mean.val[ 0 ] + stddev.val[ 0 ] ) );
  
  //now filter out small regions and keep the one with maximum overlap....
  auto occlusionCandidates = connectedComponents< uchar >( occluder );


  //Find the bounding box that has the largest overlapping region with the target, excluding the background.
  cv::Rect_< int > occluderBBCandidate=*(std::max_element( occlusionCandidates.begin() + 1, occlusionCandidates.end(),
                                                                         [ boundingBoxModified ]( const cv::Rect_< int > & a, const cv::Rect_< int > & b ) -> bool
    {
                                                                           cv::Rect_< int > bb = rectRound( boundingBoxModified );
      return ( a & bb ).area() < ( b & bb ).area();
    }
  ));
  
  cv::Rect_< int > occluderBB = rectRound( window ) & occluderBBCandidate;

  this->m_searchWindow = extremeRect< double >( rectCast< double >( occluderBB ), boundingBox );

  if (occluderBB.area()>0)
	  //Initialise the occluder tracker with the new bounding box
	  this->initialiseOccluder(frame[0], occluderBB);
  else{
	  //printf("ERROR\n");
	  this->m_isOccluded=false;
  }
  //Store the target object's depth and std
  this->m_targetDepthSTD = this->m_depthSegmenter->getTargetSTD();
  this->m_targetDepthMean = this->m_depthSegmenter->getTargetDepth();

  //Output the occluders bounding box
  return occluderBB;
}



void OcclusionHandler::onVisible( const std::array< cv::Mat, 2 > & frame, std::vector< std::shared_ptr< FC > > & features, const Point & position )
{
  this->m_isOccluded = false;
}

std::vector< Point > OcclusionHandler::findCandidateRegions( const cv::Mat1d & depth, const double targetDepth, const double targetSTD, const cv::Mat1b occluderMask )
{
  cv::Mat1b candidates = cv::Mat1b::zeros( depth.rows, depth.cols );
  std::vector< Point > result;

  if( targetSTD != 0.0 )
  {
    for( int row = 0; row < depth.rows; row++ )
    {
      for( int col = 0; col < depth.cols; col++ )
      {
        if(
            ( depth( row, col ) > ( targetDepth - 2 * targetSTD ) ) &&
            ( depth( row, col ) < ( targetDepth + 2 * targetSTD ) ) &&
            ( occluderMask( row, col ) != 0 )
        )
        {
          candidates( row, col ) = 1;
        }
        else
        {
          candidates( row, col ) = 0;
        }
      }
    }

    auto rects = connectedComponents< uchar >( candidates );

    std::for_each( rects.begin(), rects.end(), [&result]( const cv::Rect_< double > & rect ){ result.push_back( centerPoint( rect ) ); } );
  }

  return result;
}

std::vector< cv::Point_< double > >::iterator OcclusionHandler::findBestCandidateRegion( const std::array< cv::Mat, 2 > & frame, std::vector< cv::Point_< double > > & candidates )
{
  if( candidates.size() > 0 )
  {
	  std::vector< std::vector< cv::Point_< double > >::iterator > iterators;
	  for( auto itr = candidates.begin(); itr != candidates.end(); itr++ )
	  {
		  iterators.push_back( itr );
	  }

	  tbb::concurrent_vector< ThreadResult > scoredCandidates;
	  tbb::parallel_for_each( iterators.begin(), iterators.end(), [this, &frame, &scoredCandidates]( std::vector< cv::Point_< double > >::iterator & candidate ) -> void
		  {
		      scoredCandidates.push_back( this->scoreCandidate( frame, candidate ) );
      }
	  );

    auto max_elem = *std::max_element( scoredCandidates.begin(), scoredCandidates.end() );

    if( ( max_elem.score > this->m_lambdaR2 ) )
    {
      this->m_isOccluded = false;
      return max_elem.value;
    }
  }

  return candidates.end();
}

std::vector< cv::Point_< double > >::iterator OcclusionHandler::findBestCandidateRegion( const std::array< cv::Mat, 2 > & frame, std::vector< cv::Point_< double > > & candidates,std::vector< float > &centersCandidate )
{
  if( candidates.size() > 0 )
  {
    std::vector< std::vector< cv::Point_< double > >::iterator > iterators;
    for( auto itr = candidates.begin(); itr != candidates.end(); itr++ )
    {
      iterators.push_back( itr );
    }

    tbb::concurrent_vector< ThreadResult > scoredCandidates;

	tbb::parallel_for< int >( 0, iterators.size(), 1, [ this, &frame, &scoredCandidates, &iterators, &centersCandidate ]( int i ) -> void
    {
      scoredCandidates.push_back( this->scoreCandidate( frame, iterators[ i ], centersCandidate[ i ] ) );
    }
	);

    auto max_elem = *std::max_element( scoredCandidates.begin(), scoredCandidates.end() );

    if( ( max_elem.score > this->m_lambdaR2 ) )
    {
      this->m_isOccluded = false;
      return max_elem.value;
    }
  }

  return candidates.end();
}

bool OcclusionHandler::evaluateOcclusion( const DepthHistogram & histogram, const int objectBin, const double maxResponse )
{
  // ( f(z)_max < λ_r1 ) ∧ ( Φ( Ω_obj ) > λ_occ l)
  return ( ( maxResponse < this->m_lambdaR1 ) && ( this->phi( histogram, objectBin ) > this->m_lambdaOcc ) );
}


bool OcclusionHandler::evaluateOcclusion( const DepthHistogram & histogram, const int objectBin, const double maxResponse,const double totalArea )
{
  // ( f(z)_max < λ_r1 ) ∧ ( Φ( Ω_obj ) > λ_occ l)
  return ( ( maxResponse < this->m_lambdaR1 ) && ( this->phi( histogram, objectBin,totalArea ) > this->m_lambdaOcc ) );
}

bool OcclusionHandler::evaluateVisibility( const DepthHistogram & histogram, const int objectBin, const double maxResponse ) const
{
  //( f(z)_n > λ_r2 ) ∧ ( Φ( Ω_Tbc ) < λ_occ )
  return ( ( maxResponse > this->m_lambdaR2 ) && ( this->phi( histogram, objectBin ) < this->m_lambdaOcc ) );
}

double OcclusionHandler::phi( const DepthHistogram & histogram, const int objectBin )const
{
  double totalArea = 0.0;
  double occluderArea = 0.0;

  for( uint i = 0; i < histogram.size(); i++ )
  {
    if( i < objectBin )
    {
      occluderArea += histogram[ i ];
    }

    totalArea += histogram[ i ];
  }

  return occluderArea / totalArea;
}


double OcclusionHandler::phi( const DepthHistogram & histogram, const int objectBin,const double totalArea ) const
{

  double occluderArea = 0.0;

  for( uint i = 0; i < histogram.size(); i++ )
  {
    if( i < objectBin )
    {
      occluderArea += histogram[ i ];
    }
	else
		break;
  }

  return occluderArea / totalArea;
}
void OcclusionHandler::onScaleChange( const Size & targetSize, const Size & windowSize, const cv::Mat2d & yf, const cv::Mat1d & cosineWindow )
{
  if( !this->m_isOccluded )
  {
    this->m_targetSize = targetSize;
    this->m_windowSize = windowSize;
    this->m_cosineWindow = cosineWindow;
  }
}

void OcclusionHandler::initialiseOccluder( const cv::Mat & frame, const Rect boundingBox )
{
  cv::Mat2d yf;

  this->m_occluderSize = boundingBox.size();
  this->m_occluderWindowSize = sizeRound( boundingBox.size() * this->m_paras.padding );
  Rect window = resizeBoundingBox( boundingBox, this->m_occluderWindowSize );
  Point position = centerPoint( boundingBox );

  //Create all of the parameters normally created by the scale analyser
  double outputSigma = sqrt( boundingBox.area() ) * this->m_paras.outputSigmaFactor / this->m_paras.cellSize;

  cv::dft(
      gaussianShapedLabelsShifted2D(
          outputSigma, sizeFloor( window.size() * ( 1.0 / static_cast< double >( this->m_paras.cellSize ) ) )
      ), yf, cv::DFT_COMPLEX_OUTPUT
  );

  this->m_occluderCosineWindow = hanningWindow< double >( yf.rows ) * hanningWindow< double >( yf.cols ).t();

  //Extract features
  auto features = this->m_featureExtractor->getFeatures( frame, window );
  FC::mulFeatures( features, this->m_occluderCosineWindow );

  //Setup the occluder tracker
  this->m_occluderTracker = std::make_shared< KcfTracker >( this->m_paras, this->m_kernel );
  this->m_occluderTracker->onScaleChange( boundingBox.size(), window.size(), yf, this->m_occluderCosineWindow );
  this->m_occluderTracker->init( frame, features, position );
}

const bool OcclusionHandler::isOccluded() const
{
  return this->m_isOccluded;
}

ThreadResult OcclusionHandler::scoreCandidate( const std::array< cv::Mat, 2 > & frame, std::vector< cv::Point_< double > >::iterator candidate ) const
{
  cv::Mat patch[ 2 ];
  ThreadResult result;
  result.score = 0.0;
  result.value = candidate;

  std::vector< std::shared_ptr< FC > > features( 2 );
  std::vector< double > maxResponses;
  std::vector< Point > positions;
  Rect window = boundingBoxFromPointSize( *result.value, this->m_windowSize );

  tbb::parallel_for< uint >( 0, 2, 1, [this,&frame,&features,&window]( uint index ) -> void
	  {
		  features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
		  FC::mulFeatures( features[ index ], this->m_cosineWindow );
	  }
  );

  //Concat features and images here...
  features = this->m_featureProcessor->concatenate( features );
  std::vector< cv::Mat > frames_ = this->m_featureProcessor->concatenate( std::vector< cv::Mat >( frame.begin(), frame.end() ) );

  //Calculate the response of the target tracker at the candidate point
  for( uint i = 0; i < features.size(); i++ )
  {
    DetectResult detection = this->m_targetTracker[ i ]->detect( frames_[ i ], features[ i ], *result.value, this->m_targetDepthMean, this->m_targetDepthSTD );
    positions.push_back( detection.position );
    maxResponses.push_back( detection.maxResponse );
  }

  *result.value = this->m_featureProcessor->concatenate( positions );
  double maxResponse = this->m_featureProcessor->concatenate( maxResponses );

  if( getSubWindow( frame[ 0 ], patch[ 0 ], this->m_targetSize, *result.value ) && getSubWindow( frame[ 1 ], patch[ 1 ], this->m_targetSize, *result.value ) )
  {
    cv::Mat1b mask = createMask< ushort >( patch[ 1 ], 0 );
    DepthHistogram histogram = DepthHistogram::createHistogram( cvRound( modelNoise( this->m_targetDepthMean, this->m_targetDepthSTD ) ), patch[ 1 ], mask );
    std::vector< int > peaks = histogram.getPeaks();

	//modification here to fit the matlab version

    int peak = histogram.depthToPeak( this->m_targetDepthMean, peaks );

    //Check to see if the response and depth are good enough to be the target objects
	if( ( peak <= 1 ) && this->evaluateVisibility( histogram, histogram.depthToBin( this->m_targetDepthMean - this->m_targetDepthSTD ), maxResponse ) )
    {
      result.score = maxResponse;
    }
  }

  return result;
}

ThreadResult OcclusionHandler::scoreCandidate( const std::array< cv::Mat, 2 > & frame, std::vector< cv::Point_< double > >::iterator candidate,float &candidateCenterDepth ) const
{
  cv::Mat patch[2];
  ThreadResult result;
  result.score = 0.0;
  result.value = candidate;

  std::vector< std::shared_ptr< FC > > features( 2 );
  std::vector< double > maxResponses;
  std::vector< Point > positions;
  Rect window = boundingBoxFromPointSize( *result.value, this->m_windowSize );

  tbb::parallel_for< uint >( 0, 2, 1, [this,&frame,&features,&window]( uint index ) -> void
	  {
		  features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
		  FC::mulFeatures( features[ index ], this->m_cosineWindow );
	  }
  );

  //Concat features and images here...
  features = this->m_featureProcessor->concatenate( features );
  std::vector< cv::Mat > frames_ = this->m_featureProcessor->concatenate( std::vector< cv::Mat >( frame.begin(), frame.end() ) );

  //Calculate the response of the target tracker at the candidate point
  for( uint i = 0; i < features.size(); i++ )
  {
    DetectResult detection = this->m_targetTracker[ i ]->detect( frames_[ i ], features[ i ], *result.value, this->m_targetDepthMean, this->m_targetDepthSTD );
    positions.push_back( detection.position );
    maxResponses.push_back( detection.maxResponse );
  }

  *result.value = this->m_featureProcessor->concatenate( positions );
  double maxResponse = this->m_featureProcessor->concatenate( maxResponses );

  if( getSubWindow( frame[ 0 ], patch[ 0 ], this->m_targetSize, *result.value ) && getSubWindow( frame[ 1 ], patch[ 1 ], this->m_targetSize, *result.value ) )
  {
    cv::Mat1b mask = createMask< ushort >( patch[ 1 ], 0 );
    DepthHistogram histogram = DepthHistogram::createHistogram( cvRound( modelNoise( this->m_targetDepthMean, this->m_targetDepthSTD ) ), patch[ 1 ], mask );
    std::vector< int > peaks = histogram.getPeaks();

	//modification here to fit the matlab version
	int peak = histogram.depthToPeak( (double)candidateCenterDepth, peaks );

    //Check to see if the response and depth are good enough to be the target objects
	if( ( peak <= 1 ) && this->evaluateVisibility( histogram, histogram.depthToBin( (double)candidateCenterDepth - this->m_targetDepthSTD ), maxResponse ) )
    {
      result.score = maxResponse;
    }
  }

  return result;
}

const bool ThreadResult::operator<( const ThreadResult & rval ) const
{
	return this->score < rval.score;
}