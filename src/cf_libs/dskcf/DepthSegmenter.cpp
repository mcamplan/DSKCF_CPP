#include <iostream>

#include "DepthSegmenter.hpp"
#include "math_helper.hpp"
#include "tbb/tick_count.h"

typedef cv::Size_< double > Size;
typedef cv::Rect_< double > Rect;
typedef cv::Point_< double > Point;

DepthSegmenter::DepthSegmenter()
{
	this->m_targetDepth = 0.0;
	this->m_targetSTD = 0.0;
	this->minSTD=20;
}

cv::Mat1i DepthSegmenter::init( const cv::Mat & image, const Rect & boundingBox )
{
	//Extract the target region of interest from the depth image
	Size windowSize = boundingBox.size();
	Point windowPosition = centerPoint( boundingBox );

	cv::Mat1w front_depth;
	if( getSubWindow( image, front_depth, windowSize, windowPosition  ) )
	{
		cv::Scalar mean, stddev;

		//Find and store the empty depth values to be excluded from the histogram
		cv::Mat1b mask = createMask( front_depth );

		//Create the histogram of depths in the region excluding the masked
		this->m_histogram = DepthHistogram::createHistogram( 50, front_depth, mask );

		//Find the peaks in the histogram
		std::vector< int > peaks = this->m_histogram.getPeaks( 5, 0.02 );

		//Group the points and label them
		this->labelsResults = this->m_histogram.getLabels( peaks );

		cv::Mat1i L = this->createLabelImageCC( front_depth, mask, this->labelsResults.centers, this->labelsResults.labels,this->labelsResults.labelsC);
		//Find the nearest object and calculate its mean depth and standard deviation
		int indexCenter=selectClosestObject( this->labelsResults.centers);
		cv::Mat objectMask = createMask< uchar >( L, this->labelsResults.labelsC[indexCenter], false );
		
		objectMask.mul( mask );
		cv::meanStdDev( front_depth, mean, stddev, objectMask );

		this->m_targetDepth = mean.val[ 0 ];
		if(stddev.val[ 0 ]< this->minSTD)
			stddev.val[ 0 ]= this->minSTD;

		this->m_targetSTD = stddev.val[ 0 ];
		cvCeil( modelNoise( this->m_targetDepth, this->m_targetSTD ) );

		return L;
	}

	return cv::Mat1i();
}

static int frame = 0;

int DepthSegmenter::update( const cv::Mat & image, const Rect & boundingBox )
{
	frame++;
	//Extract the target region of interest from the depth image
	Rect boundingBoxNEW=resizeBoundingBox( boundingBox, boundingBox.size() * 1.05 );//Rect boundingBoxNEW=boundingBox;
	Size windowSize = boundingBoxNEW.size();
	Point windowPosition = centerPoint( boundingBoxNEW );

	cv::Mat1w front_depth;
	if( getSubWindow( image, front_depth, windowSize, windowPosition  ) )
	{
		double minDepth, maxDepth;
		cv::Scalar mean, stddev;

		//Find and store the empty depth values to be excluded from the histogram
		cv::Mat1b mask = createMask( front_depth );

		//Create the histogram of depths in the region excluding the mask
    this->m_histogram = DepthHistogram::createHistogram( cvFloor( modelNoise( this->m_targetDepth, this->m_targetSTD) ), front_depth, mask );

		cv::minMaxLoc( front_depth, &minDepth, &maxDepth, nullptr, nullptr, mask );

		//Find the peaks in the histogram
    std::vector< int > peaks = this->m_histogram.getPeaks();

		bool emptyDepth=minDepth==0 && maxDepth==0;

		if( peaks.size() > 0 && (emptyDepth==false))
		{
			//Group the points and label them
			this->labelsResults = this->m_histogram.getLabels( peaks );
			this->m_labeledImage = this->createLabelImageCC( front_depth, mask, this->labelsResults.centers, this->labelsResults.labels,this->labelsResults.labelsC );
			if( maxDepth == minDepth )
			{
				maxDepth += 1;
			}

			int indexCloseCenter=selectClosestObject( this->labelsResults.centers);
			//Find the nearest object and calculate its mean depth and standard deviation
			cv::Mat1b objectMask = createMask< uchar >( this->m_labeledImage, this->labelsResults.labelsC[ indexCloseCenter ], false );
			objectMask.mul( mask );
			cv::meanStdDev( front_depth, mean, stddev, objectMask );

			int indexCenter= this->handleOcclusion(front_depth, this->labelsResults.centers,this->labelsResults.labelsC, this->m_targetDepth, this->m_targetSTD, mean.val[ 0 ], stddev.val[ 0 ] );
			//float centerDepth=(indexCenter>-1) ? this->labelsResults.centers[indexCenter] : this->labelsResults.centers.size()-1;
			float centerDepth = (indexCenter>-1) ? this->labelsResults.centers[indexCenter] : this->labelsResults.labels.size() - 1;
			//float centerDepthBUG = (indexCenter>-1) ? this->labelsResults.centers[indexCenter] : this->labelsResults.centers.size() - 1;
			int binCenter=cvRound(centerDepth);//the center are already in bin coordinates
			//int binCenterBUG = cvRound(centerDepthBUG);//the center are already in bin coordinates
			//if (binCenterBUG >= this->labelsResults.labels.size())
			//printf("binCenter: %d labelsSize: %d\n", binCenter, this->labelsResults.labels.size());

			int bin = std::find( this->labelsResults.labels.begin(), this->labelsResults.labels.end(), this->labelsResults.labels[binCenter] ) - this->labelsResults.labels.begin();
			//int bin = std::find(this->labelsResults.labels.begin(), this->labelsResults.labels.end(), this->labelsResults.labelsC[binCenter]) - this->labelsResults.labels.begin();
			int tmpBin=(this->m_histogram.depthToBin( this->getTargetDepth() - 1.5 * this->getTargetSTD() ));
			bin = std::min< int >( bin, tmpBin );

			return bin;
		}
	}

	return 0;
}

double DepthSegmenter::getTargetDepth() const
{
	return this->m_targetDepth;
}

double DepthSegmenter::getTargetSTD() const
{
	return this->m_targetSTD;
}

const DepthHistogram & DepthSegmenter::getHistogram() const
{
	return this->m_histogram;
}

const cv::Mat1b DepthSegmenter::createLabelImage( const cv::Mat1w & region, const cv::Mat1b mask,
                                                  const std::vector< float > & C, const std::vector< int > & labels ) const
{
	double min, max;
	cv::Mat1b L = cv::Mat1b::zeros( region.rows, region.cols );

	cv::minMaxLoc( region, &min, &max, nullptr, nullptr, mask );

	for( int x = 0; x < region.cols; x++ )
	{
		for( int y = 0; y < region.rows; y++ )
		{
			double depth = static_cast< double >( region( y, x ) );

			if( depth != 0.0 )
			{

				int index=this->m_histogram.depthToBin(depth);
				L( y, x ) = static_cast< uchar >( labels[ index ] );
			}
			else
			{
				L( y, x ) = static_cast< uchar >( C.size() );
			}
		}
	}

	return L;
}

const cv::Mat1b DepthSegmenter::createLabelImage( const cv::Mat1w & region, const cv::Mat1b mask,
                                                  const std::vector< float > & C, const std::vector< int > & labels,const DepthHistogram &histogram ) const
{
	double min, max;
	cv::Mat1b L = cv::Mat1b::zeros( region.rows, region.cols );

	cv::minMaxLoc( region, &min, &max, nullptr, nullptr, mask );

	for( int x = 0; x < region.cols; x++ )
	{
		for( int y = 0; y < region.rows; y++ )
		{
			double depth = static_cast< double >( region( y, x ) );

			if( depth != 0.0 )
			{

				int index=histogram.depthToBin(depth);
				L( y, x ) = static_cast< uchar >( labels[ index ] );
			}
			else
			{
				L( y, x ) = static_cast< uchar >( C.size() );
			}
		}
	}

	return L;
}

const cv::Mat1i DepthSegmenter::createLabelImageCC( const cv::Mat1w & region, const cv::Mat1b mask,
                                                   std::vector< float > & C,  const std::vector< int > & labels, std::vector< int > & labelsC, float smallAreaFraction)
{
	int minimumArea=   cvRound(smallAreaFraction*region.rows*region.cols);
	cv::Mat1b LnoCC = this->createLabelImage( region, mask, C, labels );
		cv::Mat L = cv::Mat::zeros(region.rows,region.cols,CV_32SC1);
		cv::Mat1b tmpResults = cv::Mat1b::zeros( region.rows, region.cols );
		cv::Mat statsCC, centroidsCC,Ltemp;

	//std::vector< cv::Rect_< int > > boxesTmp,boxesL;
	labelsC.clear();//std::vector<int> labelsCnew;
	this->areaRegions.clear();
	std::vector<float> centerNew;
	//set the iterator to the first element
	std::vector<float>::iterator itC = C.begin();
	int numElements=(int)C.size();
	int offset=0;
	
	//iterate for every label and eventually split it
	for (int i=0; i<numElements;i++){
	
		float center=C[i];
		cv::compare(LnoCC,i,tmpResults,CV_CMP_EQ);
	    //check if necessary to split the matrix....
		cv::connectedComponentsWithStats(tmpResults,Ltemp,statsCC,centroidsCC);
		int elementsToAdd=centroidsCC.rows-1; //not consider the black pixels...
		//cv::add(Ltemp,100,Ltemp);
		cv::add(Ltemp,offset,Ltemp,tmpResults);
		//cv::imwrite("C:/myDocs/variousSequences/dummyFolder/face_occ5/LtempNEW.png",Ltemp);
		cv::bitwise_or(L,Ltemp,L);
		//if necessary add the new labels and indexes
		for(int j=1; j<elementsToAdd+1;j++)
			{
			   int tmpArea=statsCC.at<int>(j, cv::CC_STAT_AREA);
			   this->areaRegions.push_back(tmpArea);
			   //eventually insert a dummy number for the small regions
			   ( tmpArea > minimumArea ? centerNew.push_back(center) : centerNew.push_back(1000000) );
			   labelsC.push_back(offset+j);
			}
		offset=(int)labelsC.size();
	}

	C=centerNew;

	Ltemp.release();
	tmpResults.release();
	LnoCC.release();

	return L;
}

const std::vector< cv::Point_< double > >  DepthSegmenter::createLabelImageCCOccluder( const cv::Mat1w & region, const cv::Mat1b mask,
                                                   std::vector< float > & C,  const std::vector< int > & labels, std::vector< int > & labelsC, 
												   const DepthHistogram &histogram,int minimumArea,cv::Mat1b &objectMask) const
														
{

	std::vector< Point > result;
	std::vector< Point > tmpVector;
	std::vector<int> areaVector;

	cv::Mat1b LnoCC = this->createLabelImage( region, mask, C, labels,histogram );
	cv::Mat L = cv::Mat::zeros(region.rows,region.cols,CV_32SC1);
	cv::Mat1b tmpResults = cv::Mat1b::zeros( region.rows, region.cols );
	cv::Mat statsCC, centroidsCC,Ltemp;

	labelsC.clear();
	std::vector<float> centerNew;
	//set the iterator to the first element
	std::vector<float>::iterator itC = C.begin();
	int numElements=(int)C.size();
	int offset=0;
	
	//iterate for every label and eventually split it
	for (int i=0; i<numElements;i++){
	

		float center=C[i];
		cv::compare(LnoCC,i,tmpResults,CV_CMP_EQ);

	    
	    //check if necessary to split the matrix....
		
		cv::connectedComponentsWithStats(tmpResults,Ltemp,statsCC,centroidsCC);

		int elementsToAdd=centroidsCC.rows-1; //not consider the black pixels...
		
		cv::add(Ltemp,offset,Ltemp,tmpResults);
		cv::bitwise_or(L,Ltemp,L);
		
		//if necessary add the new labels and indexes
		for(int j=1; j<elementsToAdd+1;j++)
			{
			   int tmpArea=statsCC.at<int>(j, cv::CC_STAT_AREA);
			   areaVector.push_back(tmpArea);
			   //eventually insert a dummy number for the small regions
			   if(tmpArea > minimumArea)
			   {
				tmpVector.push_back(cv::Point(centroidsCC.at<double>(j,0),centroidsCC.at<double>(j,1)));

				centerNew.push_back(center);
			   }
			   else{
				   centerNew.push_back(1000000) ;
				   tmpVector.push_back(cv::Point(-1,-1));
			   }

			   labelsC.push_back(offset+j);
			}

		offset=(int)labelsC.size();

	}

	//now exclude from the list also the closest object....as it is the occluder
	int indexCenter=this->selectClosestObject(centerNew,areaVector);
	tmpVector[indexCenter].x=-1;
	tmpVector[indexCenter].y=-1;

	int indexLabel=labelsC[indexCenter];
	cv::Mat tmpMask =createMask< uchar >( L, indexLabel, false );
	tmpMask.copyTo(objectMask);
	tmpMask.release();

	for(int j=0; j<tmpVector.size();j++)
	{
		if(tmpVector[j].x!=-1)
		{
			result.push_back(tmpVector[j]);
		}
	}

	C=centerNew;

	Ltemp.release();
	tmpResults.release();
	LnoCC.release();
	L.release();
	return result;
}

const std::vector< cv::Point_< double > >  DepthSegmenter::createLabelImageCCOccluder( const cv::Mat1w & region, const cv::Mat1b mask,
                                                   std::vector< float > & C,  const std::vector< int > & labels, std::vector< int > & labelsC, 
												   const DepthHistogram &histogram,int minimumArea,cv::Mat1b &objectMask,std::vector< float > & centroidsCandidates,
												   cv::Rect_<double> &occluderRect) const
														
{

	std::vector< Point > result;
	std::vector< Point > tmpVector;
	std::vector<int> areaVector;
	std::vector< cv::Rect_<double> > rectVector;

	cv::Mat1b LnoCC = this->createLabelImage( region, mask, C, labels,histogram );
	cv::Mat L = cv::Mat::zeros(region.rows,region.cols,CV_32SC1);
	cv::Mat1b tmpResults = cv::Mat1b::zeros( region.rows, region.cols );
	cv::Mat statsCC, centroidsCC,Ltemp;

	labelsC.clear();//std::vector<int> labelsCnew;
	std::vector<float> centerNew;
	//set the iterator to the first element
	std::vector<float>::iterator itC = C.begin();
	int numElements=(int)C.size();
	int offset=0;
	
	//iterate for every label and eventually split it
	for (int i=0; i<numElements;i++){
	

		float center=C[i];
		cv::compare(LnoCC,i,tmpResults,CV_CMP_EQ);

	    
	    //check if necessary to split the matrix....
		
		cv::connectedComponentsWithStats(tmpResults,Ltemp,statsCC,centroidsCC);

		int elementsToAdd=centroidsCC.rows-1; //not consider the black pixels...
		
		cv::add(Ltemp,offset,Ltemp,tmpResults);
		cv::bitwise_or(L,Ltemp,L);
		
		//if necessary add the new labels and indexes
		for(int j=1; j<elementsToAdd+1;j++)
			{
			   int tmpArea=statsCC.at<int>(j, cv::CC_STAT_AREA);
			   areaVector.push_back(tmpArea);
			   rectVector.push_back(Rect(statsCC.at<int>(j, cv::CC_STAT_LEFT),statsCC.at<int>(j, cv::CC_STAT_TOP),
			   statsCC.at<int>(j, cv::CC_STAT_WIDTH),statsCC.at<int>(j, cv::CC_STAT_HEIGHT)));
			   //eventually insert a dummy number for the small regions
			   if(tmpArea > minimumArea)
			   {
				tmpVector.push_back(cv::Point(centroidsCC.at<double>(j,0),centroidsCC.at<double>(j,1)));

				centerNew.push_back(center);
				centroidsCandidates.push_back(histogram.binToDepth(center));
			   }
			   else{
				   centerNew.push_back(1000000) ;
				   tmpVector.push_back(cv::Point(-1,-1));
			   }

			   labelsC.push_back(offset+j);
			}

		offset=(int)labelsC.size();

	}

	//now exclude from the list also the closest object....as it is the occluder
	int indexCenter=this->selectClosestObject(centerNew,areaVector);
	tmpVector[indexCenter].x=-1;
	tmpVector[indexCenter].y=-1;
	occluderRect=rectVector[indexCenter];
	

	int indexLabel=labelsC[indexCenter];
	cv::Mat tmpMask =createMask< uchar >( L, indexLabel, false );
	tmpMask.copyTo(objectMask);
	tmpMask.release();

	for(int j=0; j<tmpVector.size();j++)
	{
		if(tmpVector[j].x!=-1)
		{
			result.push_back(tmpVector[j]);
		}
	}

	C=centerNew;

	Ltemp.release();
	tmpResults.release();
	LnoCC.release();
	L.release();
	return result;
}




const int DepthSegmenter::selectClosestObject( std::vector< float > & centroids) {


	std::vector<int>::iterator result;

	int numElements=(int)centroids.size();
	int returnedIndex=numElements-1;

	for (int i=0; i<numElements-1;i++){

		//if it is not a smaller region, this is the closer object
		if(centroids[i]<1000000){
			//select the interval
			int tmpIndex=i;
			for (int j=i+1; j < numElements;j++)
			{
				if (centroids[j]>centroids[i] && centroids[j]<1000000)
				{
					tmpIndex=j;
					break;
				}

				else{
				  tmpIndex++;
				}

			}

			result = std::max_element(this->areaRegions.begin()+i,this->areaRegions.begin()+tmpIndex);
			returnedIndex=std::distance(this->areaRegions.begin(), result);

			break;
		}
	}

	return returnedIndex;
}

const int DepthSegmenter::selectClosestObject( std::vector< float > & centroids, std::vector< int > & areaVector) const{


	std::vector<int>::const_iterator result;

	int numElements=(int)centroids.size();
	int returnedIndex=numElements-1;

	for (int i=0; i<numElements-1;i++){

		//if it is not a smaller region, this is the closer object
		if(centroids[i]<1000000){
			//select the interval
			int tmpIndex=i;
			for (int j=i+1; j < numElements;j++)
			{
				if (centroids[j]>centroids[i] && centroids[j]<1000000)
				{
					tmpIndex=j;
					break;
				}

				else{
				  tmpIndex++;
				}

			}

			result = std::max_element(areaVector.begin()+i,areaVector.begin()+tmpIndex);
			returnedIndex=std::distance<std::vector<int>::const_iterator>(areaVector.begin(), result);

			break;
		}
	}

	return returnedIndex;
}

const cv::Mat1i & DepthSegmenter::getLabeledImage() const
{
	return this->m_labeledImage;
}

const std::vector<int> & DepthSegmenter::getAreaRegions() const{
	return this->areaRegions;
}
	
const DepthHistogram::Labels &DepthSegmenter::getLabelsResults() const{
	return this->labelsResults;
}  



const int DepthSegmenter::handleOcclusion(
    const std::vector< float > & centroids, const double previousDepth,
    const double previousSTD, const double targetDepth, const double targetSTD )
{
	int minIndex = 0;
	double minDistance;
	std::vector< float > peakDistances = centroids;

  double max = this->m_histogram.size();

	for( auto itr = peakDistances.begin(); itr != peakDistances.end(); itr++ )
	{
    *itr = std::abs( this->m_histogram.binToDepth( *itr ) - previousDepth );
	}

  minDistance = peakDistances[ 0 ];

	for( size_t i = 1; i < peakDistances.size(); i++ )
	{
		if( peakDistances[ i ] < peakDistances[ minIndex ] )
		{
			minIndex = static_cast< int >( i );
			minDistance = peakDistances[ i ];
		}
	}

  //%register the plane index when you filtered out some small
	if( ( minIndex == 0 ) && ( minDistance < 3.0 * previousSTD ) )
	{
		this->m_occluded = false;
    //everything seems ok....no occluding object, just a movement
    //of the object....update the depth!!!
		this->m_targetDepth = targetDepth;
		this->m_targetSTD = targetSTD;
	}
	else
	{
    //// THERE IS AN OCCLUSION......WHAT TO DO?
    //find the new corresponding region (if exist) and calculate
		if( minDistance < 2.5 * previousSTD )
		{
			this->m_occluded = true;
			this->m_targetDepth = this->m_histogram.binToDepth(centroids[ minIndex ]);
			this->m_targetSTD = targetSTD;

			if( this->m_targetSTD < this->minSTD )
			{
				this->m_targetSTD = previousSTD;
			}
		}
		else
		{
			this->m_occluded = false;
			this->m_targetDepth = targetDepth;
			this->m_targetSTD = targetSTD;
		}
	}

	return minIndex;
}

const int DepthSegmenter::handleOcclusion(
    const cv::Mat& front_depth, const std::vector< float > & centroids,const std::vector< int > & labelsC, const double previousDepth,
    const double previousSTD, const double targetDepth, const double targetSTD )
{
	int minIndex = 0;
	double minDistance;
	std::vector< float > peakDistances = centroids;

	int minNonZero=0;
	bool foundMin=false;

	double max = this->m_histogram.size();

	for( auto itr = peakDistances.begin(); itr != peakDistances.end(); itr++ )
	{
		if(*itr>=(1000000-1) && foundMin==false)
			minNonZero++;
		else
			foundMin=true;

    *itr = std::abs( this->m_histogram.binToDepth( *itr ) - previousDepth );
	}

  minDistance = peakDistances[ 0 ];

	for( size_t i = 1; i < peakDistances.size(); i++ )
	{
		if( peakDistances[ i ] < peakDistances[ minIndex ] )
		{
			minIndex = static_cast< int >( i );
			minDistance = peakDistances[ i ];
		}
	}

  //%register the plane index when you filtered out some small
  //%regions....
	if( ( minIndex == minNonZero ) && ( minDistance < 3.0 * previousSTD ) )
	{
		this->m_occluded = false;
    //everything seems ok....no occluding object, just a movement
    //of the object....update the depth!!!
		this->m_targetDepth = targetDepth;
		this->m_targetSTD = targetSTD;
	}
	else
	{
    //// THERE IS AN OCCLUSION......WHAT TO DO?
    //find the new corresponding region (if exist) and calculate
		if( minDistance < 2.5 * previousSTD )
		{
			this->m_occluded = true;
			//this->m_targetDepth = ( ( ( centroids[ minIndex ] / max ) * ( this->m_histogram.maximum() - this->m_histogram.minimum() ) ) + this->m_histogram.minimum() );
			this->m_targetDepth = this->m_histogram.binToDepth(centroids[ minIndex ]);
			//WRONG!!! NEED TO BE RECALCULATED....this->m_targetSTD = targetSTD;
			int indexLabel=labelsC[minIndex];
			cv::Mat1b objectMask = createMask< uchar >( this->m_labeledImage, indexLabel, false );
			cv::Scalar mean, stddev;
			cv::meanStdDev( front_depth, mean, stddev, objectMask );
			
			objectMask.release();
			this->m_targetSTD = stddev.val[ 0 ];
			if( this->m_targetSTD < this->minSTD )
			{
				this->m_targetSTD = previousSTD;
			}
		}
		else
		{
			this->m_occluded = false;
			this->m_targetDepth = previousDepth;
			this->m_targetSTD = previousSTD;
			minIndex=-1;
		}
	}

	return minIndex;
}


const bool DepthSegmenter::isOccluded() const
{
	return this->m_occluded;
}

//debug function to save histogram
void DepthSegmenter::debugSaveHistogram(std::string filename){
	FILE *pfile=fopen(filename.c_str(),"w");
	
	for(int i=0; i<this->m_histogram.size();i++)
		fprintf(pfile,"%f %f\n",this->m_histogram.binToDepth(i),this->m_histogram[i]);

	fclose(pfile);
}


const cv::Mat1b DepthSegmenter::segment( const cv::Mat1w & frame, const cv::Rect_< double > & boundingBox ) const
{
	//Extract the target region of interest from the depth image
	Size windowSize = boundingBox.size();
	Point windowPosition = centerPoint( boundingBox );

	cv::Mat1w front_depth;
	if( getSubWindow( frame, front_depth, windowSize, windowPosition  ) )
	{
		double minDepth, maxDepth;

		//Find and store the empty depth values to be excluded from the histogram
		cv::Mat1b mask = createMask(front_depth);

		//Create the histogram of depths in the region excluding the mask
		DepthHistogram histogram = DepthHistogram::createHistogram( cvCeil( modelNoise( this->m_targetDepth, this->m_targetSTD) ), front_depth, mask );

		cv::minMaxLoc( front_depth, &minDepth, &maxDepth, nullptr, nullptr, mask );

		//Find the peaks in the histogram
		int minimumPeakDistance = ( histogram.size() < 50 ) ? 1 : 3;
		std::vector< int > peaks = histogram.getPeaks( minimumPeakDistance, 0.005 );

		if( peaks.size() > 0 )
		{
			//Group the points and label them
			DepthHistogram::Labels labels = histogram.getLabels( peaks );
			return this->createLabelImage( front_depth, mask, labels.centers, labels.labels,histogram );
		}
	}

	return cv::Mat1b();
}
														
const std::vector< cv::Point_< double > > DepthSegmenter::segmentOccluder( const cv::Mat1w & frame, const cv::Rect_< double > & boundingBox,const int minimumArea,cv::Mat1b &objectMask) const
{
	std::vector< Point > result;
	//Extract the target region of interest from the depth image
	Size windowSize = boundingBox.size();
	Point windowPosition = centerPoint( boundingBox );

	cv::Mat1w front_depth;
	if( getSubWindow( frame, front_depth, windowSize, windowPosition  ) )
	{
		double minDepth, maxDepth;

		//Find and store the empty depth values to be excluded from the histogram
		cv::Mat1b mask = createMask(front_depth);

		//Create the histogram of depths in the region excluding the mask
		DepthHistogram histogram = DepthHistogram::createHistogram( cvCeil( modelNoise( this->m_targetDepth, this->m_targetSTD) ), front_depth, mask );

		cv::minMaxLoc( front_depth, &minDepth, &maxDepth, nullptr, nullptr, mask );

		//Find the peaks in the histogram
		int minimumPeakDistance = ( histogram.size() < 50 ) ? 1 : 3;
		std::vector< int > peaks = histogram.getPeaks( minimumPeakDistance, 0.005 );

		if( peaks.size() > 0 )
		{
			//Group the points and label them
			DepthHistogram::Labels labels = histogram.getLabels( peaks );
			result= this->createLabelImageCCOccluder( front_depth, mask, labels.centers, labels.labels,labels.labelsC,histogram,minimumArea,objectMask);
			
			return result;
		}
	}

	return result;
}

                                                          
const std::vector< cv::Point_< double > > DepthSegmenter::segmentOccluder( const cv::Mat1w & frame, const cv::Rect_< double > & boundingBox,const int minimumArea,cv::Mat1b &objectMask,
		std::vector< float > &centersCandidate,cv::Rect_<double> &occluderRect ) const
{
	std::vector< Point > result;
	//Extract the target region of interest from the depth image
	Size windowSize = boundingBox.size();
	Point windowPosition = centerPoint( boundingBox );

	cv::Mat1w front_depth;
	if( getSubWindow( frame, front_depth, windowSize, windowPosition  ) )
	{
		double minDepth, maxDepth;

		//Find and store the empty depth values to be excluded from the histogram
		cv::Mat1b mask = createMask(front_depth);

		//Create the histogram of depths in the region excluding the mask
		DepthHistogram histogram = DepthHistogram::createHistogram( cvCeil( modelNoise( this->m_targetDepth, this->m_targetSTD) ), front_depth, mask );

		cv::minMaxLoc( front_depth, &minDepth, &maxDepth, nullptr, nullptr, mask );

		//Find the peaks in the histogram
		int minimumPeakDistance = ( histogram.size() < 50 ) ? 1 : 3;
		std::vector< int > peaks = histogram.getPeaks( minimumPeakDistance, 0.005 );
		bool emptyHIST = (minDepth == 0 && maxDepth == 0);
		if (peaks.size() > 0 &&  emptyHIST==false)
		{
			//Group the points and label them
			DepthHistogram::Labels labels = histogram.getLabels( peaks );
			result= this->createLabelImageCCOccluder( front_depth, mask, labels.centers, labels.labels,labels.labelsC,histogram,minimumArea,objectMask,centersCandidate,occluderRect);
			
			return result;
		}
	}

	return result;
}