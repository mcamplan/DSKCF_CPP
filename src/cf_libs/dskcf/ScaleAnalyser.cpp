#include <iostream>

#include "ScaleAnalyser.hpp"
#include "kcf_tracker.hpp"

typedef cv::Rect_< double > Rect;

ScaleAnalyser::ScaleAnalyser( DepthSegmenter * depthSegmenter, double padding )
{
	double step = 0.1;

	this->m_depthSegmenter = depthSegmenter;
	this->m_scales.resize( 19 );

	for( size_t i = 0; i < this->m_scales.size(); i++ )
	{
		this->m_scales[ i ] = 0.4 + ( i * step );
	}

	this->m_padding = padding;
	this->m_step = step;
	this->m_minStep = std::abs( this->m_step );
	this->m_i = 6;
	this->m_initialDepth = this->m_currentDepth = 1.0;
	this->m_cellSize = 4;
	this->m_outputSigmaFactor = 0.1;
	this->m_scaleFactor = 1.0;

	//Pre-allocate all the memory we need
	this->m_windowSizes.resize( this->m_scales.size() );
	this->m_targetSizes.resize( this->m_scales.size() );
	this->m_targetPositions.resize( this->m_scales.size() );
	this->m_outputSigmas.resize( this->m_scales.size() );
	this->m_yfs.resize( this->m_scales.size() );
	this->m_cosineWindows.resize( this->m_scales.size() );
}

ScaleAnalyser::ScaleAnalyser( const std::vector< double > & scales, const double outputSigmaFactor, const int cellSize, double padding )
{
	this->m_scales = scales;
	this->m_step = 10000.0;

	for( size_t i = 1; i < this->m_scales.size(); i++ )
	{
		this->m_step = std::min< double >(
			this->m_step,
			std::abs( this->m_scales[ i ] - this->m_scales[ i - 1 ] )
		);
	}

	this->m_padding = padding;
	this->m_minStep = this->m_step;
	this->m_initialDepth = this->m_currentDepth = 1.0;
	this->m_cellSize = cellSize;
	this->m_outputSigmaFactor = outputSigmaFactor;
	this->m_scaleFactor = 1.0;

	this->m_windowSizes.resize( scales.size() );
	this->m_targetSizes.resize( scales.size() );
	this->m_targetPositions.resize( scales.size() );
	this->m_outputSigmas.resize( scales.size() );
	this->m_yfs.resize( scales.size() );
	this->m_cosineWindows.resize( scales.size() );
}

Rect ScaleAnalyser::init( const cv::Mat &, const Rect & boundingBox )
{
	if( ( boundingBox.width > 0 ) && ( boundingBox.height > 0 ) )
	{
		boundingBox.size();
		this->m_currentDepth = this->m_initialDepth = this->m_depthSegmenter->getTargetDepth();
		this->m_scaleFactor = 1.0;

		for( size_t i = 0; i < this->m_scales.size(); i++ )
		{
			this->m_targetPositions[ i ] = centerPoint( boundingBox );
			this->m_targetSizes[ i ] = sizeRound( boundingBox.size() * ( this->m_scales[ i ] ) );
			this->m_windowSizes[ i ] = sizeRound( this->m_targetSizes[ i ] * this->m_padding );

			this->m_outputSigmas[ i ] = sqrt( this->m_targetSizes[ i ].area() ) * this->m_outputSigmaFactor / this->m_cellSize;

			cv::dft(
				gaussianShapedLabelsShifted2D(
					this->m_outputSigmas[ i ],
					sizeFloor( this->m_windowSizes[ i ] * ( 1.0 / static_cast< double >( this->m_cellSize ) ) )
				),
				this->m_yfs[ i ], cv::DFT_COMPLEX_OUTPUT
			);

			this->m_cosineWindows[ i ] =
				hanningWindow< double >( this->m_yfs[ i ].rows ) *
				hanningWindow< double >( this->m_yfs[ i ].cols ).t();

			if( this->m_scales[ i ] == 1.0 )
			{
				this->m_i = i;
			}
		}

		for( auto itr = this->m_observers.begin(); itr != this->m_observers.end(); itr++ )
		{
			(*itr)->onScaleChange(
				this->m_targetSizes[ this->m_i ],
				this->m_windowSizes[ this->m_i ],
				this->m_yfs[ this->m_i ],
				this->m_cosineWindows[ this->m_i ]
			);
		}
	}

	return boundingBox;
}

Rect ScaleAnalyser::update( const cv::Mat & image, const Rect & boundingBox )
{
	this->m_currentDepth = this->m_depthSegmenter->getTargetDepth();
	double sf = this->m_initialDepth / this->m_currentDepth;
#ifdef _WIN32
  if( _finite( sf ) )
#elif __linux
	if( std::isfinite( sf ) )
#endif
	{
		this->m_scaleFactor = sf;
	}

	double scaleOffset = this->m_scaleFactor - this->m_scales[ this->m_i ];

	if( std::abs( scaleOffset ) > this->m_minStep )
	{
		if( ( scaleOffset < 0 ) && ( this->m_i > 0 ) )
		{
			size_t ind = 0;
			std::vector< double > diffs(
				this->m_scales.begin(),
				this->m_scales.begin() + this->m_i
			);

			for( size_t i = 0; i < diffs.size(); i++ )
			{
				diffs[ i ] = std::abs( diffs[ i ] - this->m_scaleFactor );
			}

			double a = *std::min_element( diffs.begin(), diffs.end() );

			for( size_t i = 0; i < diffs.size(); i++ )
			{
				if( diffs[ i ] == a )
				{
					ind = i;
				}
			}

			if( this->m_i != ind )
			{
				this->m_i = ind;

				for( auto itr = this->m_observers.begin(); itr != this->m_observers.end(); itr++ )
				{
					(*itr)->onScaleChange(
						this->m_targetSizes[ this->m_i ],
						this->m_windowSizes[ this->m_i ],
						this->m_yfs[ this->m_i ],
						this->m_cosineWindows[ this->m_i ]
					);
				}
			}
		}
		else if( ( scaleOffset > 0 ) && ( this->m_i < this->m_scales.size() ) )
		{
			size_t ind = 0;
			std::vector< double > diffs(
				this->m_scales.begin() + this->m_i,
				this->m_scales.end()
			);

			for( size_t i = 0; i < diffs.size(); i++ )
			{
				diffs[ i ] = std::abs( diffs[ i ] - this->m_scaleFactor );
			}

			double a = *std::min_element( diffs.begin(), diffs.end() );

			for( size_t i = 0; i < diffs.size(); i++ )
			{
				if( diffs[ i ] == a )
				{
					ind = i;
				}
			}

			ind = ind + this->m_i - 1;

			if( this->m_i != ind )
			{
				this->m_i = ind;

				for( auto itr = this->m_observers.begin(); itr != this->m_observers.end(); itr++ )
				{
					(*itr)->onScaleChange(
						this->m_targetSizes[ this->m_i ], this->m_windowSizes[ this->m_i ],
						this->m_yfs[ this->m_i ], this->m_cosineWindows[ this->m_i ]
					);
				}
			}
		}
	}

	return boundingBox;
}

cv::Mat2d ScaleAnalyser::scaleImageFourier( const cv::Mat2d & image_f, const cv::Size2i & size )
{
	if(
		( image_f.size().width != size.width ) ||
		( image_f.size().height != size.height )
	)
	{
		cv::Mat result;
		cv::Rect2i inputRect, outputRect;
		cv::Size2i inputSize, outputSize;

		inputSize = image_f.size();
		outputSize = size;

		inputRect.height = std::min( image_f.rows, size.height );
		inputRect.width = std::min( image_f.cols, size.width );
		outputRect.width = inputRect.width;
		outputRect.height = inputRect.height;

		if(
			(
				( inputSize.width % 2 == 0 ) &&
				( outputSize.width > inputSize.width )
			) ||
			(
				( inputSize.width % 2 == 1 ) &&
				( outputSize.width < inputSize.width )
			)
		)
		{
			inputRect.x = cvCeil( ( image_f.cols - inputRect.width ) / 2.0 );
			outputRect.x = cvCeil( ( size.width - outputRect.width ) / 2.0 );
		}
		else
		{
			inputRect.x = cvFloor( ( image_f.cols - inputRect.width ) / 2.0 );
			outputRect.x = cvFloor( ( size.width - outputRect.width ) / 2.0 );
		}

		if(
			(
				( inputSize.height % 2 == 0 ) &&
				( outputSize.height > inputSize.height )
			) ||
			(
				( inputSize.height % 2 == 1 ) &&
				( outputSize.height < inputSize.height )
			)
		)
		{
			inputRect.y = cvCeil( ( image_f.rows - inputRect.height ) / 2.0 );
			outputRect.y = cvCeil( ( size.height - outputRect.height ) / 2.0 );
		}
		else
		{
			inputRect.y = cvFloor( ( image_f.rows - inputRect.height ) / 2.0 );
			outputRect.y = cvFloor( ( size.height - outputRect.height ) / 2.0 );
		}

		double gainx = (
			static_cast< double >( outputRect.width ) /
			static_cast< double >( inputRect.width )
		);
		double gainy = (
			static_cast< double >( outputRect.height ) /
			static_cast< double >( inputRect.height )
		);

		int sizes[] = { size.height, size.width };
		result = cv::Mat::zeros( image_f.dims, sizes, image_f.type() );

		cv::Mat input( image_f, inputRect );
		cv::Mat output( result, outputRect );

		input.copyTo( output );

		return gainx * gainy * result;
	}
	else
	{
		return image_f;
	}
}

cv::Mat2d ScaleAnalyser::scaleImageFourierShift( const cv::Mat2d & image, const cv::Size2i & size )
{
	return ifftshift(
		ScaleAnalyser::scaleImageFourier( fftshift( image ), size )
	);
}

double ScaleAnalyser::getScaleFactor() const
{
	return this->m_scaleFactor;
}

void ScaleAnalyser::registerScaleChangeObserver( ScaleChangeObserver * observer )
{
	this->m_observers.push_back( observer );
}

void ScaleAnalyser::clearObservers()
{
	this->m_observers.clear();
}

std::vector< std::shared_ptr< KcfTracker > > ScaleAnalyser::createModelScales( std::shared_ptr< KcfTracker > tracker )
{
	std::vector< std::shared_ptr< KcfTracker > > result( this->m_targetSizes.size() );

	for( size_t i = 0; i < this->m_targetSizes.size(); i++ )
	{
		result[ i ] = tracker->duplicate();
		result[ i ]->onScaleChange( this->m_targetSizes[ i ], this->m_windowSizes[ i ], this->m_yfs[ i ], this->m_cosineWindows[ i ] );
	}

	return result;
}
