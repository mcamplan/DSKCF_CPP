#include "kcf_tracker.hpp"

KcfParameters::KcfParameters()
{
  this->padding = 2.5;
  this->lambda = 0.0001;
  this->outputSigmaFactor = 0.05;
  this->interpFactor = 0.02;
  this->cellSize = 4;
}

KcfTracker::KcfTracker(KcfParameters paras, std::shared_ptr< Kernel > kernel ) :
    m_xf( nullptr ),
    m_kernel( kernel )
{
  m_isInitialized = false;
  m_lambda = paras.lambda;
  m_interpFactor = paras.interpFactor;
  m_cellSize = paras.cellSize;
  m_frameID = 1;
}

KcfTracker::~KcfTracker()
{
}

void KcfTracker::init( const cv::Mat & image, const std::shared_ptr< FC > & features, const cv::Point_< double > & position )
{
  TrainingData trainingData = getTrainingData( image, features );

  this->m_frameID = 0;
  this->m_isInitialized = false;

  divideSpectrumsNoCcs< double >( trainingData.numeratorf, trainingData.denominatorf, this->m_alphaf );
  this->m_alphaNumeratorf = trainingData.numeratorf;
  this->m_alphaDenominatorf = trainingData.denominatorf;
  this->m_xf = trainingData.xf;
  this->m_isInitialized = true;
}

const KcfTracker::TrainingData KcfTracker::getTrainingData( const cv::Mat & image, const std::shared_ptr< FC > & features ) const
{
  TrainingData result;
  result.xf = FC::dftFeatures( features, cv::DFT_COMPLEX_OUTPUT );
  cv::Mat kf = this->m_kernel->correlation( result.xf, result.xf );
  cv::Mat kfLambda = kf + this->m_lambda;
  mulSpectrums( this->m_yf, kf, result.numeratorf, 0 );
  mulSpectrums( kf, kfLambda, result.denominatorf, 0 );

  return result;
}

void KcfTracker::update( const cv::Mat & image, const std::shared_ptr< FC > & features, const cv::Point_< double > & position )
{
  ++m_frameID;

  if( m_isInitialized )
  {
    this->updateModel( image, features );
  }
}

const DetectResult KcfTracker::detectModel( const cv::Mat & image, const std::shared_ptr< FC > & features, const Point & newPos ) const
{
  DetectResult result;
  Response newResponse = this->getResponse( image, features, newPos );

  cv::Point_< double > subDelta = subPixelDelta< double >( newResponse.response, newResponse.maxResponsePosition );

  if( subDelta.y >= newResponse.response.rows / 2 )
  {
    subDelta.y -= newResponse.response.rows;
  }
  if( subDelta.x >= newResponse.response.cols / 2 )
  {
    subDelta.x -= newResponse.response.cols;
  }
  double posDeltaX = m_cellSize * subDelta.x;
  double posDeltaY = m_cellSize * subDelta.y;
  result.position.x = newPos.x + posDeltaX;
  result.position.y = newPos.y + posDeltaY;

  result.maxResponse = newResponse.maxResponse;

  return result;
}

void KcfTracker::updateModel( const cv::Mat & image, const std::shared_ptr< FC > & features )
{
  TrainingData trainingData = getTrainingData( image, features );

  this->m_alphaNumeratorf   = ( 1 - m_interpFactor ) * m_alphaNumeratorf   + m_interpFactor * trainingData.numeratorf;
  this->m_alphaDenominatorf = ( 1 - m_interpFactor ) * m_alphaDenominatorf + m_interpFactor * trainingData.denominatorf;

  FC::mulValueFeatures( this->m_xf, ( 1 - this->m_interpFactor ) );
  FC::mulValueFeatures( trainingData.xf, this->m_interpFactor );
  FC::addFeatures( this->m_xf, trainingData.xf );
  divideSpectrumsNoCcs< double >( m_alphaNumeratorf, m_alphaDenominatorf, this->m_alphaf );
}

const KcfTracker::Response KcfTracker::getResponse( const cv::Mat & image, const std::shared_ptr< FC > & features, const Point & pos ) const
{
  Response result;

  result.response = this->detectResponse( image, features, pos );
  minMaxLoc( result.response, 0, &result.maxResponse, 0, &result.maxResponsePosition );

  return result;
}

const cv::Mat KcfTracker::detectResponse( const cv::Mat & image, const std::shared_ptr< FC > & features, const Point & pos ) const
{
  cv::Mat responsef;
  cv::Mat1d response;

  std::shared_ptr< FC > zf = FC::dftFeatures( features, cv::DFT_COMPLEX_OUTPUT );
  cv::Mat kzf = this->m_kernel->correlation( zf, this->m_xf );

  mulSpectrums( this->m_alphaf, kzf, responsef, 0, false );
  idft( responsef, response, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE );

  return response;
}

const DetectResult KcfTracker::detect( const cv::Mat & image, const std::shared_ptr< FC > & features, const Point & position ) const
{
  return this->detectModel( image, features, position );
}

void KcfTracker::onScaleChange( const Size & targetSize, const Size & windowSize, const cv::Mat2d & yf, const cv::Mat1d & cosineWindow )
{
  this->m_cosineWindow = cosineWindow;
  this->m_yf = yf;

  //If the model is initialised, resize it
  if( this->m_isInitialized )
  {
    cv::Size2i modelSize(
        cvFloor( windowSize.width / this->m_cellSize ),
        cvFloor( windowSize.height / this->m_cellSize )
    );

    tbb::parallel_for_each( this->m_xf->channels.begin(), this->m_xf->channels.end(),
      [modelSize]( cv::Mat & channel )
      {
        channel = ScaleAnalyser::scaleImageFourierShift( channel, modelSize );
      }
    );

    this->m_alphaNumeratorf = ScaleAnalyser::scaleImageFourierShift( this->m_alphaNumeratorf, modelSize );
    this->m_alphaDenominatorf = ScaleAnalyser::scaleImageFourierShift( this->m_alphaDenominatorf, modelSize );
  }
}

std::shared_ptr< KcfTracker > KcfTracker::duplicate() const
{
  std::shared_ptr< KcfTracker > result = std::make_shared< KcfTracker >( KcfParameters(), this->m_kernel );

  result->m_xf = this->m_xf;
  result->m_alphaNumeratorf = this->m_alphaNumeratorf;
  result->m_alphaDenominatorf = this->m_alphaDenominatorf;
  result->m_alphaf = this->m_alphaf;
  result->m_cosineWindow = this->m_cosineWindow;
  result->m_yf = this->m_yf;

  result->m_frameID = this->m_frameID;
  result->m_isInitialized = this->m_isInitialized;

  result->m_lambda = this->m_lambda;
  result->m_interpFactor = this->m_interpFactor;
  result->m_cellSize = this->m_cellSize;

  result->m_kernel = this->m_kernel;

  return result;
}
