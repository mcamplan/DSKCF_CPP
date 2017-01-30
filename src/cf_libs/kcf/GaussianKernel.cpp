#include "GaussianKernel.hpp"

GaussianKernel::GaussianKernel()
{
	this->sigma = 0.5;
}

cv::Mat GaussianKernel::correlation(const std::shared_ptr< FC > & xf, const std::shared_ptr< FC > & yf) const
{
	double xx, yy, numel;
	cv::Mat xy, kf;
	std::shared_ptr< FC > xyf, realXy;

	if (xf == yf)
	{
		yy = xx = FC::squaredNormFeaturesNoCcs(xf);
	}
	else
	{
		xx = FC::squaredNormFeaturesNoCcs(xf);
		yy = FC::squaredNormFeaturesNoCcs(yf);
	}

	xyf = FC::mulSpectrumsFeatures(xf, yf, true);
	realXy = FC::idftFeatures(xyf);

	xy = FC::sumFeatures(realXy);

	numel = static_cast<double>(xf->channels[0].total() * xf->numberOfChannels());
	this->calculateGaussianTerm(xy, numel, xx, yy);

	dft(xy, kf, cv::DFT_COMPLEX_OUTPUT);

	return kf;
}

void GaussianKernel::calculateGaussianTerm(cv::Mat & xy, double numel, double xx, double yy) const
{
	int width = xy.cols;
	int height = xy.rows;

	width *= height;
	height = 1;

	const double summands = xx + yy;
	const double fraction = -1 / (this->sigma * this->sigma);

	for (int row = 0; row < height; ++row)
	{
		double* xyd = xy.ptr< double >(row);

		for (int col = 0; col < width; ++col)
		{
			xyd[col] = (summands - 2 * xyd[col]) / numel;

			if (xyd[col] < 0)
			{
				xyd[col] = 0;
			}

			xyd[col] *= fraction;
			xyd[col] = exp(xyd[col]);
		}
	}
}
