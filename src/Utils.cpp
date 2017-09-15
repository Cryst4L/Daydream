#include <Daydream/Utils.h>

#include <iostream>

using namespace Eigen;

namespace Daydream
{
/*inline*/ double sigmoid(const double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

/*inline*/ double softplus(const double x)
{
	return std::log(1.0 + std::exp(x));
} 

/*inline*/ double logit(const double x)
{
	return std::log((1.0 / 1.0 - x) - 1.0);
} 

VectorXd binomialSampling(const VectorXd& sample)
{
	int n = sample.size();
	ArrayXd random = .5 * ArrayXd::Random(n) + 0.5;
	return (random < sample.array()).cast <double> ();
}
}
