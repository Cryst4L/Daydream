#pragma once

#include <Eigen/Core>

namespace Daydream
{
	//! Sigmoid (also known as logistic) activation function.
	/*inline*/ double sigmoid(const double x);
	//! SoftPlus activation, equals the derivative of the sigmoid.
	/*inline*/ double softplus(const double x);
	//! Logit activation, equals the the inverse of the sigmoid. 
	/*inline*/ double logit(const double x);

	//! Perform a binomial sampling of a vector Bernoulli variables.
	//! @return a vector of newly sampled binary values.
	Eigen::VectorXd binomialSampling(const Eigen::VectorXd& sample);
}
