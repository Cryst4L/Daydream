#pragma once

#include "EBM.h"

namespace Daydream
{
////////////////////////////////////////////////////////////////////////////////
//! @class RBM
//! Restricted Boltzmann Machine class.
//! The gradient of the model with respect to its parameters  
//! is computed using the Contrastive Divergence (CD-k) algorithm.
//! By default, the length of the Gibbs sampling chain is set to 1.
////////////////////////////////////////////////////////////////////////////////
class RBM : public EBM
{
  private :

	static int CD_ORDER; //TODO

	typedef struct
	{
		Eigen::MatrixXd W;	// interconnexion weights
		Eigen::VectorXd v; // visible units biases
		Eigen::VectorXd h; // hidden units biases
	} Theta;

	Theta m_theta, m_theta_grad;

  public :

	RBM(int n_visible, int n_hidden);

	//! Compute the energy associated to an explicit (v, h) configuration.
	double fullEnergy(const Eigen::VectorXd& visible, 
	                  const Eigen::VectorXd& hidden);
	//! Compute the Free Energy associated to visible unit vector.
	double freeEnergy(const Eigen::VectorXd& visible);

	//! Initialize the paramters. 
	//! The weights are uniformely sampled in [-r; +r] with:
	//! 	r = -4 * sqrt(6 / (n_visible + n_hidden))
	//! The hidden and visible biases are set to zero.
	void initParameters();

	//! Infer the hidden units probabilities given a vector of visible units.
	//! @return a vector of probabilities of activation of the visible units.
	Eigen::VectorXd inferHiddenUnits(const Eigen::VectorXd& visible);
	//! Infer the visible units probabilities given a vector of hidden units.
	//! @return a vector of probabilities of activation of the hidden units.
	Eigen::VectorXd inferVisibleUnits(const Eigen::VectorXd& hidden);

	//! Perform a step of Gibbs sampling in the context of the CD-k algorithm.
	//! The sampling is done in place (i.e. the input vector is modified). 
	void performGibbsStep(Eigen::VectorXd& visible);

	//! Compute the derivatives of the Free Energy with respect
	//! to the RBM parameters, at the given input point.
	Theta phase(const Eigen::VectorXd& sample);

	//! Compute the derivative of the negative log-likelyhood of the RBM
	//! density, with respect to it's parameters, at a given input point. 
	void evaluateGradient(const Eigen::VectorXd& sample);

	//! Flatten a structured set of RBM parameters into a single vector.
	//! @return a single vector containing all the given parameters. 
	Eigen::VectorXd unshapeTheta(Theta theta);
	// Reshape the parameters given a flat version of them.
	// @return a structured version of the given parameters.
	Theta reshapeTheta(const Eigen::VectorXd& unshaped);

	//! Get a vector containing the current RBM parameters.
	//! @return a vector containing the current RBM parameters.
	Eigen::VectorXd getUnshapedParameters();
	//! Get a vector containing the current RBM parameter gradients.
	//! @return a vector containing the current RBM parameter gradients.
	Eigen::VectorXd getUnshapedGradient();

	//! Set the current RBM parameters to the given ones.	
	void setUnshapedParameters(const Eigen::VectorXd& unshaped);
	//! Set the current RBM parameter gradients to the given ones.	
	void setUnshapedGradient(const Eigen::VectorXd& unshaped);

	//! Get the number of visible units
	//! @return the number of visible units
	int nVisible();
	//! Get the number of hidden units
	//! @return the number of visible units
	int nHidden();

	//! Get a reference to the RBM weight matrix.
	//! @return a a reference to the RBM weight matrix.
	Eigen::MatrixXd& weightMatrix();
	//! Get a reference to the visible unit biases.
	//! @return a a reference to the visible unit biases.
	Eigen::VectorXd& visibleBias();
	//! Get a reference to the hidden unit biases.
	//! @return a a reference to the hidden unit biases.
	Eigen::VectorXd& hiddenBias();

	~RBM();	
};
}
