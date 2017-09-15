#pragma once

#include "EBM.h"

// TODO: Put 'SGD' as an Enum argument. Add adapative techniques.

namespace Daydream
{
////////////////////////////////////////////////////////////////////////////////
//! @class Optimizer
//! A simple tool for optimizing Energy Based Models 
//! using the Stochastic Gradient Descent (SGD) method.
//! Support regularisation (L1, L2) and data shuffling.
////////////////////////////////////////////////////////////////////////////////
class Optimizer
{
  private:

	EBM* m_ebm_p;

	Eigen::MatrixXd* m_data_p;

	double m_learning_rate;
	double m_l1_penality;
	double m_l2_penality;

  public:
	
	Optimizer(EBM& ebm, Eigen::MatrixXd& data, double learning_rate=0.01);

	//! Perform a training step on at sample point
	void trainingStep(const Eigen::VectorXd& sample);
	//! Perform an epoch of training on the full dataset
	void trainingEpoch();

	//! Shuffle the dataset
	void shuffleData();

	//! Compute the average Reconstruction Error on the dataset
	double averageRE();
	//! Compute the average Pseudo Log Likelyhood on the dataset
	double averagePLL();

	//! @return a reference to the learning rate
	double& learningRate();
	//! @return a reference to the norm-1 penality
	double& l1Penality();
	//! @return a reference to the norm-2 penality
	double& l2Penality();
};
}
