#include <Daydream/Optimizer.h>
#include <Daydream/Utils.h>

#include <iostream>

using namespace Eigen;

namespace Daydream
{
Optimizer::Optimizer(EBM& ebm, MatrixXd& data, double learning_rate)
  :  m_ebm_p(&ebm), m_data_p(&data), 
     m_learning_rate(learning_rate),
     m_l1_penality(0), m_l2_penality(0) 
{}

void Optimizer::trainingStep(const VectorXd& sample)
{
	m_ebm_p->evaluateGradient(sample);

	// Retreive parameters and gradients		
	VectorXd theta = m_ebm_p->getUnshapedParameters();
	VectorXd theta_grad = m_ebm_p->getUnshapedGradient();

	// Apply regularisations
	if (m_l1_penality != 0)
		theta_grad.array() += m_l1_penality * theta.array().sign(); 

	if (m_l2_penality != 0)
		theta_grad += m_l2_penality * theta / theta.norm();

	// Perform step update
	theta -= m_learning_rate * theta_grad;
	m_ebm_p->setUnshapedParameters(theta);
}

void Optimizer::trainingEpoch()
{
	for (int n = 0; n < m_data_p->rows(); n++)
		trainingStep(m_data_p->row(n));
}

void Optimizer::shuffleData()
{
	int rows = m_data_p->rows(); 

	VectorXi index = VectorXi::LinSpaced(rows, 0, rows);
	std::random_shuffle(index.data(), index.data() + rows);

	(*m_data_p) = index.asPermutation() * (*m_data_p);
}

double Optimizer::averageRE()
{
	double acc = 0.0;

	VectorXd sample, hidden, estimate;

	for (int n = 0; n < m_data_p->rows(); n++) 
	{
		sample = m_data_p->row(n);
		hidden = m_ebm_p->inferHiddenUnits(sample);
		estimate = m_ebm_p->inferVisibleUnits(hidden);

		acc += (sample - estimate).cwiseAbs().mean();
	}

	return (acc / m_data_p->rows());
}

double Optimizer::averagePLL()
{
	double acc = 0.0;

	VectorXd sample, turned;

	for (int n = 0; n < m_data_p->rows(); n++) 
	{
		sample = m_data_p->row(n);

		int i = std::rand() % sample.size();
		
		turned = sample;
		turned(i) = -turned(i);

		double sample_energy = m_ebm_p->freeEnergy(sample);
		double turned_energy = m_ebm_p->freeEnergy(turned);

		double p_i = /*EBM::*/sigmoid(turned_energy - sample_energy);

		acc += sample.size() * std::log(p_i);
	}

	return (acc / m_data_p->rows());
}

double& Optimizer::learningRate()
{
	return m_learning_rate;
}

double& Optimizer::l1Penality()
{
	return m_l1_penality;
}

double& Optimizer::l2Penality()
{
	return m_l2_penality;
}
}
