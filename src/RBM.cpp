#include <Daydream/RBM.h>
#include <Daydream/Utils.h>

#include <iostream>

using namespace Eigen;

namespace Daydream
{
int RBM::CD_ORDER = 10;

RBM::RBM(int n_visible, int n_hidden)
//	  : m_visible_units(n_visible), m_hidden_units(n_hidden)
{
	m_theta.W.resize(n_hidden, n_visible);	
	m_theta.v.resize(n_visible);
	m_theta.h.resize(n_hidden);

	m_theta_grad.W.resize(n_hidden, n_visible);	
	m_theta_grad.v.resize(n_visible);
	m_theta_grad.h.resize(n_hidden);

	initParameters();
}

double RBM::fullEnergy(const VectorXd& visible, const VectorXd& hidden)
{
	double e = .0;

	e -= visible.transpose() * m_theta.W * hidden;
	e -= m_theta.v.transpose() * visible;
	e -= m_theta.h.transpose() * hidden;

	return e;
} 

double RBM::freeEnergy(const VectorXd& visible)
{
	double e = .0;

	for (int k = 0; k < m_theta.h.size(); k++)
		e -= softplus(m_theta.h(k) + m_theta.W.row(k) * visible);

	e -= m_theta.v.transpose() * visible;

	return e;
}

void RBM::initParameters()
{		  
	m_theta.W.setRandom();
	m_theta.W *= 4. * std::sqrt(6. / (m_theta.W.rows() + m_theta.W.cols())); 		

	m_theta.v.setZero();
	m_theta.h.setZero();
}

VectorXd RBM::inferHiddenUnits(const VectorXd& visible)
{
	VectorXd hidden = m_theta.h + m_theta.W * visible;
	hidden = hidden.unaryExpr(&sigmoid);
	return hidden;
}

VectorXd RBM::inferVisibleUnits(const VectorXd& hidden)
{
	VectorXd visible = m_theta.v + m_theta.W.transpose() * hidden;
	visible = visible.unaryExpr(&sigmoid);
	return visible;
} 

void RBM::performGibbsStep(VectorXd& visible)
{
	VectorXd hidden; 

//	hidden  = binomialSampling(inferHiddenUnits(visible));
//	visible = binomialSampling(inferVisibleUnits(hidden));

	hidden = inferHiddenUnits(visible);
	hidden = binomialSampling(hidden);

	visible = inferVisibleUnits(hidden);
	visible = binomialSampling(visible);
}

RBM::Theta RBM::phase(const VectorXd& sample)
{
	RBM::Theta phase;

	VectorXd hidden = 
	  (m_theta.h + m_theta.W * sample).unaryExpr(&sigmoid);

	phase.W = -hidden * sample.transpose();
	phase.v = -sample;
	phase.h = -hidden; 

	return phase;
}

/*  
// Straight but not optimal version
void RBM::evaluateGradient(const VectorXd& sample)
{
	sample = binomialSampling(sample);

	RBM::Theta pos_phase = phase(sample);

	for (int k = 0; k < CD_ORDER; k++)
		performGibbsStep(sample);

	RBM::Theta neg_phase = phase(sample);

	m_theta_grad.W = pos_phase.W - neg_phase.W;
	m_theta_grad.v = pos_phase.v - neg_phase.v;
	m_theta_grad.h = pos_phase.h - neg_phase.h; 				
}
*/

void RBM::evaluateGradient(const VectorXd& sample)
{
	VectorXd visible = binomialSampling(sample);

	VectorXd hidden = inferHiddenUnits(visible);

	RBM::Theta pos_phase = 
	  {-hidden * visible.transpose(), -visible, -hidden};

	for (int k = 0; k < CD_ORDER; k++)
	{
		hidden = binomialSampling(hidden);
		visible = inferVisibleUnits(hidden);
		visible = binomialSampling(visible);
		hidden = inferHiddenUnits(visible);
	}

	RBM::Theta neg_phase = 
	  {-hidden * visible.transpose(), -visible, -hidden};

	m_theta_grad.W = pos_phase.W - neg_phase.W;
	m_theta_grad.v = pos_phase.v - neg_phase.v;
	m_theta_grad.h = pos_phase.h - neg_phase.h;
}

VectorXd RBM::unshapeTheta(RBM::Theta theta)
{
	Map <VectorXd> weights(theta.W.data(), theta.W.size());

	VectorXd joined(weights.size() + theta.v.size() + theta.h.size());

	joined << weights, theta.v, theta.h;

	return joined;		
}

RBM::Theta RBM::reshapeTheta(const VectorXd& unshaped) 
{
	RBM::Theta reshaped;

	int n_v = m_theta.v.size();
	int n_h = m_theta.h.size();

	VectorXd weights = unshaped.segment(0, n_v * n_h);
	
	reshaped.W = Map <MatrixXd> (weights.data(), n_h, n_v);
	reshaped.v = unshaped.segment(weights.size(), n_v);
	reshaped.h = unshaped.segment(weights.size() + n_v, n_h);

	return reshaped;
}

VectorXd RBM::getUnshapedParameters()
{
	return unshapeTheta(m_theta);
}

VectorXd RBM::getUnshapedGradient()
{
	return unshapeTheta(m_theta_grad);
}

void RBM::setUnshapedParameters(const VectorXd& unshaped)
{
	m_theta = reshapeTheta(unshaped);
}

void RBM::setUnshapedGradient(const VectorXd& unshaped)
{
	m_theta_grad = reshapeTheta(unshaped);
}

int RBM::nVisible()
{
	return m_theta.W.cols();
}

int RBM::nHidden()
{
	return m_theta.W.rows();
}

MatrixXd& RBM::weightMatrix() 
{
	return m_theta.W; 
}

VectorXd& RBM::visibleBias()
{
	return m_theta.v;
}

VectorXd& RBM::hiddenBias()
{
	return m_theta.h;
}

RBM::~RBM() {}

}

