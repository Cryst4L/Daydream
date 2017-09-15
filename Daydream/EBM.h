#pragma once

#include <Eigen/Core>

namespace Daydream
{
////////////////////////////////////////////////////////////////////////////////
//! @class EBM
//! Abstract class used as a base for all Energy Based Models
////////////////////////////////////////////////////////////////////////////////
class EBM
{
  public :

	//! Parameters initialization routine
	virtual void initParameters() = 0;

	//! Visible units probability estimation
	virtual Eigen::VectorXd inferHiddenUnits(const Eigen::VectorXd& visible) = 0;
	//! Hidden units probability estimation
	virtual Eigen::VectorXd inferVisibleUnits(const Eigen::VectorXd& hidden) = 0;

	//! Gradient computing routine
	virtual void evaluateGradient(const Eigen::VectorXd& sample) = 0;

	//! Parameters accessor
	virtual Eigen::VectorXd getUnshapedParameters() = 0;
	//! Parameters mutator
	virtual void setUnshapedParameters(const Eigen::VectorXd& unshaped) = 0;

	//! Gradient accessor
	virtual Eigen::VectorXd getUnshapedGradient() = 0;
	//! Gradient mutator
	virtual void setUnshapedGradient(const Eigen::VectorXd& unshaped) = 0;

	//! Compute the energy associated to an explicit (v, h) configuration.
	virtual double fullEnergy(const Eigen::VectorXd& visible, 
	                          const Eigen::VectorXd& hidden) = 0;
	//! Compute the Free Energy associated to visible unit vector.
	virtual double freeEnergy(const Eigen::VectorXd& visible) = 0;

	//! Number of Visible Units
	virtual int nVisible() = 0;
	//! Number of Hidden Units
	virtual int nHidden() = 0;

	//! Dtor
	virtual ~EBM() {};
};
}
