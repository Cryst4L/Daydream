#pragma once

#undef Success //TODO: this !!!!
#include <Eigen/Core>

namespace Daydream
{
////////////////////////////////////////////////////////////////////////////////
//! @class MNIST
//! MNIST dataset loader
////////////////////////////////////////////////////////////////////////////////
class MNIST
{
  public:

	enum DatasetType {TRAIN, TEST, VALIDATION};

	MNIST(std::string folder_path, DatasetType ds_type=TRAIN, int size=-1);

	//! Get a reference to the loaded samples. 
	//! Each row store a sample data using a row-major indexing. 
	Eigen::MatrixXd& samples();
	//! Get a reference to the loaded labels. 
	//! Each row represent a sample class in a one-hot manner. 
	Eigen::MatrixXd& labels();

	//! Get the number of samples loaded
	int NbSamples();

  private:

	int m_size;

	Eigen::MatrixXd m_samples;
	Eigen::MatrixXd m_labels;

	//! Rearrange and IDX 32-bit integer
	inline int reverseInt(int i);

	//! Load the MNIST samples
	void loadSamples(std::string folder_path, DatasetType ds_type);
	//! Load the MNIST labels
	void loadLabels(std::string folder_path, DatasetType ds_type);
};
}
