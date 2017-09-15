#include <Daydream/Core>
#include "Display.h"

using namespace Eigen;
using namespace Daydream;

// Macro: build a grid of 2D patches from a set of row vectors.
MatrixXd buildGridfromData(MatrixXd& data)
{
	int grid_rows = std::sqrt(data.rows());
	int grid_cols = (data.rows() + grid_rows - 1) / grid_rows;

	int wh = std::sqrt(data.cols());
	MatrixXd grid(wh * grid_rows, wh * grid_cols);

	for (int n = 0; n < data.rows(); n++)
	{
		int row = n / grid_cols;
		int col = n % grid_cols;

		VectorXd sample = data.row(n);
		MatrixXd patch = Map <MatrixXd> (sample.data(), wh, wh);

		grid.block(row * wh, col * wh, wh, wh) = patch.transpose();
	}

	return grid;
}

// Main: train an RBM on a subset of the MNIST.
int main(void)
{
	RBM my_rbm(784, 25);

	MNIST dataset("../data", MNIST::TRAIN, 10000);

	Optimizer optimizer(my_rbm, dataset.samples());	
	optimizer.learningRate() = 1e-2;
	optimizer.l2Penality() = 1e-2;

	MatrixXd feature_map = buildGridfromData(my_rbm.weightMatrix());
	Display display(&feature_map, "Feature Vectors of the RBM");

	for (int i = 0; i < 10; i++) 
	{ 
		optimizer.shuffleData();
		if (i != 0) optimizer.trainingEpoch();

		feature_map = buildGridfromData(my_rbm.weightMatrix());
		display.render();

		std::cout << "Epoch : " << i << " | "
		          << "RE : " << optimizer.averageRE() << '\n';
	}

	return 0;
}
