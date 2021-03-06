Daydream
========

The __Restricted Boltzmann Machine__ is one of the most fundamental and insightful Machine Learning model. 
It aims to capture the underlying distribution of the data in a factorized manner by relying on a Graph and an Energy Function which measures the likelyhood of a configuration according to Statistical Mechanics root concepts. 

__Eigen3__ is a [C++ template based library](http://eigen.tuxfamily.org/index.php?title=Main_Page) for Linear Algebra, that is widely use for its speed and memory efficiency.

__Daydream__ is an Eigen3 implementation of Restricted Boltzmann Machines. It is structured as a library, and thus could be extended to a whole Energy Based Model library. 

<p align="center">
  <img src="https://github.com/Cryst4L/Daydream/blob/master/example.png"/>
</p>

Dependencies
------------

- Eigen3 (required) a C++ based Linear Algebra library: ```[sudo apt-get install libeigen3-dev] ```
- Python (optional) for dowloading the MNIST dataset: ```[sudo apt-get install python3.5] ```
- GNUPlot (optional) for displaying the parameters while training the RBM: ```[sudo apt-get install gnuplot]```

And here's how to build the project:
```sh
mkdir build && cd build
cmake ..
make
```
Copyright
----------
This project is released under the MIT license. Enjoy :)

