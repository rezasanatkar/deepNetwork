#include <iostream>
#include <cstdlib>
#include "neuralnetwork.h"
#include <math.h>

class tanhFunction : public function<double, double>{
public:
	virtual double invoke(double arg) const{
		return tanh(arg);
	}
};

class tanhFunctionD : public function<double, double>{
public:
	virtual double invoke(double arg) const{
		return 1 - tanh(arg) * tanh(arg);
	}
};

int main(int argc, char ** argv){
	printf("This is a test program to verify the correctness the methods implemented in the other files\n");
	int numInput = 10;
	int numLayers = 3;
	int numNodesPerLayers[3] = {9, 5, 7};
	neuralNetwork<double, double> nl(numInput, numLayers, numNodesPerLayers, new tanhFunction, new tanhFunctionD);
	double * inputs = new double[numInput];
	for (int i = 0; i < numInput; i++)
		inputs[i] = 5.0 / (3 + i);

	double *** weights;
	weights = new double **[numLayers];
	for (int i = 0; i < numLayers; i++)
		weights[i] = new double*[numNodesPerLayers[i]];

	for (int i = 0; i < numLayers; i++)
	for (int j = 0; j < numNodesPerLayers[i]; j++)
		weights[i][j] = new double[(i == 0 ? numInput : numNodesPerLayers[i - 1])];

	for (int i = 0; i < numLayers; i++)
	for (int j = 0; j < numNodesPerLayers[i]; j++)
	for (int k = 0; k < (i == 0 ? numInput : numNodesPerLayers[i - 1]); k++)
		weights[i][j][k] = 1.0 / (i + j + k + 1);	

	nl.setWeights((double ***)weights);
	double * ans = nl.feedForward(inputs);
	for (int i = 0; i < numNodesPerLayers[2]; i++)
		printf("%.4f ", ans[i]);
	printf("\n");
	delete[] ans;
	
	for (int t = 0; t < 100; t++){
		printf("\n%d-th iteration:\n", t + 1);
		nl.backPropagation(inputs, 3, 0.01);
		ans = nl.feedForward(inputs);
		printf("\nnew output:\n");
		for (int i = 0; i < numNodesPerLayers[2]; i++)
			printf("%.4f ", ans[i]);
		printf("\n");
		printf("J = %.4f\n", nl.computeMSE(inputs, 3));
		delete[] ans;
	}

	delete[] inputs;
	for (int i = 0; i < numLayers; i++)
	for (int j = 0; j < numNodesPerLayers[i]; j++)
		delete[] weights[i][j];
	for (int i = 0; i < numLayers; i++)
		delete[] weights[i];
	delete[] weights;

	getchar();
	return EXIT_SUCCESS;
}
