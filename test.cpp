#include <iostream>
#include <cstdlib>
#include "neuralnetwork.h"
#include "readfile.h"
#include <math.h>
#include <algorithm> 
#include <ctime> 

class tanhFunction : public function<double, double>{
public:
	virtual double invoke(double arg) const{
		//return tanh(arg);
		return 1.0 / (1.0 + exp(-arg));
	}
};

class tanhFunctionD : public function<double, double>{
public:
	virtual double invoke(double arg) const{
		//return 1 - tanh(arg) * tanh(arg);
		return exp(arg) / (pow(exp(arg) + 1, 2));
	}
};

int myrandom(int i) { return std::rand() % i; }

int main(int argc, char ** argv){
	//printf("This is a test program to verify the correctness the methods implemented in the other files\n");

	//int numInput = 10;
	//int numLayers = 3;
	//int numNodesPerLayers[3] = {9, 5, 7};
	//neuralNetwork<double, double> nl(numInput, numLayers, numNodesPerLayers, new tanhFunction, new tanhFunctionD);
	//double * inputs = new double[numInput];
	//for (int i = 0; i < numInput; i++)
	//	inputs[i] = 5.0 / (3 + i);

	//double *** weights;
	//weights = new double **[numLayers];
	//for (int i = 0; i < numLayers; i++)
	//	weights[i] = new double*[numNodesPerLayers[i]];

	//for (int i = 0; i < numLayers; i++)
	//for (int j = 0; j < numNodesPerLayers[i]; j++)
	//	weights[i][j] = new double[(i == 0 ? numInput : numNodesPerLayers[i - 1])];

	//for (int i = 0; i < numLayers; i++)
	//for (int j = 0; j < numNodesPerLayers[i]; j++)
	//for (int k = 0; k < (i == 0 ? numInput : numNodesPerLayers[i - 1]); k++)
	//	weights[i][j][k] = 1.0 / (i + j + k + 1);	

	//nl.setWeights((double ***)weights);
	//double * ans = nl.feedForward(inputs);
	//for (int i = 0; i < numNodesPerLayers[2]; i++)
	//	printf("%.4f ", ans[i]);
	//printf("\n");
	//delete[] ans;
	//
	//for (int t = 0; t < 100; t++){
	//	printf("\n%d-th iteration:\n", t + 1);
	//	nl.backPropagation(inputs, 3, 0.01);
	//	ans = nl.feedForward(inputs);
	//	printf("\nnew output:\n");
	//	for (int i = 0; i < numNodesPerLayers[2]; i++)
	//		printf("%.4f ", ans[i]);
	//	printf("\n");
	//	printf("J = %.4f\n", nl.computeMSE(inputs, 3));
	//	delete[] ans;
	//}

	std::srand(unsigned(std::time(0)));

	vector<vector<double>> image;
	vector<int> label;
	printf("read files\n");
	read_image("t10k-images.idx3-ubyte", image);
	read_label("t10k-labels.idx1-ubyte", label);

	int maxIter = 30000;
	double step = 0.02;

	printf("construct neuron network\n");
	int numInput = image[0].size();
	int numLayers = 2;
	int numNodesPerLayers[2] = { 300, 10 };
	neuralNetwork<double, double> nl(numInput, numLayers, numNodesPerLayers, new tanhFunction, new tanhFunctionD);
	
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
		weights[i][j][k] = 1.0 / (i == 0 ? numInput : numNodesPerLayers[i - 1]);

	nl.setWeights((double ***)weights);

	double * inputs = new double[numInput];
	double * outputs;
	vector<int> perm;
	for (int i = 0; i < (int)image.size(); i++)
		perm.push_back(i);

	//for (int i = 0; i < numInput; i++)
	//	inputs[i] = image[0][i] / 255;
	//for (int t = 0; t < 20; t++){
	//	printf("\n%d-th iteration:\n", t + 1);
	//	nl.backPropagation(inputs, 5, 0.01);
	//	outputs = nl.feedForward(inputs);
	//	printf("\nnew output:\n");
	//	for (int i = 0; i < numNodesPerLayers[1]; i++)
	//		printf("%.4f ", outputs[i]);
	//	printf("\n");
	//	printf("J = %.4f\n", nl.computeMSE(inputs, 5));
	//	delete[] outputs;
	//}

	for (int t = 1; t <= maxIter; t++){
		double err = 0;
		printf("Iteration %d : ", t);
		// random permutation of samples
		//std::random_shuffle(perm.begin(), perm.end(), myrandom);
		// train the netword with all random permuted samples
		for (int n = 0; n < (int)image.size() / 5000; n++){
			if (n % 500 == 0)
				printf(">");
			for (int i = 0; i < numInput; i++)
				inputs[i] = image[perm[n]][i] / 255;
			nl.backPropagation(inputs, label[perm[n]], step);
		}

		for (int n = 0; n < (int)image.size() / 5000; n++)
			err += nl.computeMSE(inputs, label[n]);
		printf("total error is %.4f\n", err);

		// count correct predictions
		int count = 0;
		for (int n = 0; n < (int)image.size() / 5000; n++){
			for (int i = 0; i < numInput; i++)
				inputs[i] = image[n][i] / 255;
			// do prediction with updated weights
			outputs = nl.feedForward(inputs);
			
			// find most probable label
			int maxIndex = 0;
			double max = outputs[0];
			for (int i = 1; i < numNodesPerLayers[i]; i++){
				if (outputs[i] > max)
				{
					max = outputs[i];
					maxIndex = i;
				}
			}
			// if prediction matches true label increment count by 1
			if (label[n] == maxIndex + 1)
				count++;
		}
		printf("%d predictions are coorect\n", count);
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
