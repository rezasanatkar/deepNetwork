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
	// set random number seed
	std::srand(unsigned(std::time(0)));

	// read image and labels
	vector<vector<double>> image;
	vector<int> label;
	printf("read files\n");
	read_image("train-images.idx3-ubyte", image);
	read_label("train-labels.idx1-ubyte", label);

	// set iteration number and learning rate
	int maxIter = 100;
	double step = 0.02;

	// construct the network
	printf("construct neuron network\n");
	int numInput = image[0].size();
	int numLayers = 2;
	int numNodesPerLayers[2] = { 300, 10 };
	neuralNetwork<double, double> nl(numInput, numLayers, numNodesPerLayers, new tanhFunction, new tanhFunctionD);
	
	// initialize weight with random numbers
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
		weights[i][j][k] = 0.0001 * (rand() % 201 - 100);

	nl.setWeights((double ***)weights);

	// construct permutation array
	int numImage = (int)image.size();
	vector<int> perm;
	for (int i = 0; i < numImage; i++)
		perm.push_back(i);

	// assign inputs from images
	double ** inputs = new double*[numImage];
	for (int i = 0; i < numImage; i++){
		inputs[i] = new double[image[i].size()];
		for (int j = 0; j < (int)image[i].size(); j++)
			inputs[i][j] = image[i][j] / 255;
	}

	for (int t = 1; t <= maxIter; t++){
		double err = 0;
		printf("Iteration %d : ", t);
		// random permutation of samples
		std::random_shuffle(perm.begin(), perm.end(), myrandom);
		// train the netword with all random permuted samples
		for (int n = 0; n < numImage; n++){
			if (n % (numImage / 20) == 0)
				printf(">");
			nl.backPropagation(inputs[perm[n]], label[perm[n]], step);
		}

		// count correct predictions
		int count = 0;
		for (int n = 0; n < numImage; n++){
			// do prediction with updated weights
			double * outputs = nl.feedForward(inputs[n]);
			// find most probable label
			int maxIndex = 0;
			double max = outputs[0];
			for (int i = 1; i < numNodesPerLayers[numLayers-1]; i++){
				if (outputs[i] > max)
				{
					max = outputs[i];
					maxIndex = i;
				}
			}
			// compute accumulate error
			for (int i = 1; i < numNodesPerLayers[numLayers - 1]; i++){
				double ti = 2 * (double)(label[n] == maxIndex) - 1;
				err += (outputs[i] - ti) *(outputs[i] - ti);
			}
			delete[] outputs;
			// if prediction matches true label increment count by 1
			if (label[n] == maxIndex)
				count++;
		}
		printf("\ttotal error is %.4f\n", err);
		printf("%d predictions are coorect, correct rate %.4f\n", count, (double)count / numImage);
	}

	// release memories
	for (int i = 0; i < (int)image.size(); i++)
		vector<double > ().swap(image[i]);
	vector<vector<double>>().swap(image);
	vector<int>().swap(label);

	for (int i = 0; i < numImage; i++)
		delete[] inputs[i];
	delete[] inputs;

	// read test data
	read_image("t10k-images.idx3-ubyte", image);
	read_label("t10k-labels.idx1-ubyte", label);

	// assign inputs from images
	inputs = new double*[image.size()];
	for (int i = 0; i < (int)image.size(); i++){
		inputs[i] = new double[image[i].size()];
		for (int j = 0; j < (int)image[i].size(); j++)
			inputs[i][j] = image[i][j] / 255;
	}

	int count = 0;
	for (int n = 0; n < (int)image.size(); n++){
		// do prediction
		double * outputs = nl.feedForward(inputs[n]);
		// find most probable label
		int maxIndex = 0;
		double max = outputs[0];
		for (int i = 1; i < numNodesPerLayers[numLayers - 1]; i++){
			if (outputs[i] > max)
			{
				max = outputs[i];
				maxIndex = i;
			}
		}
		delete[] outputs;
		// if prediction matches true label increment count by 1
		if (label[n] == maxIndex)
			count++;
	}
	printf("%d predictions are coorect\n", count);

	// release memories
	for (int i = 0; i < (int)image.size(); i++)
		vector<double >().swap(image[i]);
	vector<vector<double>>().swap(image);
	vector<int>().swap(label);

	for (int i = 0; i < numImage; i++)
		delete[] inputs[i];
	delete[] inputs;

	// store weights in file
	ofstream file;
	file.open("weight1.csv", ios::trunc);
	for (int i = 0; i < numNodesPerLayers[0]; i++)
	{
		for (int j = 0; j < numInput; j++)
			cout << weights[0][i][j];
		cout << endl;
	}
	file.close();
	file.open("weight2.csv");
	for (int i = 0; i < numNodesPerLayers[1]; i++)
	{
		for (int j = 0; j < numNodesPerLayers[0]; j++)
			file << weights[0][i][j] << ",";
		file << endl;
	}
	file.close();

	// releease memory for weights
	for (int i = 0; i < numLayers; i++)
	for (int j = 0; j < numNodesPerLayers[i]; j++)
		delete[] weights[i][j];
	for (int i = 0; i < numLayers; i++)
		delete[] weights[i];
	delete[] weights;

	getchar();
	return EXIT_SUCCESS;
}
