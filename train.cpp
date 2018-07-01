#include <iostream>
#include <cstdlib>
#include "neuralnetwork.h"
#include "readfile.h"
#include <math.h>
#include <algorithm> 
#include <ctime> 


int myrandom(int i) { return std::rand() % i; }

int main(int argc, char ** argv){
	// set random number seed
	std::srand(unsigned(std::time(0)));

	// read training image and labels
	vector<vector<double>> train_image;
	vector<int> train_label;
	printf("read training files\n");
	read_image("train-images.idx3-ubyte", train_image);
	read_label("train-labels.idx1-ubyte", train_label);

	// read testing image and labels
	vector<vector<double>> test_image;
	vector<int> test_label;
	printf("read testing files\n");
	read_image("t10k-images.idx3-ubyte", test_image);
	read_label("t10k-labels.idx1-ubyte", test_label);

	// set iteration number and learning rate
	int maxIter = 100;
	double step = 0.02;

	// construct the network
	printf("construct neural network\n");
	int numInput = train_image[0].size();
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
	int numImage = (int)train_image.size();// / 100;
	vector<int> perm;
	for (int i = 0; i < numImage; i++)
		perm.push_back(i);

	// assign inputs from training images
	double ** train_input = new double*[numImage];
	for (int i = 0; i < numImage; i++){
		train_input[i] = new double[train_image[i].size()];
		for (int j = 0; j < (int)train_image[i].size(); j++)
			train_input[i][j] = train_image[i][j] / 255;
	}


	// assign inputs from testing images
	double ** test_input = new double*[test_image.size()];
	for (int i = 0; i < (int)test_image.size(); i++){
		test_input[i] = new double[test_image[i].size()];
		for (int j = 0; j < (int)test_image[i].size(); j++)
			test_input[i][j] = test_image[i][j] / 255;
	}
	double ** weightsRBM = new double*[numInput];
	for(int i = 0; i < numInput; i++){
	  weightsRBM[i] = new double[numNodesPerLayers[0]];
	}
	for(int i = 0; i < numInput; i++){
	  for(int j = 0; j < numNodesPerLayers[0]; j++){
	    weightsRBM[i][j] = weights[0][j][i];
	  } 
	}
	int RBMapply = 1;
	if(RBMapply == 1){
	  hiddenLayer<double, double> * tempHiddenLayer = new hiddenLayer<double, double>(numNodesPerLayers[0], numInput, new tanhFunction); 
	  double epsilon = 0.02;
	  int t1 = 0;
	  for (int n = 0; n < numImage; n++){
	    if((n * 100) / numImage >= t1 + 1){
	    printf("%d%%\n", (n * 100) / numImage);
	    t1 = (n * 100) / numImage;
	    }
	    nl.RBM(train_input[n], 0, tempHiddenLayer, weightsRBM, epsilon);
	  }
	  nl.setWeights();
	}




	// releease memory for weights
	for (int i = 0; i < numLayers; i++)
	for (int j = 0; j < numNodesPerLayers[i]; j++)
		delete[] weights[i][j];
	for (int i = 0; i < numLayers; i++)
		delete[] weights[i];
	delete[] weights;
	
	double * train_err = new double[maxIter];
	double * test_err = new double[maxIter];

	for (int t = 1; t <= maxIter; t++){
		double err = 0;

		printf("Iteration %d : \n", t);

		// random permutation of samples
		std::random_shuffle(perm.begin(), perm.end(), myrandom);
		// train the netword with all random permuted samples
		for (int n = 0; n < numImage; n++){
			if (n % (numImage / 20) == 0)
				printf(">\n");
			nl.backPropagation(train_input[perm[n]], train_label[perm[n]], step);
		}

		// count correct predictions
		int count = 0;
		for (int n = 0; n < numImage; n++){
			// do prediction with updated weights
			double * outputs = nl.feedForward(train_input[n]);
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
				double ti = 2 * (double)(train_label[n] == maxIndex) - 1;
				err += (outputs[i] - ti) *(outputs[i] - ti);
			}
			delete[] outputs;
			// if prediction matches true label increment count by 1
			if (train_label[n] == maxIndex)
				count++;
		}
		printf("\ttotal error is %.4f\n", err);
		printf("error rate on training set %.4f, ", 1 - (double)count / numImage);
		train_err[t] = 1 - (double)count / numImage;

		count = 0;
		for (int n = 0; n < (int)test_image.size(); n++){
			// do prediction
			double * outputs = nl.feedForward(test_input[n]);
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
			if (test_label[n] == maxIndex)
				count++;
		}
		printf("error rate on testing set %.4f\n", 1 - (double)count / test_image.size());
		test_err[t] = 1 - (double)count / test_image.size();
	}

	// release memories
	for (int i = 0; i < (int)train_image.size(); i++)
		vector<double > ().swap(train_image[i]);
	vector<vector<double>>().swap(train_image);
	vector<int>().swap(train_label);

	for (int i = 0; i < numImage; i++)
		delete[] train_input[i];
	delete[] train_input;

	for (int i = 0; i < (int)test_image.size(); i++)
		vector<double >().swap(test_image[i]);
	vector<vector<double>>().swap(test_image);
	vector<int>().swap(test_label);

	for (int i = 0; i < (int)test_image.size(); i++)
		delete[] test_input[i];
	delete[] test_input;

	// store weights in file
	ofstream file;
	file.open("weight1.csv",ios::trunc);
	for (int i = 0; i < numNodesPerLayers[0]; i++)
	{
		for (int j = 0; j < numInput; j++)
			file << nl.getWeight(0,i,j) << ",";
		file << endl;
	}
	file.close();
	file.open("weight2.csv", ios::trunc);
	for (int i = 0; i < numNodesPerLayers[1]; i++)
	{
		for (int j = 0; j < numNodesPerLayers[0]; j++)
			file << nl.getWeight(1, i, j) << ",";
		file << endl;
	}
	file.close();
	file.open("error.csv", ios::trunc);
	for (int t = 1; t <= maxIter; t++)
		file << train_err[t] << ",";
	file << endl;
	for (int t = 1; t <= maxIter; t++)
		file << test_err[t] << ",";
	file << endl;
	file.close();

	getchar();
	return EXIT_SUCCESS;
}
