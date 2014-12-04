#include "hiddenLayer.h"
#include "visiblelayer.h"
#include <assert.h>
template <typename R, typename T>
class neuralNetwork{
	int numLayers;						// number of all layers in the network
	int * numNodesPerLayers;			// list of number of nodes for each layer
	function<R, T> * transferFunction;	// transfer function between activation and output of all nodes
	function<R, T> * derivative;		// derivative of transfer function
	hiddenLayer<R, T> ** hiddenLayers;	// array of hidden layers
	visibleLayer<T> * vlayer;			// visible layer of the network
	T ** activations;					// activations of all nodes
	R ** outputs;						// outputs of all nodes
	R ** delta;							// sensitivity components of back-propagation
	T *** tempWeights;					// temporary weights used in back propagation
public:
	// Constructor
	neuralNetwork(int _numLayers, int * _numNodesPerLayers, function<R, T> * _transferFunction, function<R, T> * _derivative) : numLayers(_numLayers), numNodesPerLayers(_numNodesPerLayers), transferFunction(_transferFunction), derivative(_derivative){
		assert(numLayers > 1);
		vlayer = new visibleLayer<T>(numNodesPerLayers[0]);
		hiddenLayers = new hiddenLayer<R, T>*[numLayers - 1];
		delta = new R*[numLayers - 1];
		for (int i = 0; i < numLayers - 1; i++){
			hiddenLayers[i] = new hiddenLayer<R, T>(numNodesPerLayers[i], numNodesPerLayers[i + 1], transferFunction);
			delta[i] = new R[numNodesPerLayers[i + 1]];
		}

		activations = new T*[numLayers];
		outputs = new R*[numLayers];
		for (int i = 0; i < numLayers; i++){
			activations[i] = new T[numNodesPerLayers[i]];
			outputs[i] = new R[numNodesPerLayers[i]];
		}

		tempWeights = new T **[numLayers - 1];
		for (int i = 0; i < numLayers - 1; i++){
			tempWeights[i] = new T *[numNodesPerLayers[i + 1]];
			for (int j = 0; j < numNodesPerLayers[i + 1]; j++)
				tempWeights[i][j] = new T[numNodesPerLayers[i]];
		}
	}

	// Destructor
	virtual ~neuralNetwork(){
		delete vlayer;
		for (int i = 0; i < numLayers - 1; i++){
			delete hiddenLayers[i];
			delete[] delta[i];
		}
		delete[] hiddenLayers;
		delete[] delta;

		for (int i = 0; i < numLayers; i++){
			delete[] activations[i];
			delete[] outputs[i];
		}
		delete[] activations;
		delete[] outputs;


		for (int i = 0; i < numLayers - 1; i++){
			for (int j = 0; j < numNodesPerLayers[i + 1]; j++)
				delete[] tempWeights[i][j];
			delete[] tempWeights[i];
		}
		delete[] tempWeights;
	}

	// Assign weights for all edges in the network
	void setWeights(T ***weights){
		for (int i = 0; i < numLayers - 1; i++){
			hiddenLayers[i]->setWeights(weights[i]);
		}
		for (int i = 0; i < numLayers - 1; i++){
			for (int j = 0; j < numNodesPerLayers[i + 1]; j++){
				for (int k = 0; k < numNodesPerLayers[i]; k++){
					tempWeights[i][j][k] = weights[i][j][k];
				}
			}
		}
	}

	// Feed-forward algorithm to compute the output of network
	R * feedForward(T * inputs){
		vlayer->setInputs(inputs);
		R * tempInput = vlayer->computeOutputs();
		R * tempOutput;
		for (int i = 0; i < numLayers - 1; i++){
			tempOutput = hiddenLayers[i]->computeOutputs(tempInput);
			delete[] tempInput;
			tempInput = tempOutput;
		}
		return tempInput;
	}

	// Back-propagation algorithm to update weights
	void backPropagation(T * inputs, const int label, const R epsilon){
		// compute all activations and outputs
		computeAlphaBeta(inputs);
		// compute all delta's
		computeSensitivity(tempWeights, label);
		// update all weights
		updateWeights(epsilon);
	}
private:
	// Compute activation (alpha) and output (beta) for all nodes
	void computeAlphaBeta(T * inputs){
		vlayer->setInputs(inputs);
		T * temp = vlayer->computeOutputs();
		for (int j = 0; j < numNodesPerLayers[0]; j++){
			outputs[0][j] = temp[j];
			activations[0][j] = inputs[j];
		}
		delete[] temp;
		
		for (int i = 1; i < numLayers - 1; i++){
			temp = hiddenLayers[i]->computeOutputs(outputs[i - 1]);
			for (int j = 0; j < numNodesPerLayers[0]; j++){
				outputs[i][j] = temp[j];
			}
			delete[] temp;

			temp = hiddenLayers[i]->getActivations();
			for (int j = 0; j < numNodesPerLayers[0]; j++){
				activations[i][j] = temp[j];
			}
			delete[] temp;
		}
	}

	// Compute delta's
	void computeSensitivity(T ***weights, const int label){
		for (int i = numLayers - 2; i >= 0; i--){
			for (int k = 0; k < numNodesPerLayers[i + 1]; k++){
				if (i == numLayers - 2)
					delta[i][k] = 2 * ((int)(label == k) - outputs[i][k]);
				else{
					delta[i][k] = 0;
					for (int l = 0; l < numNodesPerLayers[i + 1]; l++)
						delta[i][k] += weights[i + 1][k][l] * delta[i + 1][l];
					delta[i][k] *= derivative->invoke(activations[i][k]);
				}
			}
		}
	}

	// Update weights
	void updateWeights(T epsilon){
		for (int i = 0; i < numLayers - 1; i++){
			for (int j = 0; j < numNodesPerLayers[i+1]; j++){
				for (int k = 0; k < numNodesPerLayers[i]; k++){
					tempWeights[i][j][k] = tempWeights[i][j][k] + epsilon * delta[i][k] * outputs[i][j];
					T change = epsilon * delta[i][k] * outputs[i][j];
					change = change;
				}
			}
		}
		for (int i = 0; i < numLayers - 1; i++){
			hiddenLayers[i]->setWeights(tempWeights[i]);
		}
	}
};