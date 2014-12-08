#include "function.h"
template <typename R, typename T>
class node{
	int numInputs;						// number of adjacent nodes from previous layer
	T * weights;						// weights of corresponding adjacent nodes from previous layer
	T activation;						// activation of the nodes
	function<R, T> * transferFunction;	// transfer function between activation and output
public:
	// Constructor
	node(int _numInputs, function<R, T> * _transferFunction) : numInputs(_numInputs), transferFunction(_transferFunction){
		weights = new T[numInputs];
	}

	// Destructor
	virtual ~node(){
		delete[] weights;
	}

	// Assign input weights for the node
	void setWeights(T * _weights){
		for (int i = 0; i < numInputs; i++){
			weights[i] = _weights[i];
		}
	}

	// Compute the output given input
	R computeOutput(T * inputs){
		activation = computeActivation(inputs);
		return transferFunction->invoke(activation);
	}

	// Return activation value
	T getActivation(){ 
		return activation;  
	}

	// Compute the activation given input
	T computeActivation(T * inputs){
		T temp = 0;
		for (int i = 0; i < numInputs; i++){
			temp += weights[i] * inputs[i];
		}
		return temp;
	}

	T getWeight(int index)
	{
		return weights[index];
	}
};

