#include "node.h"

template <typename R, typename T>
class hiddenLayer{
	int numInputs;						// number of input value, i.e. number of nodes in previous layer
	int numNodes;						// number of nodes in this layer
	node<R, T> ** nodes;				// array of nodes
	function<R, T> * transferFunction;	// transfer function between activation and output of all nodes
public:
	// Constructor
	hiddenLayer(int _numInputs, int _numNodes, function<R, T> * _transferFunction) : numInputs(_numInputs), numNodes(_numNodes), transferFunction(_transferFunction){
		nodes = new node<R, T>*[numNodes];
		for (int i = 0; i < numNodes; i++){
			nodes[i] = new node<R, T>(numInputs, transferFunction);
		}
	}

	// Destructor
	virtual ~hiddenLayer(){
		for (int i = 0; i < numNodes; i++){
			delete nodes[i];
		}
		delete[] nodes;
	}

	// Assign weights for incoming edges of all nodes
	void setWeights(T ** weights){
		for (int i = 0; i < numNodes; i++){
			nodes[i]->setWeights(weights[i]);
		}
	}

	// Compute the output of all nodes
	R * computeOutputs(T * inputs){
		R * ans = new R[numNodes];
		for (int i = 0; i < numNodes; i++){
			ans[i] = nodes[i]->computeOutput(inputs);
		}
		return ans;
	}

	// Return activation value
	T * getActivations(){
		T * ans = new T[numNodes];
		for (int i = 0; i < numNodes; i++){
			ans[i] = nodes[i]->getActivation();
		}
		return ans;
	}

	T getWeight(int node, int index)
	{
		return nodes[node]->getWeight(index);
	}
};

