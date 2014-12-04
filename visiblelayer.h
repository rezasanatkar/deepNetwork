#if !defined _VISIBLELAYER_H
#define _VISIBLELAYER 1
template <typename T>
class visibleLayer{
	int numNodes;	// number of nodes in visible layer
	T * nodes;		// array of nodes
public:
	// Constructor
	visibleLayer(int _numNodes) : numNodes(_numNodes){
		nodes = new T[numNodes];
	}

	// Destructor
	virtual ~visibleLayer(){
		delete[] nodes;
	}

	// Assign input of visible nodes
	void setInputs(T * input){
		for (int i = 0; i < numNodes; i++){
			nodes[i] = input[i];
		}
	}

	// Compute outputs of visible layer, i.e. the observations
	T * computeOutputs(){
		T * ans = new T[numNodes];
		for (int i = 0; i < numNodes; i++){
			ans[i] = nodes[i];
		}
		return ans;
	}
};
#endif
