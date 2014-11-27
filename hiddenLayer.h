#if !defined _HIDDENLAYER_H 
#define _HIDDENLAYER_H 1
#include "node.h"

template <typename R, typename T>
  class hiddenLayer{
  int numInputs;
  int numNodes;
  node<R, T> ** nodes;
  function<R, T> * transferFunction;
 public:
 hiddenLayer(int _numInputs, int _numNodes, function<R, T> * _transferFunction): numInputs(_numInputs), numNodes(_numNodes), transferFunction(_transferFunction){
    nodes = new node<R, T>*[numNodes];
    for(int i = 0; i < numNodes; i++){
      nodes[i] = new node<R, T>(numInputs, transferFunction);
    }
  }
  virtual ~hiddenLayer(){
    for(int i = 0; i < numNodes; i++){
      delete nodes[i];
    }
    delete[] nodes;
  }
  void setWeights(const T ** weights){
    for(int i = 0; i < numNodes; i++){
      nodes[i]->setWeights(weights[i]);
    }
  }
  R * computeOutputs(const T * inputs){
    R * ans = new R[numNodes];
    for(int i = 0; i < numNodes; i++){
      ans[i] = nodes[i]->computeOutput(inputs);
    }
    return ans;
  }
};
#endif
