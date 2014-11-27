#include "hiddenLayer.h"
#include "visiblelayer.h"
#include <assert.h>
template <typename R, typename T>
   class neuralNetwork{
  int numLayers;
  int * numNodesPerLayers;
  function<R, T> * transferFunction;
  hiddenLayer<R, T> ** hiddenLayers;
  visibleLayer<T> * vlayer;
 public:
 neuralNetwork(int _numLayers, int * _numNodesPerLayers, function<R, T> * _transferFunction): numLayers(_numLayers), numNodesPerLayers(_numNodesPerLayers), transferFunction(_transferFunction){
    assert(numLayers > 1);
    vlayer = new visibleLayer<T>(numNodesPerLayers[0]);
    hiddenLayers = new hiddenLayer<R,T>*[numLayers - 1];
    for(int i = 0; i < numLayers - 1; i++){
      hiddenLayers[i] = new hiddenLayer<R,T>(numNodesPerLayers[i], numNodesPerLayers[i + 1], transferFunction);
    }
  }
  virtual ~neuralNetwork(){
    delete vlayer;
    for(int i = 0; i < numLayers - 1; i++){
      delete hiddenLayers[i];
    }
    delete[] hiddenLayers;
  }
  void setWeights(const T ***weights){
    for(int i = 0; i < numLayers - 1; i++){
      hiddenLayers[i]->setWeights(weights[i]);
    }
  }
  const R * feedForward(const T * inputs){
    vlayer->setInputs(inputs);
    const R * tempInput = vlayer->computeOutputs();
    const R * tempOutput;
    for(int i = 0; i < numLayers - 1; i++){
      tempOutput = hiddenLayers[i]->computeOutputs(tempInput);
      delete[] tempInput;
      tempInput = tempOutput;
    }
    return tempInput;
  }
};
