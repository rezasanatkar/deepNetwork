#include "function.h"
template <typename R, typename T>
class node{
  int numInputs;
  T * weights;
  function<T, T> * transferFunction;
 public:
 node(int _numInputs, function<R, T> * _transferFunction): numInputs(_numInputs), transferFunction(_transferFunction){
    weights = new T[numInputs];
  }
  virtual ~ node(){
    delete[] weights;
  }
  void setWeights(const T * _weights){
    for(int i = 0; i < numInputs; i++){
      weights[i] = _weights[i];
    }
  }
  R computeOutput(const T * inputs){
    T temp = 0;
    for(int i = 0; i < numInputs; i++){
      temp += weights[i] * inputs[i];
    }
    return transferFunction->invoke(temp);
  }
};
