#include <iostream>
#include <cstdlib>
#include "neuralnetwork.h"
#include <math.h>

class tanhFunction: public function<double, double>{
public:
  virtual double invoke(double arg) const{
    return arg;
  }
};
int main(int argc, char ** argv){
  printf("This is a test program to verify the correctness the methods implemented in the other files\n");
  int numLayers = 3;
  int numNodesPerLayers[3] = {10, 5, 3};
  neuralNetwork<double, double> nl(numLayers, numNodesPerLayers, new tanhFunction);
  double * inputs = new double[numNodesPerLayers[0]];
  for(int i = 0; i < numNodesPerLayers[0]; i++){
    inputs[i] = 5 / (1 + i) ;
  }
  double *** weights;
  weights = new double **[numLayers - 1];
  for(int i = 0; i < numLayers - 1; i++){
    weights[i] = new double*[numNodesPerLayers[i + 1]];
  }
  for(int i = 0; i < numLayers - 1; i++){
    for(int j = 0; j < numNodesPerLayers[i + 1]; j++){
      weights[i][j] = new double[numNodesPerLayers[i]];
    }
  }
  for(int i = 0; i < numLayers - 1; i++){
    for(int j = 0; j < numNodesPerLayers[i + 1]; j++){
      for(int k = 0; k < numNodesPerLayers[i]; k++){
	weights[i][j][k] = i + j + k;
      }
    }
  }

  nl.setWeights((const double ***)weights);
  const double * ans = nl.feedForward(inputs);
  for(int i = 0; i < numNodesPerLayers[0]; i++){
    printf("%f\n", ans[i]);
  }
  delete[] ans;
  delete[] inputs;
  for(int i = 0; i < numLayers - 1; i++){
    for(int j = 0; j < numNodesPerLayers[i + 1]; j++){
      delete[] weights[i][j];
    }
  }
  for(int i = 0; i < numLayers - 1; i++){
    delete[] weights[i];
  }
  delete[] weights;
  return EXIT_SUCCESS;
}
