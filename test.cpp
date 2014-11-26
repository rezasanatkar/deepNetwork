#include <iostream>
#include <cstdlib>
#include "visiblelayer.h"
int main(int argc, char ** argv){
  printf("This is a test program to verify the correctness the methods implemented in the other files\n");
  int N = 40;
  visibleLayer<int> vl(N);
  int * temp = new int[N];
  for(int i = 0; i < N; i++){
    temp[i] = i;
  }
  vl.setNodes(temp);
  const int * out = vl.getNodes();
  for(int i = 0; i < N; i++){
    printf("%d\n", out[i]);
  }
  delete[] temp;
  return EXIT_SUCCESS;
}
