#if !defined _VISIBLELAYER_H
#define _VISIBLELAYER 1
template <typename T>
class visibleLayer{
  int numNodes;
  T * nodes;
public:
  visibleLayer(int _numNodes): numNodes(_numNodes){
    nodes = new T[numNodes];
  }
  void setInputs(const T * input){
    for(int i = 0; i < numNodes; i++){
      nodes[i] = input[i];
    }
  }
  virtual ~visibleLayer(){
    delete[] nodes;
  }
  const T * computeOutputs(){
    T * ans = new T[numNodes];
    for(int i = 0; i < numNodes; i++){
      ans[i] = nodes[i];
    }
    return ans;
  }  
}; 
#endif
