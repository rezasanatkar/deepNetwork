template <typename T>
class visibleLayer{
  int numNodes;
  T * nodes;
public:
  visibleLayer(int _numNodes): numNodes(_numNodes){
    nodes = new T[numNodes];
  }
  virtual void setNodes(T * input){
    for(int i = 0; i < numNodes; i++){
      nodes[i] = input[i];
    }
  }
  virtual ~visibleLayer(){
    delete[] nodes;
  }
  virtual const T * getNodes(){
    return nodes;
  }  
}; 
