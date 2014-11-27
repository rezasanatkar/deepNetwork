#if !defined _FUNCTION_H 
#define _FUNCTION_H 1
template <typename R, typename T>
class function{
 public:
  virtual R invoke(T arg) const = 0;
  virtual ~function(){}
};
#endif
