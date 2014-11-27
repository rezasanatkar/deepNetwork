template <typename R, typename T>
class function{
 public:
  virtual R invoke(T arg) const = 0;
  virtual ~function(){}
};
