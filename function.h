#include <math.h>
template <typename R, typename T>
class function{
public:
	virtual R invoke(T arg) const = 0;
	virtual ~function(){}
};

class tanhFunction : public function<double, double>{
public:
	virtual double invoke(double arg) const{
		//return tanh(arg);
		return 1.0 / (1.0 + exp(-arg));
	}
};

class tanhFunctionD : public function<double, double>{
public:
	virtual double invoke(double arg) const{
		//return 1 - tanh(arg) * tanh(arg);
		return exp(arg) / (pow(exp(arg) + 1, 2));
	}
};

