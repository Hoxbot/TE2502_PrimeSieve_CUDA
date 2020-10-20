#ifndef SIEVE_SUNDARAM_CUDA_H
#define SIEVE_SUNDARAM_CUDA_H

#include "sieve_base.h"

//CUDA Functions here

//Class
class SieveSundaramCUDA : public SieveBase {
private:
	size_t start_ = 0;
	size_t end_ = 0;
	//unsigned int step_length_ = 1;

	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveSundaramCUDA(size_t in_n);
	~SieveSundaramCUDA();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_SUNDARAM_CUDA_H