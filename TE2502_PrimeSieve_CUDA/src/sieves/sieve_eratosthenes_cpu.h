#ifndef SIEVE_ERATOSTHENES_CPU_H
#define SIEVE_ERATOSTHENES_CPU_H

#include "sieve_base.h"

class SieveErathosthenesCPU : public SieveBase {
private:
	//size_t start_ = 0;
	//size_t end_ = 0;
	//unsigned int step_length_ = 1;

	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveErathosthenesCPU(size_t in_n);
	~SieveErathosthenesCPU();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_ERATOSTHENES_CPU_H