#ifndef SIEVE_ERATOSTHENES_CPU_H
#define SIEVE_ERATOSTHENES_CPU_H

#include "sieve_base.h"

class SieveErathosthenesCPU : public SieveBase {
private:
	unsigned int start_ = 0;
	unsigned int end_ = 0;
	//unsigned int step_length_ = 1;

	void DoSieve();
	unsigned int IndexToNumber(unsigned int in_i);
public:
	SieveErathosthenesCPU(unsigned int in_n);
	~SieveErathosthenesCPU();

	bool IsPrime(unsigned int in_num);
};

#endif // !SIEVE_ERATOSTHENES_CPU_H