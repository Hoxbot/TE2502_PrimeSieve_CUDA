#ifndef SIEVE_ATKIN_CPU_H
#define SIEVE_ATKIN_CPU_H

#include "sieve_base.h"

class SieveAtkinCPU : public SieveBase {
private:
	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveAtkinCPU(size_t in_n);
	~SieveAtkinCPU();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_ATKIN_CPU_H