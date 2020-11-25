#ifndef SIEVE_SUNDARAM_CPU_H
#define SIEVE_SUNDARAM_CPU_H

#include "sieve_base.h"

class SieveSundaramCPU : public SieveBase {
private:
	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveSundaramCPU(size_t in_n);
	~SieveSundaramCPU();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_SUNDARAM_CPU_H