#include "sieve_base.h"

class SieveErathosthenesCPU : public SieveBase {
private:

	void DoSieve();
public:
	SieveErathosthenesCPU(unsigned int in_n);
	~SieveErathosthenesCPU();
};