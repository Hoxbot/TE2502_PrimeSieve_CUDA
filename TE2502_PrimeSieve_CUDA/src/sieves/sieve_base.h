#ifndef SIEVE_BASE_H
#define SIEVE_BASE_H

#include <string>
#include <vector>

#include "prime_memory/prime_memory_bool.h"
#include "../support/stat_handler.h"

class SieveBase {
private:
	float VerifyByFile();

protected:
	unsigned int n_ = 0;

	PrimeMemory* mem_class_ptr_ = nullptr;

	StatHandler private_timer_;

	virtual void DoSieve() = 0;
	virtual unsigned int IndexToNumber(unsigned int in_i) = 0;

public:
	SieveBase(unsigned int in_n);
	~SieveBase();

	virtual bool IsPrime(unsigned int in_num) = 0;

	std::string StringifyPrimes();
	std::string StringifyTrackerArr();
	std::string StringifyExecutionTime();
	std::string StringifyResults(std::string in_title);

	std::vector<int> PrimeVector();

	
};

#endif // !SIEVE_BASE_H