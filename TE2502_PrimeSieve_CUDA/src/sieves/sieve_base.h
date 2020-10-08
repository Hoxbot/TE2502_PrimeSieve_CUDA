#ifndef SIEVE_BASE_H
#define SIEVE_BASE_H

#include <string>
#include <vector>

#include "../support/stat_handler.h"

class SieveBase {
private:
	unsigned int index_offset_ = 0;
	unsigned int n_alt_ = 0;

protected:
	unsigned int n_ = 0;
	bool* tracker_arr_;

	StatHandler private_timer_;

	bool CheckIndex(unsigned int in_i);

	void SetNonPrime(unsigned int in_i);
	void SetPrime(unsigned int in_i);

	virtual void DoSieve() = 0;

	float VerifyByFile();

public:
	SieveBase(unsigned int in_n, unsigned int in_first_val);
	~SieveBase();

	bool IsPrime(unsigned int in_num);

	std::string StringifyPrimes();
	std::string StringifyTrackerArr();
	std::string StringifyExecutionTime();
	std::string StringifyResults(std::string in_title);

	std::vector<int> PrimeVector();
};

#endif // !SIEVE_BASE_H