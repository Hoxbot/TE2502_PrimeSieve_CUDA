#include <string>

#include "stat_handler.h"

class SieveBase {
protected:
	unsigned int n_ = 0;
	bool* tracker_arr_;

	StatHandler private_timer_;

	void SetNonPrime(unsigned int in_i);
	void SetPrime(unsigned int in_i);

	virtual void DoSieve() = 0;

public:
	SieveBase(unsigned int in_n);
	~SieveBase();

	std::string StringifyPrimes();
	std::string StringifyTrackerArr();
	std::string StringifyExecutionTime();
	std::string StringifyResults(std::string in_title);
};