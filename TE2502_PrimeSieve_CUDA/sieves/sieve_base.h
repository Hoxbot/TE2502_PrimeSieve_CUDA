#include <string>

#include "stat_handler.h"

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

public:
	SieveBase(unsigned int in_n, unsigned int in_first_val);
	~SieveBase();

	std::string StringifyPrimes();
	std::string StringifyTrackerArr();
	std::string StringifyExecutionTime();
	std::string StringifyResults(std::string in_title);
};