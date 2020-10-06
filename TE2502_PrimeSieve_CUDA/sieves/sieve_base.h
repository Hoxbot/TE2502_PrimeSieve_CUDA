#include <string>

class SieveBase {
private:
	int n_ = 0;
	bool* tracker_arr_;

	void flip(unsigned int in_index);

	void SimpleEratosthenes();

public:
	SieveBase(unsigned int in_n);
	~SieveBase();

	std::string PrimeString();
};