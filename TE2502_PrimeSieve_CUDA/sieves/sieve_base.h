#include <vector>

class SieveBase {
private:
	int n_ = 0;
	bool* tracker_arr_;

	void flip(in_index);

	void SimpleEratosthenes();

public:
	SieveBase(unsigned int in_n);
	~SieveBase();
};