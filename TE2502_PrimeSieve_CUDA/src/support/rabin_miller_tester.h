#ifndef RABIN_MILLER_TESTER_H
#define RABIN_MILLER_TESTER_H

#include <random>

class RabinMillerTester {
private:
	size_t k_limit_;
	std::default_random_engine* generator_ptr_ = nullptr;

	size_t ModularExponentiation(size_t in_x, size_t in_y, size_t in_p);
	bool RabinMillerTest(size_t in_n, size_t in_d);
	bool RabinMillerLoop(size_t in_n);
	
public:
	RabinMillerTester(size_t in_rm_precision_k);
	~RabinMillerTester();

	bool DoTest(size_t in_n);
};



#endif // !RABIN_MILLER_TESTER_H
