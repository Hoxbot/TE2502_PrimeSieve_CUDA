#include "rabin_miller_tester.h"

#include <iostream>

//Private------------------------------------------------------------------------------------------
size_t RabinMillerTester::ModularExponentiation(size_t in_x, size_t in_y, size_t in_p) {
	size_t r = 1;
	in_x = in_x % in_p;

	while (in_y > 0) {
		if (in_y % 2 != 0/*in_y & 1*/) { r = (r*in_x) % in_p; }

		in_y /= 2;//in_y = in_y >> 1;//y/=2;
		in_x = (in_x*in_x) % in_p;
	}

	return r; //NTS:?
}

bool RabinMillerTester::RabinMillerTest(size_t in_n, size_t in_d) {

	//Generate a random number 'a' in range [2, n-2]
	std::uniform_int_distribution<int> distribution(2, (in_n - 2));
	size_t a = distribution(*(this->generator_ptr_));

	//std::cout << "\t\t\t" << a << "\n";

	//Compute a^d % n
	size_t x = ModularExponentiation(a, in_d, in_n);

	//Determine if prime or not
	//A
	if (x == 1 || x == (in_n - 1)) { return true; }
	//B
	while (in_d != (in_n - 1)) {
		x = (x*x) % in_n;
		in_d *= 2;

		if (x == 1) { return false; }
		if (x == (in_n - 1)) { return true; }
	}
	//C
	return false;
}

bool RabinMillerTester::RabinMillerLoop(size_t in_n) {

	if (in_n < 2) { return false; }
	if (in_n == 2) { return true; }
	if (in_n % 2 == 0) { return false; }

	if (in_n == 3) { return true; }	//NTS: This line is here to ensure a <= b 
									//in std::uniform_int_distribution<int>(a,b)
									//in the RabinMillerTest() function

	//Find r so that n = 2^d * r + 1 for some r >=1
	size_t d = (in_n - 1);
	while (d % 2 == 0) {
		d /= 2;
	}

	for (size_t k = 0; k < this->k_limit_; k++) {
		if (!RabinMillerTest(in_n, d)) { return false; }
	}

	return true;
}

//Public-------------------------------------------------------------------------------------------
RabinMillerTester::RabinMillerTester(size_t in_rm_precision_k) {
	this->k_limit_ = in_rm_precision_k;

	std::random_device seed_thingy;
	this->generator_ptr_ = new std::mt19937(seed_thingy());
}

RabinMillerTester::~RabinMillerTester() {
	delete this->generator_ptr_;
}

bool RabinMillerTester::DoTest(size_t in_n) {
	return RabinMillerLoop(in_n);
}
