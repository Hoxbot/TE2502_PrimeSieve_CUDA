#include "sieve_base.h"

#include <cmath>

//Private------------------------------------------------------------------------------------------
void SieveBase::flip(unsigned int in_index) {
	this->tracker_arr_[in_index] = !this->tracker_arr_[in_index];
}

void SieveBase::SimpleEratosthenes() {
	unsigned int root_of_n = std::sqrt(this->n_)+1;		//NTS: Root of n?
	for (unsigned int i = 2; i < root_of_n; i++) {
		if (this->tracker_arr_[i]) {
			for (unsigned int j = i+i; j < this->n_; j=j+i) {	//NTS: Start value i^2?
				//this->flip(j);
				this->tracker_arr_[j] = false;
			}
		}
	}
}


//Public-------------------------------------------------------------------------------------------
SieveBase::SieveBase(unsigned int in_n) {
	this->n_ = in_n;
	this->tracker_arr_ = new bool[n_]; //NTS: Includes '1' currently 
	//Start all values as true ("known as primes")
	for (unsigned int i = 0; i < this->n_; i++) {
		this->tracker_arr_[i] = true;
	}
}

SieveBase::~SieveBase() {
	delete[] this->tracker_arr_;
}

std::string SieveBase::PrimeString() {
	//Do the thing
	this->SimpleEratosthenes();

	//Fix the string
	std::string ret_str = "";
	for (unsigned int i = 0; i < this->n_; i++) {
		if (this->tracker_arr_[i]) {
			ret_str += std::to_string(i) + ", ";
		}
	}

	//Return
	return ret_str;
}
