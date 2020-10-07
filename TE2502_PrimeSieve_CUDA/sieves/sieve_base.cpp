#include "sieve_base.h"

#include <cmath>

//Private------------------------------------------------------------------------------------------
void SieveBase::SetNonPrime(unsigned int in_i) {
	this->tracker_arr_[in_i] = false;
}

void SieveBase::SetPrime(unsigned int in_i) {
	this->tracker_arr_[in_i] = true;
}

//Public-------------------------------------------------------------------------------------------
SieveBase::SieveBase(unsigned int in_n) {
	this->n_ = in_n;
	this->tracker_arr_ = new bool[n_]; //NTS: indexes [0, n-1]
	//Start all values as true ("known as primes")
	for (unsigned int i = 0; i < this->n_; i++) {
		this->tracker_arr_[i] = true;
	}
}

SieveBase::~SieveBase() {
	delete[] this->tracker_arr_;
}

std::string SieveBase::StringifyPrimes() {
	//Fix the string
	std::string ret_str = "";
	for (unsigned int i = 0; i < this->n_; i++) {
		if (this->tracker_arr_[i]) {
			ret_str += std::to_string(i) + ", ";
		}
	}

	ret_str.resize(ret_str.size()-2);

	//Return
	return ret_str;
}

std::string SieveBase::StringifyTrackerArr() {
	//Fix the string
	std::string ret_str = "";
	for (unsigned int i = 0; i < this->n_; i++) {
		ret_str += "[i=" + std::to_string(i) + "]:" + (this->tracker_arr_[i] ? "T\n" : "F\n");
	}

	//Return
	return ret_str;
}

std::string SieveBase::StringifyExecutionTime() {
	//Fix the string
	std::string ret_str = "";
	ret_str += this->private_timer_.StringifyLapTimes();
	ret_str += this->private_timer_.StringifyTotalTime();

	//Return
	return ret_str;
}

std::string SieveBase::StringifyResults(std::string in_title) {
	//Fix the string
	std::string ret_str = "";

	//Set title
	ret_str += "---" + in_title + "---\n";

	//Count number of primes found
	int num_of_p = 0;
	for (unsigned int i = 0; i < this->n_; i++) {
		if (this->tracker_arr_[i]) { num_of_p++; }
	}

	//Fill fields:
	ret_str += "Upper Limit 'n':\t" + std::to_string(this->n_) + "\n";
	ret_str += "Number of primes found:\t" + std::to_string(num_of_p) + "\n";
	ret_str += "Accuracy:\t\t[WIP]\n";
	ret_str += this->StringifyExecutionTime() + "\n";
	ret_str += "Identified Primes:\t" + this->StringifyPrimes() + "\n";

	//Return
	return ret_str;
}
