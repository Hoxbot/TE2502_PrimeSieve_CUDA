#include "sieve_base.h"

#include <cmath>
#include <stdio.h>

//Private------------------------------------------------------------------------------------------
bool SieveBase::CheckIndex(unsigned int in_i) {
	return this->tracker_arr_[in_i];
}

void SieveBase::SetNonPrime(unsigned int in_i) {
	this->tracker_arr_[in_i] = false;
}

void SieveBase::SetPrime(unsigned int in_i) {
	this->tracker_arr_[in_i] = true;
}

float SieveBase::VerifyByFile() {

	//Constants
	const char* checkset_path = "resources/primes1_edit.txt";
	const int checkset_max = 15485863;

	//Return if n too big
	if (this->n_ > checkset_max) { return -1.0f; }

	//Open file
	FILE* file_ptr = nullptr;
	errno_t error;
	error = fopen_s(&file_ptr, checkset_path, "r");

	//Return if file couldn't be opened
	if (file_ptr == nullptr) { return -1.0f; }

	//Loop over all numbers from [2, n]
	bool read_next = true;
	unsigned int prime_from_file = 0;
	unsigned int num_of_misses = 0;
	for (unsigned int i = 2; i <= this->n_; i++) {
		//Read the next number from the file
		if (read_next) {
			char c = ' ';
			std::string str = "";

			//First read all non-numbers
			while (c < '0' || c > '9') {
				fread(&c, sizeof(char), 1, file_ptr);
			}

			//When a number is found add it to string
			//and then read all following numbers to add then as well
			while (c >= '0' && c <= '9') {
				str += c;
				fread(&c, sizeof(char), 1, file_ptr);
			}

			//Save the number found
			prime_from_file = std::stoi(str);

			//Set false
			read_next = false;
		}

		//Compare i to the number from the file
		//->If i != that number, i should not be prime (if it is we log miss)
		//->If i == that number, i should be prime (if it isn't we log miss)
		//-->Once i == that number we should read the next number
		if ((i != prime_from_file) && this->IsPrime(i)) { num_of_misses++; }
		else if (i == prime_from_file) {
			if (!this->IsPrime(i)) { num_of_misses++; }
			read_next = true;
		}
	}

	//Close file
	fclose(file_ptr);

	//Calculate how many percent of the numbers where identified correctly
	float percentage_correct = (1.0f - ((float)num_of_misses / (float)(this->n_ - 1)))*100;

	//Return
	return percentage_correct;
}

//Public-------------------------------------------------------------------------------------------
SieveBase::SieveBase(unsigned int in_n, unsigned int in_first_val) {
	this->n_ = in_n;
	
	this->index_offset_ = in_first_val;
	this->n_alt_ = in_n - (in_first_val - 1);
	
	this->tracker_arr_ = new bool[this->n_alt_]; //NTS: indexes [0, n-1]
	
									   //Start all values as true ("known as primes")
	for (unsigned int i = 0; i < this->n_alt_; i++) {
		this->SetPrime(i);
	}
}

SieveBase::~SieveBase() {
	delete[] this->tracker_arr_;
}

bool SieveBase::IsPrime(unsigned int in_num) {
	//Everything outside scope is false
	if (in_num > this->n_) { return false; }
	//Otherwise return the stored bool for that value
	return this->CheckIndex(in_num-this->index_offset_);
}

std::string SieveBase::StringifyPrimes() {
	//Fix the string
	std::string ret_str = "";
	for (unsigned int i = 0; i < this->n_alt_; i++) {
		if (this->CheckIndex(i)) {
			ret_str += std::to_string(i+this->index_offset_) + ", ";
		}
	}

	ret_str.resize(ret_str.size()-2);

	//Return
	return ret_str;
}

std::string SieveBase::StringifyTrackerArr() {
	//Fix the string
	std::string ret_str = "";
	for (unsigned int i = 0; i < this->n_alt_; i++) {
		ret_str += "[i=" + std::to_string(i) + "]\t:\t(" + std::to_string(i + this->index_offset_) + (this->CheckIndex(i) ? ":T)\n" : ":F)\n");
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
	for (unsigned int i = 0; i < this->n_alt_; i++) {
		if (this->CheckIndex(i)) { num_of_p++; }
	}

	//Calculate accuracy and format string
	float accuracy = this->VerifyByFile();
	std::string accuracy_str = std::to_string(accuracy);
	accuracy_str.resize(accuracy_str.size() - 3);

	//Fill fields:
	ret_str += "Upper Limit 'n':\t" + std::to_string(this->n_) + "\n";
	ret_str += "Number of primes found:\t" + std::to_string(num_of_p) + "\n";
	ret_str += "Accuracy:\t\t" + accuracy_str + "%\n";
	ret_str += this->StringifyExecutionTime() + "\n";
	ret_str += "Identified Primes:\t" + this->StringifyPrimes() + "\n";

	//Return
	return ret_str;
}

std::vector<int> SieveBase::PrimeVector() {
	std::vector<int> ret_vec;

	for (unsigned int i = 0; i < this->n_alt_; i++) {
		if (this->CheckIndex(i)) {
			ret_vec.push_back(i + this->index_offset_);
		}
	}

	return ret_vec;
}
