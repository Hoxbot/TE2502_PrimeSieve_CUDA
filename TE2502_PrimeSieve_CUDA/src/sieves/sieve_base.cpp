#include "sieve_base.h"

#include <cmath>
#include <stdio.h>

//#include <iostream>

//Private------------------------------------------------------------------------------------------
float SieveBase::VerifyByFile() {
	/*
	//TEST
	FILE* file_test = nullptr;
	errno_t error_test;
	error_test = fopen_s(&file_test, "HERE", "w");
	if (file_test == nullptr) { return -1.0f; }
	fclose(file_test);
	//TEST
	*/

	//Constants
	const char* checkset_path = "resources/primes1_edit.txt";
	const int checkset_max = 15485863;

	//Return if end number too big
	if (this->end_ > checkset_max) { return -1.0f; }

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
	for (unsigned int i = 2; i <= this->end_; i++) {
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

		//std::cout << "Checking:\t" << i << "\t(" << this->IsPrime(i) << ")\n";

		//Compare i to the number from the file
		//->If i != that number, i should not be prime (if it is we log miss)
		//->If i == that number, i should be prime (if it isn't we log miss)
		//-->Once i == that number we should read the next number
		if ((i != prime_from_file) && this->IsPrime(i)) { num_of_misses++; }	//NTS: IsPrime() is pure virtual
		else if (i == prime_from_file) {
			if (!this->IsPrime(i)) { num_of_misses++; }
			read_next = true;
		}
	}

	//Close file
	fclose(file_ptr);

	//Calculate how many percent of the numbers where identified correctly
	size_t nums_in_memory = this->mem_class_ptr_->NumberCapacity();
	float percentage_correct = (1.0f - ((float)num_of_misses / (float)nums_in_memory))*100;

	//Return
	return percentage_correct;
}

//Public-------------------------------------------------------------------------------------------
SieveBase::SieveBase(size_t in_start, size_t in_end) {
	//Calculate the number of numbers in span
	this->start_ = in_start;
	this->end_ = in_end;
	//this->mem_size_ = in_mem_size;
}

SieveBase::~SieveBase() {
	/*
	if (this->mem_class_ptr != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}
	*/
}

std::string SieveBase::StringifyPrimes() {
	//Fix the string for every index in memory
	std::string ret_str = "";
	for (size_t i = 0; i < this->mem_class_ptr_->NumberCapacity(); i++) {
		if (this->mem_class_ptr_->CheckIndex(i)) {
			ret_str += std::to_string(this->IndexToNumber(i)) + ", ";
		}
	}

	//Remove the last ", "
	if (ret_str.size() >= 2) { ret_str.resize(ret_str.size() - 2); }

	//Return
	return ret_str;
}

std::string SieveBase::StringifyTrackerArr() {
	//Fix the string
	std::string ret_str = "";
	for (size_t i = 0; i < this->mem_class_ptr_->NumberCapacity(); i++) {
		ret_str += "[i=" + std::to_string(i) + "]\t:\t(" + std::to_string(this->IndexToNumber(i)) + (this->mem_class_ptr_->CheckIndex(i) ? ":T)\n" : ":F)\n");
	}

	//Return
	return ret_str;
}

std::string SieveBase::StringifyExecutionTime() {
	//Fix the string
	std::string ret_str = "";
	ret_str += this->timer_.StringifyLapTimes();
	ret_str += this->timer_.StringifyTotalTime();

	//Return
	return ret_str;
}

std::string SieveBase::StringifyResults(std::string in_title) {
	//Fix the string
	std::string ret_str = "";

	//Set title
	ret_str += "---" + in_title + "---\n";

	//Loop over memory, count number of primes found
	int num_of_p = 0;
	for (size_t i = 0; i < this->mem_class_ptr_->NumberCapacity(); i++) {
		if (this->mem_class_ptr_->CheckIndex(i)) { num_of_p++; }
	}

	//Calculate accuracy and format string
	float accuracy = this->VerifyByFile();
	std::string accuracy_str = std::to_string(accuracy);
	accuracy_str.resize(accuracy_str.size() - 3);	//Remove some 0:s

	//Fill fields:
	ret_str += "Range:\t\t\t[" + std::to_string(this->start_) + ", " + std::to_string(this->end_) + "]\n";
	ret_str += "Numbers in memory:\t" + std::to_string(this->mem_class_ptr_->NumberCapacity()) + "\n";
	//ret_str += "Number of primes found:\t" + std::to_string(num_of_p) + "\n";
	//ret_str += "Accuracy:\t\t" + accuracy_str + "%\n";
	ret_str += this->StringifyExecutionTime() + "\n";
	//ret_str += "Identified primes:\t" + this->StringifyPrimes() + "\n";

	//Return
	return ret_str;
}

std::vector<size_t> SieveBase::PrimeVector() {
	std::vector<size_t> ret_vec;

	for (size_t i = 0; i < this->mem_class_ptr_->NumberCapacity(); i++) {
		if (this->mem_class_ptr_->CheckIndex(i)) {
			ret_vec.push_back(this->IndexToNumber(i));
		}
	}

	return ret_vec;
}
