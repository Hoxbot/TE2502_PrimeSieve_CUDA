#include "sieve_base.h"

#include <cmath>
#include <stdio.h>

#include <iostream>

#include "../support/rabin_miller_tester.h"

//Private------------------------------------------------------------------------------------------
SieveBase::VerificationData SieveBase::VerifyByFile() {
	/*
	//TEST
	FILE* file_test = nullptr;
	errno_t error_test;
	error_test = fopen_s(&file_test, "HERE", "w");
	if (file_test == nullptr) { return -1.0f; }
	fclose(file_test);
	//TEST
	*/

	//Return-value struct
	VerificationData ret_data;
	std::string missed_primes = "";
	std::string false_primes = "";

	//Constants
	const char* checkset_path = "resources/primes1_edit.txt";
	const int checkset_max = 15485863;

	//Return if end number too big
	if (this->end_ > checkset_max) { return ret_data; }

	//Open file
	FILE* file_ptr = nullptr;
	errno_t error;
	error = fopen_s(&file_ptr, checkset_path, "r");

	//Return if file couldn't be opened
	if (file_ptr == nullptr) { return ret_data; }

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
		if ((i != prime_from_file) && this->IsPrime(i)) { //NTS: IsPrime() is pure virtual
			num_of_misses++;
			false_primes += std::to_string(i) + ", ";
		}
		else if (i == prime_from_file) {
			if (!this->IsPrime(i)) { 
				num_of_misses++;
				missed_primes += std::to_string(i) + ", ";
			}
			read_next = true;
		}
	}

	//Close file
	fclose(file_ptr);

	//Calculate how many percent of the numbers where identified correctly
	size_t nums_in_memory = this->mem_class_ptr_->NumberCapacity();
	float percentage_correct = (1.0f - ((float)num_of_misses / (float)nums_in_memory))*100;

	//Add data to return struct
	ret_data.accuracy_str = std::to_string(percentage_correct);
	ret_data.accuracy_str.resize(ret_data.accuracy_str.size() - 3);										//Remove some 0:s
	//if (ret_data.miss_str.size() >= 2) { ret_data.miss_str.resize(ret_data.miss_str.size() - 2); }	//Remove the last ", " from the miss string (if there is one)
	if (!false_primes.empty() || !missed_primes.empty()) {
		ret_data.miss_str = "False Primes: <" + false_primes + ">\t";
		ret_data.miss_str += "Missed Primes: <" + missed_primes + ">";
	}

	//Return
	return ret_data;
}

SieveBase::VerificationData SieveBase::VerifyByRabinMiller() {
	//Return-value struct
	VerificationData ret_data;
	std::string false_primes = "";
	std::string missed_primes = "";
	
	//Create a Rabin-Miller tester (accuracy factor 4)
	RabinMillerTester tester(4);

	//Go through all numbers in range
	size_t num_of_misses = 0;
	for (size_t i = this->start_; i < this->end_; i++) {
		bool sieve_result = this->IsPrime(i);
		bool rabin_miller_result = tester.DoTest(i);

		//When the results differ, log false primes / misses
		if (sieve_result && !rabin_miller_result) {			//Sieve finding prime when RM doesn't -> false prime
			num_of_misses++;
			false_primes += std::to_string(i) + ", ";
		}
		else if (!sieve_result && rabin_miller_result) {	//Sieve not finding prime when RM does -> miss
			num_of_misses++;
			missed_primes += std::to_string(i) + ", ";
		}
	}

	//Calculate how many percent of the numbers where identified correctly
	size_t nums_in_memory = this->mem_class_ptr_->NumberCapacity();
	float percentage_correct = (1.0f - ((float)num_of_misses / (float)nums_in_memory)) * 100;

	//Add data to return struct
	ret_data.accuracy_str = std::to_string(percentage_correct);
	ret_data.accuracy_str.resize(ret_data.accuracy_str.size() - 3);										//Remove some 0:s
	//if (ret_data.miss_str.size() >= 2) { ret_data.miss_str.resize(ret_data.miss_str.size() - 2); }	//Remove the last ", " from the miss string (if there is one)
	if (!false_primes.empty() || !missed_primes.empty()) {
		ret_data.miss_str = "False Primes: <" + false_primes + ">\t";
		ret_data.miss_str += "Missed Primes: <" + missed_primes + ">";
	}

	//Return
	return ret_data;
}

SieveBase::VerificationData SieveBase::VerifyByEratosthenes() {
	//Return-value struct
	VerificationData ret_data;
	std::string false_primes = "";
	std::string missed_primes = "";

	//Create a memory to hold the memory of the checker sieve
	PrimeMemoryBool tester_mem(this->end_ + 1); //+1 : inclusive
	tester_mem.SetAllPrime();
	tester_mem.SetNonPrime(0);
	tester_mem.SetNonPrime(1);

	//Sieve via Eratosthenes
	unsigned int root_of_end = std::sqrt(this->end_) + 1;
	for (size_t i = 2; i < root_of_end; i++) {
		if (tester_mem.CheckIndex(i)) {
			for (size_t j = i * i; j <= this->end_; j = j + i) {
				tester_mem.SetNonPrime(j);
			}
		}
	}

	//Go through all numbers in range
	size_t num_of_misses = 0;
	for (size_t i = this->start_; i < this->end_; i++) {
		bool sieve_result = this->IsPrime(i);
		bool tester_result = tester_mem.CheckIndex(i);

		//When the results differ, log false primes / misses
		if (sieve_result && !tester_result) {			//Sieve finding prime when RM doesn't -> false prime
			num_of_misses++;
			false_primes += std::to_string(i) + ", ";
		}
		else if (!sieve_result && tester_result) {	//Sieve not finding prime when RM does -> miss
			num_of_misses++;
			missed_primes += std::to_string(i) + ", ";
		}
	}

	//Calculate how many percent of the numbers where identified correctly
	size_t nums_in_memory = this->mem_class_ptr_->NumberCapacity();
	float percentage_correct = (1.0f - ((float)num_of_misses / (float)nums_in_memory)) * 100;

	//Add data to return struct
	ret_data.accuracy_str = std::to_string(percentage_correct);
	ret_data.accuracy_str.resize(ret_data.accuracy_str.size() - 3);										//Remove some 0:s
	//if (ret_data.miss_str.size() >= 2) { ret_data.miss_str.resize(ret_data.miss_str.size() - 2); }	//Remove the last ", " from the miss string (if there is one)
	if (!false_primes.empty() || !missed_primes.empty()) {
		ret_data.miss_str = "False Primes: <" + false_primes + ">\t";
		ret_data.miss_str += "Missed Primes: <" + missed_primes + ">";
	}

	//Return
	return ret_data;
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
	//VerificationData v = this->VerifyByFile();
	//VerificationData v = this->VerifyByRabinMiller();
	VerificationData v = this->VerifyByEratosthenes();


	//Fill fields:
	ret_str += "Range:\t\t\t[" + std::to_string(this->start_) + ", " + std::to_string(this->end_) + "]\n";
	ret_str += "Numbers in memory:\t" + std::to_string(this->mem_class_ptr_->NumberCapacity()) + "\n";
	ret_str += "Number of primes found:\t" + std::to_string(num_of_p) + "\n";
	ret_str += "Accuracy:\t\t" + v.accuracy_str + "%\n";
	if (v.miss_str.size() != 0) { ret_str += "Misses:\t\t\t[" + v.miss_str + "]\n"; }
	ret_str += this->StringifyExecutionTime() + "\n";
	//ret_str += "Identified primes:\t" + this->StringifyPrimes() + "\n";

	//TEMP: Nulls string if accuracy_tring is empty (means we will only see errors)
	//if (v.miss_str.empty()) { ret_str = ""; } else { ret_str += "\n"; }

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

void SieveBase::SaveToFile(std::string in_folder_path, std::string in_file_name) {
	//Open file
	FILE* file_ptr = nullptr;
	errno_t error;
	error = fopen_s(&file_ptr, (in_folder_path + in_file_name).c_str(), "a");
	if (file_ptr == nullptr) { 
		std::cerr << ("Error: Could not open file '" + in_folder_path + in_file_name + "'\n");
		return;
	}


	//Calculate accuracy and format string
	//VerificationData v = this->VerifyByRabinMiller();
	VerificationData v = this->VerifyByEratosthenes();

	//Build line to be appended into file
	std::string str = "";
	std::string separator = "\t";
	str = std::to_string(this->end_) + separator;
	str += v.accuracy_str + separator;
	str += this->timer_.GetTotalSeparatorString(separator);
	str += this->timer_.GetLapsSeparatorString(separator);
	str += "\n"; //End entry

	//Write to file
	fwrite(str.c_str(), sizeof(char), str.size(), file_ptr);

	//Close file
	fclose(file_ptr);
}
