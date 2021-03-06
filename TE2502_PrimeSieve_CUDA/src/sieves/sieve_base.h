#ifndef SIEVE_BASE_H
#define SIEVE_BASE_H

#include <string>
#include <vector>

#include "prime_memory/prime_memory_bool.h"
#include "prime_memory/prime_memory_bit.h"
#include "../support/stat_handler.h"

#include "prime_memory/prime_memory_fragsafe.h"

class SieveBase {
private:
	struct VerificationData {
		std::string accuracy_str = "";
		std::string miss_str = "";

		std::vector<size_t> false_composites;
		std::vector<size_t> false_primes;
	};

	VerificationData VerifyByFile();
	VerificationData VerifyByRabinMiller();
	//VerificationData VerifyByEratosthenes();
	VerificationData VerifyByEratosthenes(PrimeMemoryFragsafe* in_ptr);
	VerificationData VerifyBySuppliedMemory(PrimeMemoryFragsafe* in_ptr);

	size_t CountNumbersInRegion(size_t in_start, size_t in_end, std::vector<size_t>& in_vec_ref);

protected:
	size_t start_ = 0;
	size_t end_ = 0;
	//size_t mem_size_ = 0;

	//PrimeMemory* mem_class_ptr_ = nullptr;
	PrimeMemoryBool* mem_class_ptr_ = nullptr;	//NTS: Currently only this works with GPU upload

	StatHandler timer_;

	virtual void DoSieve() = 0;
	virtual size_t IndexToNumber(size_t in_i) = 0;

public:
	SieveBase(size_t in_start, size_t in_end);
	~SieveBase();

	virtual bool IsPrime(size_t in_num) = 0;

	std::string StringifyPrimes();
	std::string StringifyTrackerArr();
	std::string StringifyExecutionTime();
	//std::string StringifyResults(std::string in_title);
	std::string StringifyResults(std::string in_title, PrimeMemoryFragsafe* in_ptr);


	std::vector<size_t> PrimeVector();
	
	//void SaveToFile(std::string in_folder_path, std::string in_file_name);
	void SaveToFile(std::string in_folder_path, std::string in_file_name, PrimeMemoryFragsafe* in_ptr);

	void SaveRegionalDataToFile(std::string in_folder_path, std::string in_file_name, std::string in_entry_name, PrimeMemoryFragsafe* in_ptr);

};

#endif // !SIEVE_BASE_H