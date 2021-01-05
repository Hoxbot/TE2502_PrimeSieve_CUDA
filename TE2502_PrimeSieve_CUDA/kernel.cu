#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <sstream>

#include <map>
#include <string>

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

//For memory leaks
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

//CPU Sieves
#include "src/sieves/sieve_eratosthenes_cpu.h"
#include "src/sieves/sieve_sundaram_cpu.h"
#include "src/sieves/sieve_atkin_cpu.h"

//GPGPU Sieves
#include "src/sieves/sieve_eratosthenes_cuda.cuh"
#include "src/sieves/sieve_sundaram_cuda.cuh"
#include "src/sieves/sieve_sundaram_cuda_batches.cuh"
#include "src/sieves/sieve_atkin_cuda.cuh"

//Memory
#include "src/sieves/prime_memory/prime_memory_fragsafe.h"

//Misc
inline void WaitForEnter() {
	std::string str;
	std::cout << "Enter to continue..." << std::endl;
	std::getline(std::cin, str);
}

enum SieveType {
	ERATOSTHENES_CPU,
	ERATOSTHENES_GPGPU,
	SUNDARAM_CPU,
	SUNDARAM_GPGPU,
	ATKIN_CPU,
	ATKIN_GPGPU,
	ENUM_END,
};

int main() {
	//Check for memory leaks at each exit point of the program
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//---
	std::cout << "<Program Start>" << std::endl;
	

	/*
	//TEST
	FILE* file_test = nullptr;
	errno_t error_test;
	error_test = fopen_s(&file_test, "HERE", "w");
	if (file_test == nullptr) { return -1.0f; }
	fclose(file_test);
	//TEST
	*/

	//Get GPU capabilities
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout
		<< "\tC++ ver.: " << __cplusplus << "\n"
		<< "\tCUDA ver.: " << 10.2 << "\n"
		<< "\t---CPU SIDE---\n"
		<< "\tMax allocation capacity (bytes):\t" << SIZE_MAX << "\n"
		<< "\t---CUDA SIDE---\n"
		<< "\tGlobal memory capacity (bytes):\t\t" << prop.totalGlobalMem << "\n"
		<< "\tShared memory capacity (bytes):\t\t" << prop.sharedMemPerBlock << "\n"
		<< "\tMax threads per block:\t\t\t" << prop.maxThreadsPerBlock << "\n"
		<< "\t---DATA TYPES---\n"
		<< "\tUnsigned int max:\t\t\t" << UINT_MAX << "\n"
		<< "\tSize of size_t:\t\t\t\t" << sizeof(size_t) << "\n"
		<< "\n\n";
	

	//size_t n = 1024*100 + 522;
	//size_t n = ((size_t)3221225472) * 11;	//WORKING HERE: Only requires 1 batch, it should need 10. Overflow somewhere?
	//size_t n = 65535;

	std::map<SieveType, std::string> m;
	m[ERATOSTHENES_CPU]		= "ERATOSTHENES_CPU";
	m[ERATOSTHENES_GPGPU]	= "ERATOSTHENES_GPGPU";
	m[SUNDARAM_CPU]			= "SUNDARAM_CPU";
	m[SUNDARAM_GPGPU]		= "SUNDARAM_GPGPU";
	m[ATKIN_CPU]			= "ATKIN_CPU";
	m[ATKIN_GPGPU]			= "ATKIN_GPGPU";

	//         3221225472
	size_t n = 1000000000;
	//size_t n = 1000;
	size_t n_s = 100;
	//size_t n_s = 100000000;
	unsigned int sleep_sec = 1;

	PrimeMemoryFragsafe* safe_mem_ptr = new PrimeMemoryFragsafe(n);
	PrimeMemoryFragsafe* verification_mem_ptr = new PrimeMemoryFragsafe(n);

	//SieveEratosthenesCPU(n, safe_mem_ptr).SaveToFile("sieve results/", "fragsafetest.tsv", verification_mem_ptr);
	//SieveEratosthenesCUDA(n, safe_mem_ptr).SaveToFile("sieve results/", "fragsafetest.tsv", verification_mem_ptr);

	/* GENERAL RUN */
	/*
	for (SieveType t = ERATOSTHENES_CPU; t < ENUM_END; t = (SieveType)((unsigned int)t + 1)) {

		size_t inc = n_s;

		for (size_t n_i = n_s; n_i <= n; n_i = n_i + inc) {

			if (n_i >= 10 * inc) { inc *= 10; }	//Scales it to be 10 steps per iteration

			SieveBase* sieve_ptr;

			std::cout << ">Starting sieve " << m[t] << " (n=" << n_i << ")\n";

			switch (t) {
			case ERATOSTHENES_CPU:
				sieve_ptr = new SieveEratosthenesCPU(n_i);
				break;
			case ERATOSTHENES_GPGPU:
				sieve_ptr = new SieveEratosthenesCUDA(n_i);
				break;
			case SUNDARAM_CPU:
				sieve_ptr = new SieveSundaramCPU(n_i);
				break;
			case SUNDARAM_GPGPU:
				sieve_ptr = new SieveSundaramCUDA(n_i);
				break;
			case ATKIN_CPU:
				sieve_ptr = new SieveAtkinCPU(n_i);
				break;
			case ATKIN_GPGPU:
				sieve_ptr = new SieveAtkinCUDA(n_i);
				break;
			default:
				break;
			}

			std::cout << ">Sieve done. Verifying and saving to file.\n";
			sieve_ptr->SaveToFile("sieve results/", m[t] + "_4.tsv");
			//std::cout << sieve_ptr->StringifyResults("Results") << std::endl;

			delete sieve_ptr;
		}
	}
	*/
	
	/* COUNTING NUMBER OF PRIMES */
	/*
	SieveSundaramCUDA(n).SaveRegionalDataToFile("sieve results/", "region_data.tsv", "SoS-CUDA:");
	for (size_t i = 0; i < 10; i++) {
		SieveAtkinCUDA(n).SaveRegionalDataToFile("sieve results/", "region_data.tsv", "SoA-CUDA" + std::to_string(i) + ":");
	}
	*/

	/*GENERAL RUN 2 */
	//Run a initializing GPGPU sieve
	std::cout << ">Running init sieve\n";
	SieveSundaramCUDA(10).SaveToFile("sieve results/", "_init_run.tsv");
	std::cout << ">Going to sleep.\n";
	std::this_thread::sleep_for(std::chrono::seconds(sleep_sec));

	//Select Sieve
	for (SieveType t = ERATOSTHENES_CPU; t < ENUM_END; t = (SieveType)((unsigned int)t + 1)) {

		size_t inc = n_s;
		
		//Select Sieve Limit
		for (size_t n_i = n_s; n_i <= n; n_i = n_i + inc) {

			if (n_i >= 10 * inc) { inc *= 10; }	//Scales it to be 10 steps per iteration

			//Sieve 10 times on selected limit with selected sieve
			for (size_t i = 0; i < 10; i++) {
				SieveBase* sieve_ptr;

				std::cout << ">Starting sieve " << m[t] << " (n=" << n_i << ")\n";

				switch (t) {
				case ERATOSTHENES_CPU:
					sieve_ptr = new SieveEratosthenesCPU(n_i, safe_mem_ptr);
					break;
				case ERATOSTHENES_GPGPU:
					sieve_ptr = new SieveEratosthenesCUDA(n_i, safe_mem_ptr);
					break;
				case SUNDARAM_CPU:
					sieve_ptr = new SieveSundaramCPU(n_i, safe_mem_ptr);
					break;
				case SUNDARAM_GPGPU:
					sieve_ptr = new SieveSundaramCUDA(n_i, safe_mem_ptr);
					break;
				case ATKIN_CPU:
					sieve_ptr = new SieveAtkinCPU(n_i, safe_mem_ptr);
					break;
				case ATKIN_GPGPU:
					sieve_ptr = new SieveAtkinCUDA(n_i, safe_mem_ptr);
					break;
				default:
					break;
				}

				std::cout << ">Sieve done. Verifying and saving to file.\n";
				sieve_ptr->SaveToFile("sieve results/", m[t] + "_5.tsv", verification_mem_ptr);
				//std::cout << sieve_ptr->StringifyResults("Results") << std::endl;

				delete sieve_ptr;

				//Sleep for x sec to ensure program has time to deallocate memory properly
				std::cout << ">Going to sleep.\n";
				std::this_thread::sleep_for(std::chrono::seconds(sleep_sec));
				
			}	
		}
	}

	//---
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	//Clear fragsafe memories
	delete safe_mem_ptr;
	delete verification_mem_ptr;

	std::cout << "<Program End>" << std::endl;

	WaitForEnter();

    return 0;
}



