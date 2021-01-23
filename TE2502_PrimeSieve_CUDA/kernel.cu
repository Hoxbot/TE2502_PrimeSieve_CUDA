#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <sstream>

#include <map>
#include <string>

#include <thread>       // std::this_thread::sleep_for
#include <chrono>       // std::chrono::seconds

#include <Windows.h>	// For memory specs

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

inline void OutputSpecs() {
	//Get Local System capabilities
	MEMORYSTATUSEX statex;
	statex.dwLength = sizeof(statex);
	GlobalMemoryStatusEx(&statex);

	//Get GPU capabilities
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	size_t mem_tot, mem_free;
	cudaMemGetInfo(&mem_free, &mem_tot);

	//Output
	std::cout << "\n";
	std::cout
		<< "\tC++ ver.: " << __cplusplus << "\n"
		<< "\tCUDA ver.: " << 10.2 << "\n";

	std::cout
		<< "\t---CPU SIDE---\n"
		<< "\tPhysical Memory:\n"
		<< "\t\tTotal:\t\t\t\t" << statex.ullTotalPhys << " bytes\n"
		<< "\t\tUsed:\t\t\t\t" << statex.ullTotalPhys - statex.ullAvailPhys << " bytes\n"
		<< "\t\tFree:\t\t\t\t" << statex.ullAvailPhys << " bytes\n"
		<< "\tPaging File:\n"
		<< "\t\tTotal:\t\t\t\t" << statex.ullTotalPageFile << " bytes\n"
		<< "\t\tUsed:\t\t\t\t" << statex.ullTotalPageFile - statex.ullAvailPageFile << " bytes\n"
		<< "\t\tFree:\t\t\t\t" << statex.ullAvailPageFile << " bytes\n"
		<< "\tVirtual memory:\n"
		<< "\t\tTotal:\t\t\t\t" << statex.ullTotalVirtual << " bytes\n"
		<< "\t\tUsed:\t\t\t\t" << statex.ullTotalVirtual - statex.ullAvailVirtual << " bytes\n"
		<< "\t\tFree:\t\t\t\t" << statex.ullAvailVirtual << " bytes\n"
		<< "\tExtended Memory:\n"
		<< "\t\tFree:\t\t\t\t" << statex.ullAvailExtendedVirtual << " bytes\n";

	std::cout
		<< "\t---CUDA SIDE---\n"
		<< "\tProperties:\n"
		<< "\t\tGlobal memory:\t\t\t" << prop.totalGlobalMem << " bytes\n"
		<< "\t\tShared memory:\t\t\t" << prop.sharedMemPerBlock << " bytes\n"
		<< "\t\tMax threads per block:\t\t" << prop.maxThreadsPerBlock << "\n"
		<< "\tMemory:\n"
		<< "\t\tTotal:\t\t\t\t" << mem_tot << " bytes\n"
		<< "\t\tFree:\t\t\t\t" << mem_free << " bytes\n";
	
	/*
	std::cout
		<< "\t---DATA TYPES---\n"
		<< "\tMax size_t value (index capacity):\t" << SIZE_MAX << "\n"
		<< "\tUnsigned int max value:\t\t\t" << UINT_MAX << "\n"
		<< "\tSize of size_t:\t\t\t\t" << sizeof(size_t) << "\n"
		<< "\n\n";
	*/

	std::cout << "\n";
}

enum SieveType {
	ERATOSTHENES_CPU,
	ERATOSTHENES_GPGPU,
	SUNDARAM_CPU,
	SUNDARAM_GPGPU,
	SUNDARAM_GPGPU_BATCH_DIVIDED,
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

	//Output system specs
	OutputSpecs();

	std::map<SieveType, std::string> m;
	m[ERATOSTHENES_CPU]		= "ERATOSTHENES_CPU";
	m[ERATOSTHENES_GPGPU]	= "ERATOSTHENES_GPGPU";
	m[SUNDARAM_CPU]			= "SUNDARAM_CPU";
	m[SUNDARAM_GPGPU]		= "SUNDARAM_GPGPU";
	m[ATKIN_CPU]			= "ATKIN_CPU";
	m[ATKIN_GPGPU]			= "ATKIN_GPGPU";

	//         3221225472
	//size_t n = 10000000000;		//10^10
	size_t n = 400;
	size_t n_s = 100;			//10^2
	unsigned int sleep_sec = 1;

	//Test
	size_t n = 10000000000;	//10^10 works
	//size_t n = 100000000000;	//10^11 doesn't	
	//	:	It is probably about it exceeding my RAM size (16 Gb)
	//	:	But then why does 10^10 work? That requires 20 Gb. Hmm...
	//	-> My virtual memory seems to allow ~56.2 GB
	// Any limit closing in on 1.6*10^10 makes memset in the SetPrimes functions slow as fuck
	//Test

	PrimeMemoryFragsafe* safe_mem_ptr = new PrimeMemoryFragsafe(n);
	PrimeMemoryFragsafe* verification_mem_ptr = new PrimeMemoryFragsafe(n);

	size_t bytes = safe_mem_ptr->BytesAllocated() + verification_mem_ptr->BytesAllocated();
	std::cout 
		<< ">Program FragSafe Memory Total:\n\t" 
		<< bytes << " bytes\n\t"
		<< (float)bytes/1000000000.f << " gigabytes\n";

	//OutputSpecs();

	//Test
	/*
	//Set verification memory using Atkin CPU
	std::cout << ">Setting verification memory\n";
	SieveAtkinCPU(n, verification_mem_ptr);

	//Do batched sieve
	std::cout << ">Starting sieve\n";
	SieveSundaramCUDABatches* sieve_ptr = new SieveSundaramCUDABatches(n, safe_mem_ptr);
	
	std::cout << ">Verifying\n";
	std::cout << sieve_ptr->StringifyResults("Sundaram Batches", verification_mem_ptr) << "\n";
	
	std::cout << ">Cleaning\n";
	delete sieve_ptr;
	*/
	//Test

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
	/*
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
	*/

	/*BATCH DIVIDED SUNDARAM (GENERAL RUN 2 TEMPLATE) */
	//Run a initializing GPGPU sieve
	std::cout << ">Running init sieve\n";
	SieveSundaramCUDA(10).SaveToFile("sieve results/", "_init_run.tsv");
	std::cout << ">Going to sleep.\n";
	std::this_thread::sleep_for(std::chrono::seconds(sleep_sec));

	//Select Sieve
	SieveType arr[3] = { ATKIN_CPU, SUNDARAM_GPGPU, SUNDARAM_GPGPU_BATCH_DIVIDED };
	for (size_t s_i = 0; s_i < 3; s_i++) {

		size_t inc = n_s;

		//Select Sieve Limit
		for (size_t n_i = n_s; n_i <= n; n_i = n_i + inc) {

			if (n_i >= 10 * inc) { inc *= 10; }	//Scales it to be 10 steps per iteration

			//Sieve 10 times on selected limit with selected sieve
			for (size_t i = 0; i < 10; i++) {
				SieveBase* sieve_ptr;

				std::cout << ">Starting sieve " << m[arr[s_i]] << " (n=" << n_i << ")\n";

				switch (arr[s_i]) {
				case SUNDARAM_GPGPU:
					//NTS: This sieve cannot go higher than the GPU memory limit
					if (n_i <= 2000000000) {	//2*10^9
						sieve_ptr = new SieveSundaramCUDA(n_i, safe_mem_ptr);
					}
					else {
						sieve_ptr = new SieveSundaramCUDA(10, safe_mem_ptr);
					}
					break;
				case SUNDARAM_GPGPU_BATCH_DIVIDED:
					sieve_ptr = new SieveSundaramCUDABatches(n_i, safe_mem_ptr);
					break;
				case ATKIN_GPGPU:
					sieve_ptr = new SieveAtkinCUDA(n_i, safe_mem_ptr);
					break;
				default:
					break;
				}

				std::cout << ">Sieve done. Verifying and saving to file.\n";
				sieve_ptr->SaveToFile("sieve results/", m[arr[s_i]] + "_6.tsv", verification_mem_ptr);
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



