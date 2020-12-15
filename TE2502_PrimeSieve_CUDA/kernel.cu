#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <sstream>

#include <map>
#include <string>

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

	//SieveEratosthenesCPU eratosthenesA(n);
	//std::cout << eratosthenesA.StringifyResults("ERATOSTHENES CPU") << std::endl;
	//std::cout << eratosthenesA.StringifyTrackerArr() << std::endl;

	//SieveEratosthenesCUDA eratosthenesB(n);
	//std::cout << eratosthenesB.StringifyResults("ERATOSTHENES GPGPU") << std::endl;
	//std::cout << eratosthenesB.StringifyTrackerArr() << std::endl;

	//SieveSundaramCPU sundaramA(n);
	//std::cout << sundaramA.StringifyResults("SUNDARAM CPU") << std::endl;
	//std::cout << sundaramA.StringifyTrackerArr() << std::endl;
	
	//SieveSundaramCUDA sundaramB(n);
	//std::cout << sundaramB.StringifyResults("SUNDARAM GPGPU") << std::endl;
	//std::cout << sundaramB.StringifyTrackerArr() << std::endl;
	//sundaramB.SaveToFile("sieve results/", "test.txt");

	//SieveSundaramCUDABatches sundaramC(n);
	//std::cout << sundaramC.StringifyResults("SUNDARAM GPGPU (BATCHES") << std::endl;
	//std::cout << sundaramC.StringifyTrackerArr() << std::endl;

	//SieveAtkinCPU atkinA(n);
	//std::cout << atkinA.StringifyResults("ATKIN CPU") << std::endl;

	//SieveAtkinCUDA atkinB(n);
	//std::cout << atkinB.StringifyResults("ATKIN GPGPU") << std::endl;
	//std::cout << atkinA.StringifyTrackerArr() << std::endl;

	std::map<SieveType, std::string> m;
	m[ERATOSTHENES_CPU]		= "ERATOSTHENES_CPU";
	m[ERATOSTHENES_GPGPU]	= "ERATOSTHENES_CPU";
	m[SUNDARAM_CPU]			= "SUNDARAM_CPU";
	m[SUNDARAM_GPGPU]		= "SUNDARAM_GPGPU";
	m[ATKIN_CPU]			= "ATKIN_CPU";
	m[ATKIN_GPGPU]			= "ATKIN_GPGPU";

	size_t n = 65535;
	SieveType t = ERATOSTHENES_CPU;

	for (size_t n_i = 100; n_i <= n; n_i = n_i*100) {
		std::cout << ">Starting sieve " << m[t] << " (n=" << n_i << ")\n";
		SieveEratosthenesCPU theSieve(n_i);
		std::cout << ">Sieve done. Verifying and saving to file.\n"; 
		theSieve.SaveToFile("sieve results/", m[t] + ".txt");
		//std::cout << theSieve.StringifyResults("Results") << std::endl;
	}
	

	//---
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	std::cout << "<Program End>" << std::endl;

	WaitForEnter();

    return 0;
}



