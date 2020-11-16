#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <sstream>

//For memory leaks
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

//CPU Sieves
#include "src/sieves/sieve_eratosthenes_cpu.h"
#include "src/sieves/sieve_atkin_cpu.h"

//GPGPU Sieves
#include "src/sieves/sieve_sundaram_cuda.cuh"
#include "src/sieves/sieve_atkin_cuda.cuh"

//Misc
inline void WaitForEnter() {
	std::string str;
	std::cout << "Enter to continue..." << std::endl;
	std::getline(std::cin, str);
}

int main() {
	//Check for memory leaks at each exit point of the program
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//---
	std::cout << "<Program Start>" << std::endl;

	/*
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout 
		<< "\t---CPU SIDE---\n"
		<< "\tMax allocation capacity (bytes):\t" << SIZE_MAX << "\n"
		<< "\t---CUDA SIDE---\n"
		<< "\tGlobal memory capacity (bytes):\t\t" << prop.totalGlobalMem << "\n"
		<< "\tShared memory capacity (bytes):\t\t" << prop.sharedMemPerBlock << "\n"
		<< "\tMax threads per block:\t\t\t" << prop.maxThreadsPerBlock << "\n";
	*/

	size_t n = 1024;

	//SieveErathosthenesCPU eratosthenesA(n);
	//std::cout << eratosthenesA.StringifyResults("ERATOSTHENES CPU") << std::endl;
	//std::cout << eratosthenesA.StringifyTrackerArr() << std::endl;

	//SieveSundaramCUDA sundaramA(n);
	//std::cout << sundaramA.StringifyResults("SUNDARAM GPGPU") << std::endl;
	//std::cout << sundaramA.StringifyTrackerArr() << std::endl;

	//SieveAtkinCUDA atkinA(n);
	//std::cout << atkinA.StringifyResults("ATKIN GPGPU") << std::endl;
	//std::cout << atkinA.StringifyTrackerArr() << std::endl;

	//----
	for (size_t i = 1367; i < 1371; i++) {
		std::cout << "#########################################\n";
		std::cout << "CPU\n";
		std::cout << SieveAtkinCPU(i).StringifyResults(" ") << std::endl;
		std::cout << "CUDA\n";
		std::cout << SieveAtkinCUDA(i).StringifyResults(" ") << std::endl;
	}
	//----

	//Allocation test
	//std::cout << SieveSundaramCUDA(1024).StringifyResults("FIRST") << std::endl;
	//std::cout << SieveSundaramCUDA(1024).StringifyResults("SECOND") << std::endl;
	//std::cout << SieveSundaramCUDA(2048).StringifyResults("THIRD") << std::endl;
	//std::cout << SieveSundaramCUDA(4096).StringifyResults("FOURTH") << std::endl;

	//---
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	std::cout << "<Program End>" << std::endl;

	//WaitForEnter();

    return 0;
}



