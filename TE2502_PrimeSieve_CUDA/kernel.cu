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

int main() {
	//Check for memory leaks at each exit point of the program
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//---
	std::cout << "<Program Start>" << std::endl;
	
	//Get GPU capabilities
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout
		<< "\t---CPU SIDE---\n"
		<< "\tMax allocation capacity (bytes):\t" << SIZE_MAX << "\n"
		<< "\t---CUDA SIDE---\n"
		<< "\tGlobal memory capacity (bytes):\t\t" << prop.totalGlobalMem << "\n"
		<< "\tShared memory capacity (bytes):\t\t" << prop.sharedMemPerBlock << "\n"
		<< "\tMax threads per block:\t\t\t" << prop.maxThreadsPerBlock << "\n"
		<< "\t---DATA TYPES---\n"
		<< "\tUnsigned int max:\t\t\t" << UINT_MAX << "\n"
		<< "\n\n";
	

	//size_t n = 1024*90 + 522;
	//size_t n = ((size_t)3221225472) * 11;	//WORKING HERE: Only requires 1 batch, it should need 10. Overflow somewhere?
	size_t n = 65535*2;

	//SieveEratosthenesCPU eratosthenesA(n);
	//std::cout << eratosthenesA.StringifyResults("ERATOSTHENES CPU") << std::endl;
	//std::cout << eratosthenesA.StringifyTrackerArr() << std::endl;

	SieveEratosthenesCUDA eratosthenesB(n);
	std::cout << eratosthenesB.StringifyResults("ERATOSTHENES GPGPU") << std::endl;
	//std::cout << eratosthenesB.StringifyTrackerArr() << std::endl;

	//SieveSundaramCPU sundaramA(n);
	//std::cout << sundaramA.StringifyResults("SUNDARAM CPU") << std::endl;
	//std::cout << sundaramA.StringifyTrackerArr() << std::endl;
	
	SieveSundaramCUDA sundaramB(n);
	std::cout << sundaramB.StringifyResults("SUNDARAM GPGPU") << std::endl;
	//std::cout << sundaramB.StringifyTrackerArr() << std::endl;

	//SieveSundaramCUDABatches sundaramC(n);
	//std::cout << sundaramC.StringifyResults("SUNDARAM GPGPU (BATCHES") << std::endl;
	//std::cout << sundaramC.StringifyTrackerArr() << std::endl;

	SieveAtkinCUDA atkinA(n);
	std::cout << atkinA.StringifyResults("ATKIN GPGPU") << std::endl;
	//std::cout << atkinA.StringifyTrackerArr() << std::endl;


	//std::cout << SieveAtkinCUDA(n-1).StringifyResults("Atkin A") << std::endl;
	//std::cout << SieveAtkinCUDA(n).StringifyResults("Atkin B") << std::endl;
	
	
	//for (size_t i = 100; i <= 200; i++) {
	//	std::cout << SieveAtkinCUDA(n-1).StringifyResults("Atkins error at n=" + std::to_string(i));
	//	//std::cout << SieveSundaramCUDA(n-1).StringifyResults("Sundaram error at n=" + std::to_string(i));
	//}

	//std::cout << SieveAtkinCPU(n-1).StringifyResults("CPU");
	//std::cout << SieveAtkinCUDA(n-1).StringifyResults("CUDA");

	//Allocation test
	//std::cout << SieveEratosthenesCPU(n).StringifyResults("FIRST") << std::endl;
	//std::cout << SieveSundaramCPU(n).StringifyResults("SECOND") << std::endl;
	//std::cout << SieveAtkinCPU(n).StringifyResults("THIRD") << std::endl;
	//std::cout << SieveSundaramCPU(12023).StringifyResults("FOURTH") << std::endl;

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



