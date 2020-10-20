#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <sstream>

//For memory leaks
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

//Sieves
#include "src/sieves/sieve_eratosthenes_cpu.h"

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

	std::cout << "UINT_MAX:\t\t" << UINT_MAX << std::endl;

	std::cout << "SIZE_MAX:\t\t" << SIZE_MAX << std::endl; //<- An array cannot be larger (index-wise) than this
	std::cout << "int[SIZE_MAX] bytes:\t" << (SIZE_MAX * 4) << std::endl;
	std::cout << "int[SIZE_MAX] bits:\t" << (SIZE_MAX * 4 * 8) << std::endl;

	std::cout << "SIZE_MAX+10:\t\t" << SIZE_MAX+10 << std::endl;

	//int* ptr = new int[(SIZE_MAX / 4)];
	//int* ptr = new int[(SIZE_MAX/4 + 1)];
	//delete[] ptr;

	//SieveErathosthenesCPU eratosthenesA(50);
	//std::cout << eratosthenesA.StringifyResults("ERATOSTHENES CPU") << std::endl;
	//std::cout << eratosthenesA.StringifyTrackerArr() << std::endl;

	//std::cout << ": " << eratosthenesA.IsPrime(2567) << std::endl;

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



