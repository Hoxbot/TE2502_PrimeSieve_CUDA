#ifndef STAT_HANDLER_H
#define STAT_HANDLER_H

#include <chrono>
#include <vector>

#include <string>
#include <stdio.h>

class StatHandler {
private:
	std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> time_points;

	float ConvertToMicroSeconds(unsigned int a, unsigned int b);
	float ConvertToSeconds(unsigned int a, unsigned int b);

public:
	StatHandler();
	~StatHandler();

	void ClearTimes();
	void SaveTime();

	std::string StringifyLapTimes();
	std::string StringifyTotalTime();

	void WriteTimesToFile(std::string in_header, std::string in_path);
};

#endif // !STAT_HANDLER_H
