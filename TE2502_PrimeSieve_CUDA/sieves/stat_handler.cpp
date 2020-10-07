#include "stat_handler.h"

float StatHandler::ConvertToMicroSeconds(unsigned int a, unsigned int b) {
	
	float t = 
		std::chrono::duration_cast<std::chrono::microseconds>(
			this->time_points[b] - this->time_points[a]
			).count();
	
	return t;
}

float StatHandler::ConvertToSeconds(unsigned int a, unsigned int b) {

	float t =
		std::chrono::duration_cast<std::chrono::seconds>(
			this->time_points[b] - this->time_points[a]
			).count();

	return t;
}

StatHandler::StatHandler() {
}

StatHandler::~StatHandler() {
	this->ClearTimes();
}

void StatHandler::ClearTimes() {
	this->time_points.clear();
}

void StatHandler::SaveTime() {
	this->time_points.push_back(std::chrono::high_resolution_clock::now());
}

std::string StatHandler::StringifyLapTimes() {
	std::string ret_str = "";

	ret_str = "Times (ms):\n";

	for (unsigned int i = 1; i < this->time_points.size(); i++) {
		ret_str += std::to_string(i) + ":\t";
		ret_str += "Elapsed: " + std::to_string(this->ConvertToMicroSeconds(0, i)) + "\t";
		ret_str += "Lap: " + std::to_string(this->ConvertToMicroSeconds(i - 1, i)) + "\n";
	}

	return ret_str;
}

std::string StatHandler::StringifyTotalTime() {

	std::string ret_str = "";
	
	ret_str = "Total Time (ms): " + std::to_string(this->ConvertToMicroSeconds(0, (this->time_points.size() - 1)));

	return ret_str;
}

void StatHandler::WriteTimesToFile(std::string in_header, std::string in_path) {
	if (this->time_points.size() < 2) { return; }
	
	//Open file
	FILE* file_ptr = nullptr;
	errno_t error;
	error = fopen_s(&file_ptr, in_path.c_str(), "wb+"); //Plus overwrites file contents
	if (file_ptr == nullptr) { return; }

	//Write header
	fwrite(in_header.c_str(), sizeof(char), in_header.size(), file_ptr);
	fwrite("\n", sizeof(char), 1, file_ptr);

	//Create a string for each span
	std::string s = "";
	for (unsigned int i = 1; i < this->time_points.size(); i++) {
		s = "";
		s += "Time Point " + std::to_string(i) + ":\t";
		s += std::to_string(this->ConvertToMicroSeconds(i-1, i));
		
		fwrite(s.c_str(), sizeof(char), s.size(), file_ptr);
		fwrite("\n", sizeof(char), 1, file_ptr);
	}

	fwrite("\n", sizeof(char), 1, file_ptr);
	s = "";
	s += "Total Time:\t";
	s += std::to_string(this->ConvertToMicroSeconds(0, (this->time_points.size()-1)));
	fwrite(s.c_str(), sizeof(char), s.size(), file_ptr);
	fwrite("\n", sizeof(char), 1, file_ptr);
	

	//Once done, close file and delete buffer
	fclose(file_ptr);
}
