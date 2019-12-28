#ifdef _WIN32
#include <intrin.h>
//surpress crash notification windows (close or debug program window)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <x86intrin.h>
#endif
#include <string>
#include "Executor.h"

int main(int argc, char *argv[])
{
#ifdef _WIN32
	//surpress crash notification windows (close or debug program window)
	SetErrorMode(GetErrorMode() | SEM_NOGPFAULTERRORBOX);
#endif

	if (argc < 2)
	{
		printf("no .mtx file path set. please call using 'runspECK.exe C:/Path/To/matrix.mtx' or linux equivalent");
		return -1;
	}

	std::string value_type = argc > 7 ? argv[7] : "d";

	// if (value_type == "f")
	//  	return Executor<float>(argc, argv).run();
	// else
	return Executor<double>(argc, argv).run();
}
