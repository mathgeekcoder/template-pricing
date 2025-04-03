#include "utils.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX

#include <windows.h>
#include <processthreadsapi.h>
#include <psapi.h>

void MemoryUsage::snapshot() {
    PROCESS_MEMORY_COUNTERS pmc;

    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        peak = pmc.PeakWorkingSetSize;
    }
}

size_t MemoryUsage::PeakSize() {
    PROCESS_MEMORY_COUNTERS pmc;

    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize - peak;
    }

    return -1;
}

#endif
