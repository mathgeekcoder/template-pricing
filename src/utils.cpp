#include "utils.h"

std::vector<std::function<void()>*> HandleCtrlC::handlers;
std::mutex HandleCtrlC::handlers_mutex;

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

static BOOL WINAPI WinCtrlHandler(_In_ DWORD dwCtrlType) {
	if (dwCtrlType == CTRL_C_EVENT) {
		HandleCtrlC::call_all_handlers();
		return TRUE;
	}

    return FALSE;
}

bool HandleCtrlC::Enable() {
    return SetConsoleCtrlHandler(WinCtrlHandler, TRUE) != 0;
}

void HandleCtrlC::Disable() {
    SetConsoleCtrlHandler(WinCtrlHandler, FALSE);
}

#else
#include <signal.h>

void PosixCtrlHandler(int sig) {
    HandleCtrlC::call_all_handlers();
}

bool HandleCtrlC::Enable() {
    struct sigaction action;
    action.sa_handler = PosixCtrlHandler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    return (sigaction(SIGINT, &action, nullptr) == 0);
}

void HandleCtrlC::Disable() {
    struct sigaction action;
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, nullptr);
}

#endif
