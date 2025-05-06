#include <Windows.h>
#include <cstdio>
#include <exception>

namespace {

struct CHCP_UTF8 {
    static UINT old;

    CHCP_UTF8() noexcept {
        const UINT cp = GetConsoleOutputCP();
        if (cp != 0 && cp != CP_UTF8) {
            CHCP_UTF8::old = cp;
            SetConsoleOutputCP(CP_UTF8);
            std::set_terminate(restore);
        }
    }

    ~CHCP_UTF8() noexcept {
        restore();
    }

    static void restore() noexcept {
        const UINT cp = CHCP_UTF8::old;
        if (cp != 0 && cp != CP_UTF8) {
            SetConsoleOutputCP(cp);
            CHCP_UTF8::old = 0;
        }
    }

} auto_chcp;

UINT CHCP_UTF8::old = 0;

}  // namespace
