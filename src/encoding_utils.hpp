#ifndef ENCODING_UTILS_HPP
#define ENCODING_UTILS_HPP

#include <string>
#include <string_view>

inline std::string as_string(const std::u8string_view str) {
    return std::string(reinterpret_cast<const char*>(str.data()), str.length());
}

inline std::string utf8_to_ascii(const std::u8string_view str) {
    return as_string(str);
}

inline std::u8string as_utf8(const std::string_view str) {
    return std::u8string(reinterpret_cast<const char8_t*>(str.data()), str.length());
}

inline std::u8string ascii_to_utf8(const std::string_view str) {
    return as_utf8(str);
}

#endif
