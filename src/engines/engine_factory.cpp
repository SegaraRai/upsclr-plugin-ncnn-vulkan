// Super-resolution engine factory implementation
#include "engine_factory.hpp"

#include <algorithm>
#include <stdexcept>

#include "encoding_utils.hpp"

// Include all engine headers
#include "realcugan/realcugan.hpp"
#include "realesrgan/realesrgan.hpp"
// #include "waifu2x/waifu2x.hpp"

static std::vector<std::u8string> ALL_ENGINE_NAMES{
    u8"realesrgan",
    u8"realcugan",
    // u8"waifu2x",
};

const std::vector<std::u8string>& SuperResolutionEngineFactory::get_available_engines() {
    return ALL_ENGINE_NAMES;
}

const SuperResolutionEngineInfo* SuperResolutionEngineFactory::get_engine_info(const std::u8string& engine_name) {
    if (engine_name == u8"realesrgan") {
        return &RealESRGAN::get_engine_info();
    }
    if (engine_name == u8"realcugan") {
        return &RealCUGAN::get_engine_info();
    }
    // if (engine_name == u8"waifu2x") {
    //     return &Waifu2x::get_engine_info();
    // }

    return nullptr;
}

std::unique_ptr<SuperResolutionEngine> SuperResolutionEngineFactory::create_engine(
    const SuperResolutionEngineConfig& config) {
    if (config.engine_name == u8"realesrgan") {
        return std::make_unique<RealESRGAN>(config);
    }
    if (config.engine_name == u8"realcugan") {
        return std::make_unique<RealCUGAN>(config);
    }
    // if (config.engine_name == u8"waifu2x") {
    //     return std::make_unique<Waifu2x>(config);
    // }

    return nullptr;
}
