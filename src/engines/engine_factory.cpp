// Super-resolution engine factory implementation
#include "engine_factory.hpp"

#include <algorithm>
#include <stdexcept>

#include "encoding_utils.hpp"

// Include all engine headers
#include "realcugan/realcugan.hpp"
#include "realesrgan/realesrgan.hpp"
// #include "waifu2x/waifu2x.hpp"

// Helper function to register all engines
static std::vector<std::u8string> get_all_engine_names() {
    return {
        u8"realesrgan",
        u8"realcugan",
        // u8"waifu2x"
    };
}

std::vector<std::u8string> SuperResolutionEngineFactory::get_available_engines() {
    return get_all_engine_names();
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
    const std::u8string& engine_name,
    const SuperResolutionEngineConfig& config) {
    SuperResolutionEngineConfig cfg = config;
    cfg.engine_name = engine_name;

    if (engine_name == u8"realesrgan") {
        return std::make_unique<RealESRGAN>(cfg);
    }
    if (engine_name == u8"realcugan") {
        return std::make_unique<RealCUGAN>(cfg);
    }
    // if (engine_name == u8"waifu2x") {
    //     return std::make_unique<Waifu2x>(cfg);
    // }

    throw std::runtime_error("Unknown engine: " + utf8_to_ascii(engine_name));
}

std::unique_ptr<SuperResolutionEngine> SuperResolutionEngineFactory::create_engine(
    const SuperResolutionEngineConfig& config) {
    if (!config.engine_name.empty()) {
        return create_engine(config.engine_name, config);
    }

    return nullptr;
}
