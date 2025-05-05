// Super-resolution engine factory implementation
#include "engine_factory.hpp"

#include <algorithm>
#include <stdexcept>

// Include all engine headers
#include "realcugan/realcugan.hpp"
#include "realesrgan/realesrgan.hpp"
// #include "waifu2x/waifu2x.hpp"

// Helper function to register all engines
static std::vector<std::string> get_all_engine_names() {
    return {
        "realesrgan",
        "realcugan",
        // "waifu2x"
    };
}

std::vector<std::string> SuperResolutionEngineFactory::get_available_engines() {
    return get_all_engine_names();
}

const SuperResolutionEngineInfo* SuperResolutionEngineFactory::get_engine_info(const std::string& engine_name) {
    if (engine_name == "realesrgan") {
        return &RealESRGAN::get_engine_info();
    }
    if (engine_name == "realcugan") {
        return &RealCUGAN::get_engine_info();
    }
    // if (engine_name == "waifu2x") {
    //     return &Waifu2x::get_engine_info();
    // }

    return nullptr;
}

std::unique_ptr<SuperResolutionEngine> SuperResolutionEngineFactory::create_engine(
    const std::string& engine_name,
    const SuperResolutionEngineConfig& config) {
    SuperResolutionEngineConfig cfg = config;
    cfg.engine_name = engine_name;

    if (engine_name == "realesrgan") {
        return std::make_unique<RealESRGAN>(cfg);
    }
    if (engine_name == "realcugan") {
        return std::make_unique<RealCUGAN>(cfg);
    }
    // if (engine_name == "waifu2x") {
    //     return std::make_unique<Waifu2x>(cfg);
    // }

    throw std::runtime_error("Unknown engine: " + engine_name);
}

std::unique_ptr<SuperResolutionEngine> SuperResolutionEngineFactory::create_engine(
    const SuperResolutionEngineConfig& config) {
    if (!config.engine_name.empty()) {
        return create_engine(config.engine_name, config);
    }

    // Auto-detect engine
    const std::string engine_name = select_engine_for_config(config);
    if (engine_name.empty()) {
        throw std::runtime_error("Could not determine appropriate engine for the given configuration");
    }

    return create_engine(engine_name, config);
}

std::string SuperResolutionEngineFactory::select_engine_for_config(const SuperResolutionEngineConfig& config) {
    // Try to determine from model name
    if (!config.model.empty()) {
        for (const auto& engine_name : get_all_engine_names()) {
            const auto* info = get_engine_info(engine_name);
            if (info && info->supports_model(config.model)) {
                return engine_name;
            }
        }
    }

    // Try to determine from specific features
    if (config.syncgap > 0) {
        // SyncGAP is RealCUGAN-specific
        return "realcugan";
    }

    // Default to RealESRGAN if available
    if (std::find(get_all_engine_names().begin(), get_all_engine_names().end(), "realesrgan") != get_all_engine_names().end()) {
        return "realesrgan";
    }

    // Return first available engine
    const auto engines = get_all_engine_names();
    return engines.empty() ? "" : engines[0];
}
