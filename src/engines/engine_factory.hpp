// Super-resolution engine factory
#ifndef ENGINE_FACTORY_HPP
#define ENGINE_FACTORY_HPP

#include <memory>
#include <string>
#include <vector>

#include "base.hpp"

// Factory for creating super-resolution engines
class SuperResolutionEngineFactory {
   public:
    // Get available engine names
    static std::vector<std::string> get_available_engines();

    // Get engine info by name
    static const SuperResolutionEngineInfo* get_engine_info(const std::string& engine_name);

    // Create engine by name
    static std::unique_ptr<SuperResolutionEngine> create_engine(const std::string& engine_name, const SuperResolutionEngineConfig& config);

    // Create engine from config (uses engine_name from config if provided, otherwise tries to auto-detect)
    static std::unique_ptr<SuperResolutionEngine> create_engine(const SuperResolutionEngineConfig& config);

    // Auto-detect appropriate engine for config
    static std::string select_engine_for_config(const SuperResolutionEngineConfig& config);
};

#endif  // ENGINE_FACTORY_HPP
