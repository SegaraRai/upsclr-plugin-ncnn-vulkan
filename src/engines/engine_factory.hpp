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
    static const std::vector<std::u8string>& get_available_engines();

    // Get engine info by name
    static const SuperResolutionEngineInfo* get_engine_info(const std::u8string& engine_name);

    // Create engine from config
    static std::unique_ptr<SuperResolutionEngine> create_engine(const SuperResolutionEngineConfig& config);
};

#endif  // ENGINE_FACTORY_HPP
