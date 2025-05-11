/**
 * @file upsclr_plugin.cpp
 * @brief Implementation of the plugin DLL API for upsclr-plugin-ncnn-vulkan.
 */

#include "upsclr_plugin.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// glaze
#include "glaze/glaze.hpp"

// ncnn
#include "cpu.h"
#include "gpu.h"

// spdlog
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#if _WIN32
#    include <Windows.h>
#endif

#include "../encoding_utils.hpp"

#include "../engines/base.hpp"
#include "../engines/engine_factory.hpp"

template <>
struct glz::meta<std::u8string> {
    static constexpr auto read_x = [](std::u8string& s, const std::string& input) { s = as_utf8(input); };
    static constexpr auto write_x = [](const std::u8string& s) -> std::string { return as_string(s); };
    static constexpr auto value = glz::custom<read_x, write_x>;
};

struct EngineConfigEx : public SuperResolutionEngineConfig {
    int tile_size = 0;
};

// Define the UpsclrEngineInstance struct
struct UpsclrEngineInstance final {
    std::unique_ptr<SuperResolutionEngine> engine;
    EngineConfigEx engine_config;
};

namespace {
// Global storage for dynamically allocated objects

char8_t* u8dup(const char8_t* str) {
    if (str == nullptr) {
        return nullptr;
    }
    const auto size = std::strlen(reinterpret_cast<const char*>(str)) + 1;
    auto memory = std::malloc(size);
    if (memory == nullptr) {
        return nullptr;
    }
    std::memcpy(memory, str, size);
    return reinterpret_cast<char8_t*>(memory);
}

void free_u8(char8_t* str) {
    std::free(str);
}

enum class PluginState {
    READY,
    NOT_INITIALIZED,
    DESTROYED,
};

struct UpsclrEngineInfoRAII final {
    UpsclrEngineInfo info;

    UpsclrEngineInfoRAII(const char8_t* name, const char8_t* description, const char8_t* version, const char8_t* config_json_schema) {
        info.name = u8dup(name);
        info.description = u8dup(description);
        info.version = u8dup(version);
        info.config_json_schema = u8dup(config_json_schema);
    }

    ~UpsclrEngineInfoRAII() {
        free_u8(const_cast<char8_t*>(info.name));
        free_u8(const_cast<char8_t*>(info.description));
        free_u8(const_cast<char8_t*>(info.version));
        free_u8(const_cast<char8_t*>(info.config_json_schema));
    }

    UpsclrEngineInfoRAII(const UpsclrEngineInfoRAII&) = delete;
    UpsclrEngineInfoRAII& operator=(const UpsclrEngineInfoRAII&) = delete;

    UpsclrEngineInfoRAII(UpsclrEngineInfoRAII&& other) noexcept
        : info(other.info) {
        other.info.name = nullptr;
        other.info.description = nullptr;
        other.info.version = nullptr;
        other.info.config_json_schema = nullptr;
    }

    UpsclrEngineInfoRAII& operator=(UpsclrEngineInfoRAII&& other) noexcept {
        if (this != &other) {
            info = std::move(other.info);
            other.info.name = nullptr;
            other.info.description = nullptr;
            other.info.version = nullptr;
            other.info.config_json_schema = nullptr;
        }
        return *this;
    }
};

// Global mutex
std::shared_mutex g_mutex;
std::mutex g_validation_results_mutex;
std::shared_mutex g_engine_instances_mutex;

// Global plugin state (Use `g_mutex`)
PluginState g_plugin_state = PluginState::NOT_INITIALIZED;

// Storage for validation results (Use `g_validation_results_mutex`)
std::unordered_map<const UpsclrEngineConfigValidationResult*, std::unique_ptr<UpsclrEngineConfigValidationResult, std::function<void(UpsclrEngineConfigValidationResult*)>>> g_validation_result_map;

// Storage for engine instances (Use `g_engine_instances_mutex`)
std::unordered_map<UpsclrEngineInstance*, std::unique_ptr<UpsclrEngineInstance>> g_engine_instance_map;

// Plugin information
const UpsclrPluginInfo g_plugin_info = {
    .name = u8"upsclr-plugin-ncnn-vulkan",
    .version = u8"1.0.0",
    .description = u8"Image upscaling plugin using CNN-based super-resolution engines backed by ncnn and Vulkan",
};

// Cache for engine info structures
std::vector<UpsclrEngineInfoRAII> g_engine_infos;

// Helper function to convert ColorFormat
ColorFormat convert_color_format(UpsclrColorFormat format) {
    switch (format) {
        case UPSCLR_COLOR_FORMAT_RGB:
            return ColorFormat::RGB;

        case UPSCLR_COLOR_FORMAT_BGR:
            return ColorFormat::BGR;

        default:
            return ColorFormat::RGB;
    }
}

}  // namespace

// Base class for engine adapters
class EngineAdapter {
   protected:
    template <typename T>
    static std::optional<T> parse_json_config(const char8_t* config_json, size_t config_json_length, std::vector<std::u8string>& errors) {
        if (config_json == nullptr || config_json_length == 0) {
            errors.push_back(u8"Empty configuration");
            return std::nullopt;
        }

        const std::string_view json_sv(reinterpret_cast<const char*>(config_json), config_json_length);

        T config;
        const auto error_ctx = glz::read<glz::opts{.null_terminated = false, .error_on_unknown_keys = false}>(config, json_sv);
        if (error_ctx) {
            errors.push_back(ascii_to_utf8(glz::format_error(error_ctx, json_sv)));
            return std::nullopt;
        }

        return config;
    }

    void push_common_engine_config_errors(std::vector<std::u8string>& errors, const EngineConfigEx& config) const {
        if (this->engine_info == nullptr) {
            return;
        }

        if (!this->engine_info->supports_model(config.model)) {
            errors.push_back(u8"Unsupported model: " + config.model);
        }

        if (config.tta_mode && !this->engine_info->supports(SuperResolutionFeatureFlags::TTA_MODE)) {
            errors.push_back(u8"TTA mode not supported by this engine");
        }

        if (config.noise >= 0 && !this->engine_info->supports(SuperResolutionFeatureFlags::NOISE)) {
            errors.push_back(u8"Noise reduction not supported by this engine");
        }

        if (config.sync_gap > 0 && !this->engine_info->supports(SuperResolutionFeatureFlags::SYNC_GAP)) {
            errors.push_back(u8"SyncGAP not supported by this engine");
        }

        if (config.gpu_id < 0 && !this->engine_info->supports(SuperResolutionFeatureFlags::CPU)) {
            errors.push_back(u8"CPU processing not supported by this engine");
        }
    }

   public:
    const char8_t* engine_name;
    const SuperResolutionEngineInfo* engine_info;

    EngineAdapter() = delete;
    EngineAdapter(const char8_t* engine_name) : engine_name(engine_name), engine_info(SuperResolutionEngineFactory::get_engine_info(engine_name)) {}

    virtual ~EngineAdapter() = default;

    // Get engine description
    virtual std::u8string get_engine_description() const = 0;

    // Get engine version
    virtual std::u8string get_engine_version() const = 0;

    // Get JSON schema for engine configuration
    virtual std::u8string get_json_schema() const = 0;

    // Parse JSON configuration
    virtual std::optional<EngineConfigEx> parse_config(const char8_t* config_json, size_t config_json_length, std::vector<std::u8string>& errors) const = 0;

    // Validate engine configuration
    virtual void validate_config_extra(const EngineConfigEx& engine_config, std::vector<std::u8string>& errors, std::vector<std::u8string>& warnings) const {}

    // Validate JSON configuration
    virtual void validate_config(const char8_t* config_json, size_t config_json_length, std::vector<std::u8string>& errors, std::vector<std::u8string>& warnings) const {
        const auto parsed = parse_config(config_json, config_json_length, errors);
        if (!parsed.has_value()) {
            return;
        }

        const auto& engine_config = parsed.value();

        this->push_common_engine_config_errors(errors, engine_config);
        this->validate_config_extra(engine_config, errors, warnings);
    }

    // Create engine instance
    virtual std::unique_ptr<SuperResolutionEngine> create_engine(const EngineConfigEx& engine_config, std::vector<std::u8string>& errors) const {
        try {
            return SuperResolutionEngineFactory::create_engine(engine_config);
        } catch (const std::exception& e) {
            errors.push_back(ascii_to_utf8(std::format("Failed to create engine: {}", e.what())));
            return nullptr;
        }
    }
};

// RealESRGAN engine adapter
class RealESRGANAdapter final : public EngineAdapter {
   public:
    struct Config {
        std::u8string model_dir = u8"";
        int gpu_id = ncnn::get_default_gpu_index();
        int num_threads = ncnn::get_cpu_count();
        int tile_size = 0;
        bool tta_mode = false;
        std::u8string model = u8"realesrgan-x4plus";

        struct glaze_json_schema {
            glz::schema model_dir{
                .description = "Path to the model directory",
            };
            glz::schema gpu_id{
                .description = "GPU device ID",
                .minimum = 0,
            };
            glz::schema num_threads{
                .description = "Number of CPU threads when using CPU processing (No effect for now)",
                .minimum = 1,
            };
            glz::schema tile_size{
                .description = "Tile size for processing (0 = auto)",
                .minimum = 0,
            };
            glz::schema tta_mode{
                .description = "Enable TTA mode for better quality but slower processing",
            };
            glz::schema model{
                .description = "Model name",
                .enumeration = std::vector<std::string_view>({
                    "realesr-animevideov3",
                    "realesrnet-x4plus",
                    "realesrgan-x4plus",
                    "realesrgan-x4plus-anime",
                }),
            };
        };
    };

    RealESRGANAdapter() : EngineAdapter(u8"realesrgan") {}

    std::u8string get_engine_description() const override {
        return this->engine_info ? this->engine_info->description : u8"Real-ESRGAN super-resolution engine";
    }

    std::u8string get_engine_version() const override {
        return this->engine_info ? this->engine_info->version : u8"1.0.0";
    }

    std::u8string get_json_schema() const override {
        return as_utf8(glz::write_json_schema<Config>().value_or(""));
    }

    std::optional<EngineConfigEx> parse_config(const char8_t* config_json, size_t config_json_length, std::vector<std::u8string>& errors) const override {
        const auto parsed_config = EngineAdapter::parse_json_config<Config>(config_json, config_json_length, errors);
        if (!parsed_config.has_value()) {
            return std::nullopt;
        }

        const auto& config = parsed_config.value();
        return EngineConfigEx{
            {
                .model_dir = std::filesystem::path(config.model_dir),
                .model = config.model,
                .gpu_id = config.gpu_id,
                .tta_mode = config.tta_mode,
                .num_threads = config.num_threads,
                .engine_name = this->engine_name,
            },
            config.tile_size,
        };
    }
};

// RealCUGAN engine adapter
class RealCUGANAdapter final : public EngineAdapter {
   public:
    struct Config {
        std::u8string model_dir = u8"";
        int gpu_id = ncnn::get_default_gpu_index();
        int num_threads = ncnn::get_cpu_count();
        int tile_size = 0;
        bool tta_mode = false;
        std::u8string model = u8"models-se";
        int noise = -1;
        int sync_gap = 0;

        struct glaze_json_schema {
            glz::schema model_dir{
                .description = "Path to the model directory",
            };
            glz::schema gpu_id{
                .description = "GPU device ID",
                .minimum = 0,
            };
            glz::schema num_threads{
                .description = "Number of CPU threads when using CPU processing (No effect for now)",
                .minimum = 1,
            };
            glz::schema tile_size{
                .description = "Tile size for processing (0 = auto)",
                .minimum = 0,
            };
            glz::schema tta_mode{
                .description = "Enable TTA mode for better quality but slower processing",
            };
            glz::schema model{
                .description = "Model name",
                .enumeration = std::vector<std::string_view>({
                    "models-pro",
                    "models-se",
                    "models-nose",
                }),
            };
            glz::schema noise{
                .description = "Noise reduction level (-1 = conservative, 0 = no denoise, 1 = denoise 1x, 2 = denoise 2x, 3 = denoise 3x)",
                .minimum = -1,
                .maximum = 3,
            };
            glz::schema sync_gap{
                .description = "SyncGAP level (0 = no sync_gap [fastest], 1 = sync_gap [slowest], 2 = rough sync_gap [slower], 3 = very rough sync_gap [medium])",
                .minimum = 0,
                .maximum = 3,
            };
        };
    };

    RealCUGANAdapter() : EngineAdapter(u8"realcugan") {}

    std::u8string get_engine_description() const override {
        return this->engine_info ? this->engine_info->description : u8"Real-CUGAN super-resolution engine";
    }

    std::u8string get_engine_version() const override {
        return this->engine_info ? this->engine_info->version : u8"1.0.0";
    }

    std::u8string get_json_schema() const override {
        return as_utf8(glz::write_json_schema<Config>().value_or(""));
    }

    std::optional<EngineConfigEx> parse_config(const char8_t* config_json, size_t config_json_length, std::vector<std::u8string>& errors) const override {
        const auto parsed_config = EngineAdapter::parse_json_config<Config>(config_json, config_json_length, errors);
        if (!parsed_config.has_value()) {
            return std::nullopt;
        }

        const auto& config = parsed_config.value();
        return EngineConfigEx{
            {
                .model_dir = std::filesystem::path(config.model_dir),
                .model = config.model,
                .gpu_id = config.gpu_id,
                .tta_mode = config.tta_mode,
                .num_threads = config.num_threads,
                .noise = config.noise,
                .sync_gap = config.sync_gap,
                .engine_name = this->engine_name,
            },
            config.tile_size,
        };
    }

    void validate_config_extra(const EngineConfigEx& engine_config, std::vector<std::u8string>& errors, std::vector<std::u8string>& warnings) const override {
        if (engine_config.noise < -1 || engine_config.noise > 3) {
            errors.push_back(u8"Invalid noise level (valid values: -1, 0, 1, 2, 3)");
        }

        if (engine_config.sync_gap < 0 || engine_config.sync_gap > 3) {
            errors.push_back(u8"Invalid sync_gap level (valid values: 0, 1, 2, 3)");
        }
    }
};

// Engine adapter registry
class EngineAdapterRegistry final {
    std::unordered_map<std::u8string, std::unique_ptr<EngineAdapter>> adapters;
    std::vector<std::u8string> adapter_names;

    EngineAdapterRegistry() {
        // Register all adapters
        register_adapter(std::make_unique<RealESRGANAdapter>());
        register_adapter(std::make_unique<RealCUGANAdapter>());

        // Add more adapters here as needed
    }

   public:
    static EngineAdapterRegistry& instance() {
        static EngineAdapterRegistry registry;
        return registry;
    }

    void register_adapter(std::unique_ptr<EngineAdapter>&& adapter) {
        if (adapter == nullptr) {
            return;
        }

        const auto name = adapter->engine_name;
        this->adapters[name] = std::move(adapter);

        this->adapter_names.push_back(name);
    }

    const EngineAdapter* get_adapter(const std::u8string& name) const {
        const auto it = this->adapters.find(name);
        return it != this->adapters.end() ? it->second.get() : nullptr;
    }

    const EngineAdapter* get_adapter(size_t index) const {
        if (index >= this->adapter_names.size()) {
            return nullptr;
        }

        const std::u8string& name = this->adapter_names[index];
        return this->get_adapter(name);
    }

    const std::vector<std::u8string>& get_adapter_names() const {
        return this->adapter_names;
    }

    size_t get_adapter_count() const {
        return this->adapter_names.size();
    }
};

namespace {

}  // namespace

// Plugin API implementation
extern "C" {

UPSCLR_API UpsclrErrorCode upsclr_plugin_initialize() {
    std::lock_guard lk(g_mutex);

    if (g_plugin_state != PluginState::NOT_INITIALIZED && g_plugin_state != PluginState::DESTROYED) {
        return UpsclrErrorCode::UPSCLR_ERROR_PLUGIN_ALREADY_INITIALIZED;
    }

    std::lock_guard validation_storage_lk(g_validation_results_mutex);
    std::lock_guard instance_storage_lk(g_engine_instances_mutex);

    {
        auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        auto stderr_logger = std::make_shared<spdlog::logger>("upsclr-plugin-ncnn-vulkan", std::move(stderr_sink));
        spdlog::register_logger(std::move(stderr_logger));
    }

    ncnn::create_gpu_instance();

    const auto& registry = EngineAdapterRegistry::instance();
    for (const auto& name : registry.get_adapter_names()) {
        const auto adapter = registry.get_adapter(name);
        if (adapter == nullptr) {
            continue;
        }

        g_engine_infos.push_back(UpsclrEngineInfoRAII(
            adapter->engine_name,
            adapter->get_engine_description().c_str(),
            adapter->get_engine_version().c_str(),
            adapter->get_json_schema().c_str()));
    }

    g_plugin_state = PluginState::READY;

    return UpsclrErrorCode::UPSCLR_SUCCESS;
}

UPSCLR_API UpsclrErrorCode upsclr_plugin_shutdown() {
    std::lock_guard lk(g_mutex);

    if (g_plugin_state != PluginState::READY) {
        return g_plugin_state == PluginState::NOT_INITIALIZED ? UpsclrErrorCode::UPSCLR_ERROR_PLUGIN_NOT_INITIALIZED : UpsclrErrorCode::UPSCLR_ERROR_PLUGIN_ALREADY_DESTROYED;
    }

    std::lock_guard validation_storage_lk(g_validation_results_mutex);
    std::lock_guard instance_storage_lk(g_engine_instances_mutex);

    g_validation_result_map.clear();
    g_engine_instance_map.clear();
    g_engine_infos.clear();

    ncnn::destroy_gpu_instance();

    g_plugin_state = PluginState::DESTROYED;

    return UpsclrErrorCode::UPSCLR_SUCCESS;
}

UPSCLR_API const UpsclrPluginInfo* upsclr_plugin_get_info() {
    return &g_plugin_info;
}

UPSCLR_API size_t upsclr_plugin_count_engines() {
    return g_engine_infos.size();
}

UPSCLR_API const UpsclrEngineInfo* upsclr_plugin_get_engine_info(size_t engine_index) {
    if (engine_index >= g_engine_infos.size()) {
        return nullptr;
    }
    return &g_engine_infos[engine_index].info;
}

UPSCLR_API const UpsclrEngineConfigValidationResult* upsclr_plugin_validate_engine_config(
    size_t engine_index,
    const char8_t* config_json,
    size_t config_json_length) {
    std::shared_lock lk(g_mutex);

    if (g_plugin_state != PluginState::READY) {
        return nullptr;
    }

    const auto& registry = EngineAdapterRegistry::instance();
    const auto adapter = registry.get_adapter(engine_index);
    if (adapter == nullptr) {
        return nullptr;
    }

    std::vector<std::u8string> errors;
    std::vector<std::u8string> warnings;

    adapter->validate_config(config_json, config_json_length, errors, warnings);

    // Create validation result
    auto result = std::unique_ptr<UpsclrEngineConfigValidationResult, std::function<void(UpsclrEngineConfigValidationResult*)>>(
        new UpsclrEngineConfigValidationResult(),
        [](UpsclrEngineConfigValidationResult* ptr) {
            if (ptr == nullptr) {
                return;
            }

            if (ptr->error_messages) {
                for (size_t i = 0; i < ptr->error_count; ++i) {
                    free_u8(const_cast<char8_t*>(ptr->error_messages[i]));
                }
                delete[] ptr->error_messages;
            }

            if (ptr->warning_messages) {
                for (size_t i = 0; i < ptr->warning_count; ++i) {
                    free_u8(const_cast<char8_t*>(ptr->warning_messages[i]));
                }
                delete[] ptr->warning_messages;
            }

            delete ptr;
        });

    result->is_valid = errors.empty();
    result->error_count = errors.size();
    result->warning_count = warnings.size();

    // Allocate and copy error messages
    if (!errors.empty()) {
        const char8_t** error_messages = new const char8_t*[errors.size()];
        for (size_t i = 0; i < errors.size(); ++i) {
            error_messages[i] = u8dup(errors[i].c_str());
        }
        result->error_messages = error_messages;
    } else {
        result->error_messages = nullptr;
    }

    // Allocate and copy warning messages
    if (!warnings.empty()) {
        const char8_t** warning_messages = new const char8_t*[warnings.size()];
        for (size_t i = 0; i < warnings.size(); ++i) {
            warning_messages[i] = u8dup(warnings[i].c_str());
        }
        result->warning_messages = warning_messages;
    } else {
        result->warning_messages = nullptr;
    }

    // Store in global storage
    std::lock_guard storage_lk(g_validation_results_mutex);

    const auto* ptr_result = result.get();
    g_validation_result_map.emplace(ptr_result, std::move(result));

    return ptr_result;
}

UPSCLR_API void upsclr_plugin_free_validation_result(const UpsclrEngineConfigValidationResult* result) {
    std::shared_lock lk(g_mutex);

    if (g_plugin_state != PluginState::READY) {
        return;
    }

    std::lock_guard storage_lk(g_validation_results_mutex);

    g_validation_result_map.erase(result);
}

UPSCLR_API UpsclrEngineInstance* upsclr_plugin_create_engine_instance(
    size_t engine_index,
    const char8_t* config_json,
    size_t config_json_length) {
    std::shared_lock lk(g_mutex);

    if (g_plugin_state != PluginState::READY) {
        return nullptr;
    }

    const auto logger_error = spdlog::get("upsclr-plugin-ncnn-vulkan");

    const auto& registry = EngineAdapterRegistry::instance();
    const auto adapter = registry.get_adapter(engine_index);
    if (adapter == nullptr) {
        logger_error->error("[{}] adapter is nullptr", __func__);
        return nullptr;
    }

    std::vector<std::u8string> errors;
    auto opt_engine_config = adapter->parse_config(config_json, config_json_length, errors);
    if (!opt_engine_config.has_value()) {
        logger_error->error("[{}] parse_config failed", __func__);
        for (const auto& error : errors) {
            logger_error->error("{}", as_string(error));
        }
        return nullptr;
    }

    auto& engine_config = opt_engine_config.value();
    engine_config.logger_error = logger_error;

    auto engine = adapter->create_engine(engine_config, errors);
    if (engine == nullptr) {
        logger_error->error("[{}] create_engine failed", __func__);
        for (const auto& error : errors) {
            logger_error->error("{}", as_string(error));
        }
        return nullptr;
    }

    auto instance = std::make_unique<UpsclrEngineInstance>();
    instance->engine = std::move(engine);
    instance->engine_config = engine_config;

    std::lock_guard storage_lk(g_engine_instances_mutex);

    auto* ptr_instance = instance.get();
    g_engine_instance_map.emplace(ptr_instance, std::move(instance));

    return ptr_instance;
}

UPSCLR_API void upsclr_plugin_destroy_engine_instance(UpsclrEngineInstance* instance) {
    std::shared_lock lk(g_mutex);

    if (g_plugin_state != PluginState::READY) {
        return;
    }

    std::lock_guard storage_lk(g_engine_instances_mutex);

    g_engine_instance_map.erase(instance);
}

UPSCLR_API UpsclrErrorCode upsclr_plugin_preload_upscale(UpsclrEngineInstance* instance, int32_t scale) {
    std::shared_lock lk(g_mutex);

    if (g_plugin_state != PluginState::READY) {
        return g_plugin_state == PluginState::NOT_INITIALIZED ? UpsclrErrorCode::UPSCLR_ERROR_PLUGIN_NOT_INITIALIZED : UpsclrErrorCode::UPSCLR_ERROR_PLUGIN_ALREADY_DESTROYED;
    }

    std::shared_lock storage_lk(g_engine_instances_mutex);

    if (instance == nullptr || instance->engine == nullptr || scale < 1) {
        return UPSCLR_ERROR_INVALID_ARGUMENT;
    }

    try {
        const auto result = instance->engine->preload(scale);
        if (result != 0) {
            return UPSCLR_ERROR_PRELOAD_FAILED;
        }

        return UPSCLR_SUCCESS;
    } catch (...) {
        // Log error if needed
        return UPSCLR_ERROR_PRELOAD_FAILED;
    }
}

UPSCLR_API UpsclrErrorCode upsclr_plugin_upscale(
    UpsclrEngineInstance* instance,
    int32_t scale,
    const unsigned char* in_data,
    size_t in_size,
    uint32_t in_width,
    uint32_t in_height,
    uint32_t in_channels,
    UpsclrColorFormat in_color_format,
    unsigned char* out_data,
    size_t out_size,
    UpsclrColorFormat out_color_format) {
    std::shared_lock lk(g_mutex);

    if (g_plugin_state != PluginState::READY) {
        return g_plugin_state == PluginState::NOT_INITIALIZED ? UpsclrErrorCode::UPSCLR_ERROR_PLUGIN_NOT_INITIALIZED : UpsclrErrorCode::UPSCLR_ERROR_PLUGIN_ALREADY_DESTROYED;
    }

    std::shared_lock storage_lk(g_engine_instances_mutex);

    // Validate arguments
    if (instance == nullptr || instance->engine == nullptr || in_data == nullptr || out_data == nullptr || scale < 1) {
        return UPSCLR_ERROR_INVALID_ARGUMENT;
    }

    // Validate buffer sizes
    if (in_size != in_width * in_height * in_channels) {
        return UPSCLR_ERROR_INVALID_ARGUMENT;
    }

    if (out_size != in_width * in_height * in_channels * scale * scale) {
        return UPSCLR_ERROR_INVALID_ARGUMENT;
    }

    try {
        ncnn::Mat in(static_cast<int>(in_width), static_cast<int>(in_height), const_cast<unsigned char*>(in_data), static_cast<size_t>(in_channels), static_cast<int>(in_channels));
        ncnn::Mat out(static_cast<int>(in_width * scale), static_cast<int>(in_height * scale), out_data, static_cast<size_t>(in_channels), static_cast<int>(in_channels));

        const auto tile_size = instance->engine_config.tile_size > 0 ? instance->engine_config.tile_size : instance->engine->get_default_tile_size();

        // Set process config
        ProcessConfig process_config{
            .scale = scale,
            .input_format = convert_color_format(in_color_format),
            .output_format = convert_color_format(out_color_format),
            .tile_size = tile_size,
        };

        // Process image
        const int result = instance->engine->process(in, out, process_config);
        if (result != 0) {
            return UPSCLR_ERROR_UPSCALE_FAILED;
        }

        return UPSCLR_SUCCESS;
    } catch (...) {
        // Log error if needed
        return UPSCLR_ERROR_UPSCALE_FAILED;
    }
}

}  // extern "C"
