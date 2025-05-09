/**
 * @file upsclr_plugin.cpp
 * @brief Implementation of the plugin DLL API for upsclr-ncnn-vulkan-plugin.
 */

#include "upsclr_plugin.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Include glaze for JSON parsing
#include <glaze/glaze.hpp>

// ncnn
#include "cpu.h"
#include "gpu.h"

#if _WIN32
#    include <Windows.h>
#endif

// Include engine headers
#include "../engines/base.hpp"
#include "../engines/engine_factory.hpp"

// Define the UpsclrEngineInstance struct
struct UpsclrEngineInstance final {
    std::mutex mutex;
    std::unique_ptr<SuperResolutionEngine> engine;
    ProcessConfig process_config;
};

namespace {
// Global storage for dynamically allocated objects

struct UpsclrEngineInfoRAII final {
    UpsclrEngineInfo info;

    UpsclrEngineInfoRAII(const char* name, const char* description, const char* version, const char* config_json_schema) {
        info.name = strdup(name);
        info.description = strdup(description);
        info.version = strdup(version);
        info.config_json_schema = strdup(config_json_schema);
    }

    ~UpsclrEngineInfoRAII() {
        free(const_cast<char*>(info.name));
        free(const_cast<char*>(info.description));
        free(const_cast<char*>(info.version));
        free(const_cast<char*>(info.config_json_schema));
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
std::mutex g_mutex;

// Storage for validation results
std::unordered_map<const UpsclrEngineConfigValidationResult*, std::unique_ptr<UpsclrEngineConfigValidationResult, std::function<void(UpsclrEngineConfigValidationResult*)>>> g_validation_result_map;

// Storage for engine instances
std::unordered_map<UpsclrEngineInstance*, std::unique_ptr<UpsclrEngineInstance>> g_engine_instance_map;

// Plugin information
const UpsclrPluginInfo g_plugin_info = {
    .name = "upsclr-ncnn-vulkan-plugin",
    .version = "1.0.0",
    .description = "Image upscaling plugin using CNN-based super-resolution engines backed by ncnn and Vulkan",
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
    static void push_glaze_errors(std::vector<std::string>& errors, const glz::error_ctx& error_ctx) {
        if (!error_ctx.includer_error.empty()) {
            errors.push_back(std::string(error_ctx.includer_error));
        }
        if (!error_ctx.custom_error_message.empty()) {
            errors.push_back(std::string(error_ctx.custom_error_message));
        }
    }

    void push_common_engine_config_errors(std::vector<std::string>& errors, const SuperResolutionEngineConfig& config) const {
        if (this->engine_info == nullptr) {
            return;
        }

        if (!this->engine_info->supports_model(config.model)) {
            errors.push_back("Unsupported model: " + config.model);
        }

        if (config.tta_mode && !this->engine_info->supports(SuperResolutionFeatureFlags::TTA_MODE)) {
            errors.push_back("TTA mode not supported by this engine");
        }

        if (config.noise >= 0 && !this->engine_info->supports(SuperResolutionFeatureFlags::NOISE)) {
            errors.push_back("Noise reduction not supported by this engine");
        }

        if (config.syncgap > 0 && !this->engine_info->supports(SuperResolutionFeatureFlags::SYNCGAP)) {
            errors.push_back("SyncGAP not supported by this engine");
        }

        if (config.gpuid < 0 && !this->engine_info->supports(SuperResolutionFeatureFlags::CPU)) {
            errors.push_back("CPU processing not supported by this engine");
        }
    }

   public:
    const char* engine_name;
    const SuperResolutionEngineInfo* engine_info;

    EngineAdapter() = delete;
    EngineAdapter(const char* engine_name) : engine_name(engine_name), engine_info(SuperResolutionEngineFactory::get_engine_info(engine_name)) {}

    virtual ~EngineAdapter() = default;

    // Get engine description
    virtual std::string get_engine_description() const = 0;

    // Get engine version
    virtual std::string get_engine_version() const = 0;

    // Get JSON schema for engine configuration
    virtual std::string get_json_schema() const = 0;

    // Parse JSON configuration
    virtual std::optional<SuperResolutionEngineConfig> parse_config(const char* config_json, size_t config_json_length, std::vector<std::string>& errors) const = 0;

    // Validate engine configuration
    virtual void validate_config_extra(const SuperResolutionEngineConfig& engine_config, std::vector<std::string>& errors, std::vector<std::string>& warnings) const {}

    // Validate JSON configuration
    virtual void validate_config(const char* config_json, size_t config_json_length, std::vector<std::string>& errors, std::vector<std::string>& warnings) const {
        const auto parsed = parse_config(config_json, config_json_length, errors);
        if (!parsed.has_value()) {
            return;
        }

        const auto& engine_config = parsed.value();

        this->push_common_engine_config_errors(errors, engine_config);
        this->validate_config_extra(engine_config, errors, warnings);
    }

    // Create engine instance
    virtual std::unique_ptr<SuperResolutionEngine> create_engine(const char* config_json, size_t config_json_length, std::vector<std::string>& errors) const {
        const auto engine_config = parse_config(config_json, config_json_length, errors);
        if (!engine_config.has_value()) {
            return nullptr;
        }

        try {
            return SuperResolutionEngineFactory::create_engine(this->engine_name, engine_config.value());
        } catch (const std::exception& e) {
            errors.push_back(std::format("Failed to create engine: {}", e.what()));
            return nullptr;
        }
    }
};

// RealESRGAN engine adapter
class RealESRGANAdapter final : public EngineAdapter {
   public:
    struct Config {
        int gpuid = ncnn::get_default_gpu_index();
        int num_threads = ncnn::get_cpu_count();
        bool tta_mode = false;
        std::string model = "realesrgan-x4plus";

        struct glaze_json_schema {
            glz::schema gpuid{
                .description = "GPU device ID",
                .minimum = 0,
            };
            glz::schema num_threads{
                .description = "Number of CPU threads when using CPU processing (No effect for now)",
                .minimum = 1,
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

    RealESRGANAdapter() : EngineAdapter("realesrgan") {}

    std::string get_engine_description() const override {
        return this->engine_info ? this->engine_info->description : "Real-ESRGAN super-resolution engine";
    }

    std::string get_engine_version() const override {
        return this->engine_info ? this->engine_info->version : "1.0.0";
    }

    std::string get_json_schema() const override {
        return glz::write_json_schema<Config>().value_or("");
    }

    std::optional<SuperResolutionEngineConfig> parse_config(const char* config_json, size_t config_json_length, std::vector<std::string>& errors) const override {
        if (!config_json || config_json_length == 0) {
            errors.push_back("Empty configuration");
            return std::nullopt;
        }

        std::string json_str(config_json, config_json_length);
        auto result = glz::read_json<Config>(json_str);

        if (!result.has_value()) {
            EngineAdapter::push_glaze_errors(errors, result.error());
            return std::nullopt;
        }

        const auto& config = result.value();

        SuperResolutionEngineConfig engine_config;
        engine_config.engine_name = this->engine_name;
        engine_config.gpuid = config.gpuid;
        engine_config.tta_mode = config.tta_mode;
        engine_config.num_threads = config.num_threads;
        engine_config.model = config.model;

        return engine_config;
    }
};

// RealCUGAN engine adapter
class RealCUGANAdapter final : public EngineAdapter {
   public:
    struct Config {
        int gpuid = ncnn::get_default_gpu_index();
        int num_threads = ncnn::get_cpu_count();
        bool tta_mode = false;
        std::string model = "models-se";
        int noise = -1;
        int syncgap = 0;

        struct glaze_json_schema {
            glz::schema gpuid{
                .description = "GPU device ID",
                .minimum = 0,
            };
            glz::schema num_threads{
                .description = "Number of CPU threads when using CPU processing (No effect for now)",
                .minimum = 1,
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
            glz::schema syncgap{
                .description = "SyncGAP level (0 = no syncgap [fastest], 1 = syncgap [slowest], 2 = rough syncgap [slower], 3 = very rough syncgap [medium])",
                .minimum = 0,
                .maximum = 3,
            };
        };
    };

    RealCUGANAdapter() : EngineAdapter("realcugan") {}

    std::string get_engine_description() const override {
        return engine_info ? engine_info->description : "Real-CUGAN super-resolution engine";
    }

    std::string get_engine_version() const override {
        return engine_info ? engine_info->version : "1.0.0";
    }

    std::string get_json_schema() const override {
        return glz::write_json_schema<Config>().value_or("");
    }

    std::optional<SuperResolutionEngineConfig> parse_config(const char* config_json, size_t config_json_length, std::vector<std::string>& errors) const override {
        if (config_json == nullptr || config_json_length == 0) {
            errors.push_back("Empty configuration");
            return std::nullopt;
        }

        std::string json_str(config_json, config_json_length);
        auto result = glz::read_json<Config>(json_str);

        if (!result.has_value()) {
            push_glaze_errors(errors, result.error());
            return std::nullopt;
        }

        const auto& config = result.value();

        SuperResolutionEngineConfig engine_config;
        engine_config.engine_name = this->engine_name;
        engine_config.gpuid = config.gpuid;
        engine_config.tta_mode = config.tta_mode;
        engine_config.num_threads = config.num_threads;
        engine_config.model = config.model;
        engine_config.noise = config.noise;
        engine_config.syncgap = config.syncgap;

        return engine_config;
    }

    void validate_config_extra(const SuperResolutionEngineConfig& engine_config, std::vector<std::string>& errors, std::vector<std::string>& warnings) const override {
        if (engine_config.noise < -1 || engine_config.noise > 3) {
            errors.push_back("Invalid noise level (valid values: -1, 0, 1, 2, 3)");
        }

        if (engine_config.syncgap < 0 || engine_config.syncgap > 3) {
            errors.push_back("Invalid syncgap level (valid values: 0, 1, 2, 3)");
        }
    }
};

// Engine adapter registry
class EngineAdapterRegistry final {
    std::unordered_map<std::string, std::unique_ptr<EngineAdapter>> adapters;
    std::vector<std::string> adapter_names;

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
        if (!adapter) {
            return;
        }

        const auto name = adapter->engine_name;
        this->adapters[name] = std::move(adapter);

        this->adapter_names.push_back(name);
    }

    const EngineAdapter* get_adapter(const std::string& name) const {
        const auto it = this->adapters.find(name);
        return it != this->adapters.end() ? it->second.get() : nullptr;
    }

    const EngineAdapter* get_adapter(size_t index) const {
        if (index >= this->adapter_names.size()) {
            return nullptr;
        }

        const std::string& name = this->adapter_names[index];
        return this->get_adapter(name);
    }

    const std::vector<std::string>& get_adapter_names() const {
        return this->adapter_names;
    }

    size_t get_adapter_count() const {
        return this->adapter_names.size();
    }
};

namespace {

void initialize_all() {
    std::lock_guard lk(g_mutex);

    ncnn::create_gpu_instance();

    const auto& registry = EngineAdapterRegistry::instance();
    for (const auto& name : registry.get_adapter_names()) {
        const auto* adapter = registry.get_adapter(name);
        if (adapter == nullptr) {
            continue;
        }

        g_engine_infos.push_back(UpsclrEngineInfoRAII(
            adapter->engine_name,
            adapter->get_engine_description().c_str(),
            adapter->get_engine_version().c_str(),
            adapter->get_json_schema().c_str()));
    }
}

void cleanup_all() {
    std::lock_guard lk(g_mutex);

    g_validation_result_map.clear();
    g_engine_instance_map.clear();
    g_engine_infos.clear();

    ncnn::destroy_gpu_instance();
}

}  // namespace

// Plugin API implementation
extern "C" {

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

UPSCLR_API const UpsclrEngineConfigValidationResult* upsclr_validate_engine_config(
    size_t engine_index,
    const char* config_json,
    size_t config_json_length) {
    std::lock_guard lk(g_mutex);

    const auto& registry = EngineAdapterRegistry::instance();
    const auto* adapter = registry.get_adapter(engine_index);
    if (!adapter) {
        return nullptr;
    }

    std::vector<std::string> errors;
    std::vector<std::string> warnings;

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
                    free((void*)ptr->error_messages[i]);
                }
                delete[] ptr->error_messages;
            }

            if (ptr->warning_messages) {
                for (size_t i = 0; i < ptr->warning_count; ++i) {
                    free((void*)ptr->warning_messages[i]);
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
        const char** error_messages = new const char*[errors.size()];
        for (size_t i = 0; i < errors.size(); ++i) {
            error_messages[i] = strdup(errors[i].c_str());
        }
        result->error_messages = error_messages;
    } else {
        result->error_messages = nullptr;
    }

    // Allocate and copy warning messages
    if (!warnings.empty()) {
        const char** warning_messages = new const char*[warnings.size()];
        for (size_t i = 0; i < warnings.size(); ++i) {
            warning_messages[i] = strdup(warnings[i].c_str());
        }
        result->warning_messages = warning_messages;
    } else {
        result->warning_messages = nullptr;
    }

    // Store in global storage
    const auto* ptr_result = result.get();
    g_validation_result_map.emplace(ptr_result, std::move(result));

    return ptr_result;
}

UPSCLR_API void upsclr_free_validation_result(const UpsclrEngineConfigValidationResult* result) {
    std::lock_guard lk(g_mutex);

    g_validation_result_map.erase(result);
}

UPSCLR_API UpsclrEngineInstance* upsclr_plugin_create_engine_instance(
    size_t engine_index,
    const char* config_json,
    size_t config_json_length) {
    std::lock_guard lk(g_mutex);

    const auto& registry = EngineAdapterRegistry::instance();
    const auto* adapter = registry.get_adapter(engine_index);

    if (adapter == nullptr) {
        return nullptr;
    }

    std::vector<std::string> errors;
    auto engine = adapter->create_engine(config_json, config_json_length, errors);
    if (engine == nullptr) {
        return nullptr;
    }

    auto instance = std::make_unique<UpsclrEngineInstance>();
    instance->engine = std::move(engine);
    instance->process_config = instance->engine->create_default_process_config();

    const auto ptr_instance = instance.get();
    g_engine_instance_map.emplace(ptr_instance, std::move(instance));

    return ptr_instance;
}

UPSCLR_API void upsclr_plugin_destroy_engine_instance(UpsclrEngineInstance* instance) {
    std::lock_guard lk(g_mutex);

    g_engine_instance_map.erase(instance);
}

UPSCLR_API UpsclrErrorCode upsclr_preload_upscale(UpsclrEngineInstance* instance, int32_t scale) {
    std::lock_guard lk(g_mutex);

    if (instance == nullptr || instance->engine == nullptr || scale < 1) {
        return UPSCLR_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard instance_lk(instance->mutex);

    try {
        const auto result = instance->engine->preload(scale);
        if (result != 0) {
            return UPSCLR_ERROR_UPSCALE_FAILED;
        }

        return UPSCLR_SUCCESS;
    } catch (...) {
        // Log error if needed
        return UPSCLR_ERROR_UPSCALE_FAILED;
    }
}

UPSCLR_API UpsclrErrorCode upsclr_upscale(
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
    std::lock_guard lk(g_mutex);

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

    std::lock_guard instance_lk(instance->mutex);

    try {
        ncnn::Mat in(static_cast<int>(in_width), static_cast<int>(in_height), const_cast<unsigned char*>(in_data), static_cast<size_t>(in_channels), static_cast<int>(in_channels));
        ncnn::Mat out(static_cast<int>(in_width * scale), static_cast<int>(in_height * scale), out_data, static_cast<size_t>(in_channels), static_cast<int>(in_channels));

        // Set process config
        ProcessConfig process_config = instance->process_config;
        process_config.scale = scale;
        process_config.input_format = convert_color_format(in_color_format);
        process_config.output_format = convert_color_format(out_color_format);

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

// Initialize and cleanup

#if _WIN32

BOOL WINAPI DllMain(
    HINSTANCE hinstDLL,
    DWORD fdwReason,
    LPVOID lpvReserved) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            initialize_all();
            break;

        case DLL_PROCESS_DETACH:
            if (lpvReserved != nullptr) {
                break;
            }

            cleanup_all();
            break;
    }
    return TRUE;
}

#else

namespace {
class SharedLibraryLifecycle final {
   public:
    SharedLibraryLifecycle() {
        initialize_all();
    }

    ~SharedLibraryLifecycle() {
        cleanup_all();
    }
}

[[maybe_unused]]
SharedLibraryLifecycle g_lifecycle;
}  // namespace

#endif
