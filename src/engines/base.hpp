// Super-resolution base class for ncnn implementation
#ifndef SUPER_RESOLUTION_BASE_HPP
#define SUPER_RESOLUTION_BASE_HPP

#include <bitset>
#include <concepts>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

// ncnn
#include "gpu.h"
#include "layer.h"
#include "net.h"

#include "spdlog/spdlog.h"

// Supported scale factors
enum class SuperResolutionScale {
    X1 = 0,
    X2 = 1,
    X3 = 2,
    X4 = 3,
    COUNT = 4
};

// Engine feature flags
enum class SuperResolutionFeatureFlags {
    TTA_MODE = 0,   // TTA mode support
    NOISE = 1,      // Noise reduction support
    SYNC_GAP = 2,   // SyncGAP support
    CPU = 3,        // CPU processing support
    ALPHA = 4,      // Alpha channel processing support
    TILE_SIZE = 5,  // Tile size adjustment support
    COUNT = 6       // Total number of features
};

// Type-safe bitset for feature flags
template <typename EnumType>
concept EnumFlag = std::is_enum_v<EnumType>;

template <EnumFlag E, std::size_t N = static_cast<std::size_t>(E::COUNT)>
class FeatureFlagSet {
   private:
    std::bitset<N> bits;

   public:
    constexpr FeatureFlagSet() = default;

    constexpr FeatureFlagSet(E flag) {
        bits.set(static_cast<std::size_t>(flag));
    }

    template <typename... Args>
    constexpr FeatureFlagSet(E first, Args... rest) {
        bits.set(static_cast<std::size_t>(first));
        if constexpr (sizeof...(rest) > 0) {
            ((bits.set(static_cast<std::size_t>(rest))), ...);
        }
    }

    // Test if a flag is set
    [[nodiscard]] constexpr bool test(E flag) const {
        return bits.test(static_cast<std::size_t>(flag));
    }

    // Set a flag
    constexpr FeatureFlagSet& set(E flag, bool value = true) {
        bits.set(static_cast<std::size_t>(flag), value);
        return *this;
    }

    // Reset a flag
    constexpr FeatureFlagSet& reset(E flag) {
        bits.reset(static_cast<std::size_t>(flag));
        return *this;
    }

    // Reset all flags
    constexpr FeatureFlagSet& reset() {
        bits.reset();
        return *this;
    }

    // Bitwise operators
    constexpr FeatureFlagSet operator|(const FeatureFlagSet& other) const {
        FeatureFlagSet result;
        result.bits = bits | other.bits;
        return result;
    }

    constexpr FeatureFlagSet operator&(const FeatureFlagSet& other) const {
        FeatureFlagSet result;
        result.bits = bits & other.bits;
        return result;
    }

    constexpr FeatureFlagSet operator^(const FeatureFlagSet& other) const {
        FeatureFlagSet result;
        result.bits = bits ^ other.bits;
        return result;
    }

    constexpr FeatureFlagSet operator~() const {
        FeatureFlagSet result;
        result.bits = ~bits;
        return result;
    }

    // Compound assignment operators
    constexpr FeatureFlagSet& operator|=(const FeatureFlagSet& other) {
        bits |= other.bits;
        return *this;
    }

    constexpr FeatureFlagSet& operator&=(const FeatureFlagSet& other) {
        bits &= other.bits;
        return *this;
    }

    constexpr FeatureFlagSet& operator^=(const FeatureFlagSet& other) {
        bits ^= other.bits;
        return *this;
    }

    // Utility methods
    [[nodiscard]] constexpr bool any() const { return bits.any(); }
    [[nodiscard]] constexpr bool none() const { return bits.none(); }
    [[nodiscard]] constexpr bool all() const { return bits.all(); }
    [[nodiscard]] constexpr std::size_t count() const { return bits.count(); }

    // String conversion
    [[nodiscard]] std::string to_string() const { return bits.to_string(); }
};

// Helper function to create a flag set from a single flag
template <EnumFlag E>
constexpr auto make_flag(E flag) {
    FeatureFlagSet<E> result;
    result.set(flag);
    return result;
}

// Helper function to create a flag set from multiple flags
template <EnumFlag E, typename... Args>
constexpr auto make_flags(E first, Args... rest) {
    FeatureFlagSet<E> result;
    result.set(first);
    if constexpr (sizeof...(rest) > 0) {
        ((result.set(rest)), ...);
    }
    return result;
}

// Type alias for SuperResolutionFeatureFlags
using FeatureFlags = FeatureFlagSet<SuperResolutionFeatureFlags>;

// Type alias for SuperResolutionScale
using ScaleFlags = FeatureFlagSet<SuperResolutionScale>;

// Helper function to convert int scale to SuperResolutionScale
inline SuperResolutionScale int_to_scale(int scale) {
    switch (scale) {
        case 1:
            return SuperResolutionScale::X1;

        case 2:
            return SuperResolutionScale::X2;

        case 3:
            return SuperResolutionScale::X3;

        case 4:
            return SuperResolutionScale::X4;
    }
    throw std::invalid_argument("Invalid scale value: " + std::to_string(scale));
}

// Engine information structure
struct SuperResolutionEngineInfo {
    std::u8string engine_name;               // Engine name
    FeatureFlags supported_features;         // Supported feature flags
    ScaleFlags supported_scales;             // Supported scale factors (bit flags)
    std::vector<std::u8string> model_names;  // Supported model names
    std::u8string default_model;             // Default model name
    int default_noise = -1;                  // Default noise level
    std::u8string description;               // Engine description
    std::u8string version;                   // Engine version

    // Helper method to check if a feature is supported
    bool supports(SuperResolutionFeatureFlags feature) const {
        return supported_features.test(feature);
    }

    // Helper method to check if a scale is supported
    bool supports_scale(int scale) const {
        try {
            return supported_scales.test(int_to_scale(scale));
        } catch (const std::invalid_argument&) {
            return false;  // Invalid scale values are not supported
        }
    }

    // Helper method to check if a scale is supported (enum version)
    bool supports_scale(SuperResolutionScale scale) const {
        return supported_scales.test(scale);
    }

    // Helper method to check if a model is supported
    bool supports_model(const std::u8string& model) const {
        return std::find(model_names.begin(), model_names.end(), model) != model_names.end();
    }

    // Helper method to check if a config is compatible with this engine
    bool is_compatible_config(const struct SuperResolutionEngineConfig& config) const;
};

enum class ColorFormat {
    RGB = 0,
    BGR = 1
};

// Configuration for engine initialization
struct SuperResolutionEngineConfig {
    std::filesystem::path model_dir;  // Base path for model files
    std::u8string model;              // All engines

    int gpu_id = 0;         // All engines. -1 for CPU (RealCUGAN, RealSR, Waifu2x)
    bool tta_mode = false;  // All engines
    int num_threads = 1;    // RealCUGAN, RealSR, Waifu2x

    int noise = -1;    // RealCUGAN, SRMD, Waifu2x (-1, 0, 1, 2, 3). Noise level should be set during initialization as it controls model loading.
    int sync_gap = 0;  // RealCUGAN only (0, 1, 2, 3)

    // Engine name (optional)
    std::u8string engine_name;

    std::shared_ptr<spdlog::logger> logger_error = spdlog::default_logger();
};

// Forward declaration for is_compatible_config method
inline bool SuperResolutionEngineInfo::is_compatible_config(const SuperResolutionEngineConfig& config) const {
    // Check model
    if (!config.model.empty() && !supports_model(config.model)) {
        return false;
    }

    // Check TTA mode
    if (config.tta_mode && !supports(SuperResolutionFeatureFlags::TTA_MODE)) {
        return false;
    }

    // Check noise level
    if (config.noise >= 0 && !supports(SuperResolutionFeatureFlags::NOISE)) {
        return false;
    }

    // Check SyncGAP
    if (config.sync_gap > 0 && !supports(SuperResolutionFeatureFlags::SYNC_GAP)) {
        return false;
    }

    // Check CPU mode
    if (config.gpu_id < 0 && !supports(SuperResolutionFeatureFlags::CPU)) {
        return false;
    }

    return true;
}

// Configuration for processing
struct ProcessConfig {
    int scale = 2;  // All engines (1, 2, 3, 4)

    ColorFormat input_format = ColorFormat::RGB;   // All engines
    ColorFormat output_format = ColorFormat::RGB;  // All engines

    int tile_size = 0;  // All engines
};

// Bicubic layers management
class BicubicLayers final {
   public:
    BicubicLayers(ncnn::VulkanDevice* vkdev, const ncnn::Option& opt);
    ~BicubicLayers() = default;

    std::shared_ptr<ncnn::Layer> get_bicubic(int scale) const;

   private:
    std::unordered_map<int, std::shared_ptr<ncnn::Layer>> bicubics;

    ncnn::VulkanDevice* vkdev;
    ncnn::Option opt;
};

// Combined pipeline structure for preprocessing and postprocessing
struct SuperResolutionPipelines final {
    ncnn::Pipeline preprocess_rgb;
    ncnn::Pipeline preprocess_bgr;
    ncnn::Pipeline postprocess_rgb;
    ncnn::Pipeline postprocess_bgr;

    SuperResolutionPipelines(const ncnn::VulkanDevice* vkdev);
    ~SuperResolutionPipelines() = default;

    SuperResolutionPipelines(const SuperResolutionPipelines&) = delete;

    SuperResolutionPipelines& operator=(const SuperResolutionPipelines&) = delete;
};

// Pipeline cache
class PipelineCache final {
    // Key: scale, Value: pipeline structure
    mutable std::unordered_map<int, std::shared_ptr<SuperResolutionPipelines>> pipelines;

    // Factory function for creating pipelines
    std::function<std::shared_ptr<SuperResolutionPipelines>(int)> pipeline_factory;

   public:
    PipelineCache(std::function<std::shared_ptr<SuperResolutionPipelines>(int)> factory);

    std::shared_ptr<SuperResolutionPipelines> get_pipelines(int scale) const;

    void clear();
};

// Net cache
// This is required since some engines use different models for different scales
class NetCache final {
    // Key: scale, Value: net structure
    mutable std::unordered_map<int, std::shared_ptr<ncnn::Net>> nets;

    // Factory function for creating nets
    std::function<std::shared_ptr<ncnn::Net>(int)> net_factory;

   public:
    NetCache(std::function<std::shared_ptr<ncnn::Net>(int)> factory);

    std::shared_ptr<ncnn::Net> get_net(int scale) const;

    void clear();
};

// Base class for super-resolution engines
class SuperResolutionEngine {
   protected:
    // Configuration
    SuperResolutionEngineConfig config;

    // Vulkan device
    ncnn::VulkanDevice* vkdev;

    // Neural network
    mutable NetCache net_cache;

    // Pipeline cache
    mutable PipelineCache pipeline_cache;

    // Bicubic layers (managed by base class)
    BicubicLayers bicubic_layers;

    // Protected utility methods
    virtual std::shared_ptr<ncnn::Net> create_net_base() const;
    virtual std::shared_ptr<ncnn::Net> create_net(int scale, const NetCache& net_cache) const = 0;
    virtual std::shared_ptr<SuperResolutionPipelines> create_pipelines(int scale, const PipelineCache& pipeline_cache) const = 0;

    // Process implementation methods
    virtual int process_gpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const = 0;
    virtual int process_cpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const;

    // Helper methods
    virtual void prepare_net_options(ncnn::Option& options) const;
    virtual int handle_alpha_channel_gpu(const ncnn::VkMat& in_alpha_tile, ncnn::VkMat& out_alpha_tile, int scale, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

    int net_load_model(ncnn::Net& net, const std::filesystem::path& path) const;
    int net_load_param(ncnn::Net& net, const std::filesystem::path& path) const;

    int net_load_model_and_param(ncnn::Net& net, const std::filesystem::path& path) const;

   public:
    // Constructor and destructor
    SuperResolutionEngine(const SuperResolutionEngineConfig& config);
    virtual ~SuperResolutionEngine();

    virtual const SuperResolutionEngineInfo& engine_info() const = 0;

    virtual int preload(int scale) const;

    // Image processing
    virtual int process(const ncnn::Mat& in, ncnn::Mat& out, const ProcessConfig& config) const;

    virtual int get_default_tile_size() const = 0;
};

#endif  // SUPER_RESOLUTION_BASE_HPP
