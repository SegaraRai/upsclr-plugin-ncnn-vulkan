// Super-resolution base class for ncnn implementation
#ifndef SUPERRESOLUTION_BASE_HPP
#define SUPERRESOLUTION_BASE_HPP

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

// ncnn
#include "gpu.h"
#include "layer.h"
#include "net.h"

enum class ColorFormat {
    RGB = 0,
    BGR = 1
};

// Configuration for engine initialization
struct SuperResolutionEngineConfig {
    std::filesystem::path model_dir;  // Base path for model files
    std::string model;                // All engines

    int gpuid = 0;          // All engines. -1 for CPU (RealCUGAN, RealSR, Waifu2x)
    bool tta_mode = false;  // All engines
    int num_threads = 1;    // RealCUGAN, RealSR, Waifu2x

    int noise = -1;   // RealCUGAN, SRMD, Waifu2x (-1, 0, 1, 2, 3). Noise level should be set during initialization as it controls model loading.
    int syncgap = 0;  // RealCUGAN only (0, 1, 2, 3)
};

// Configuration for processing
struct ProcessConfig {
    int scale = 2;  // All engines (1, 2, 3, 4)

    ColorFormat input_format = ColorFormat::RGB;   // All engines
    ColorFormat output_format = ColorFormat::RGB;  // All engines

    int tilesize = 400;   // All engines
    int prepadding = 12;  // All engines
};

// Bicubic layers management
class BicubicLayers {
   public:
    BicubicLayers(ncnn::VulkanDevice* vkdev, const ncnn::Option& opt);
    ~BicubicLayers();

    std::shared_ptr<ncnn::Layer> get_bicubic(int scale) const;

   private:
    std::unordered_map<int, std::shared_ptr<ncnn::Layer>> bicubics;

    ncnn::VulkanDevice* vkdev;
    ncnn::Option opt;
};

// Combined pipeline structure for preprocessing and postprocessing
struct SuperResolutionPipelines {
    ncnn::Pipeline preprocess_rgb;
    ncnn::Pipeline preprocess_bgr;
    ncnn::Pipeline postprocess_rgb;
    ncnn::Pipeline postprocess_bgr;

    SuperResolutionPipelines(const ncnn::VulkanDevice* vkdev);
    ~SuperResolutionPipelines() = default;

    SuperResolutionPipelines(const SuperResolutionPipelines&) = delete;
    SuperResolutionPipelines(SuperResolutionPipelines&&) = default;

    SuperResolutionPipelines& operator=(const SuperResolutionPipelines&) = delete;
    SuperResolutionPipelines& operator=(SuperResolutionPipelines&&) = default;
};

// Pipeline cache
class PipelineCache {
   public:
    PipelineCache(std::function<std::shared_ptr<SuperResolutionPipelines>(int)> factory);

    std::shared_ptr<SuperResolutionPipelines> get_pipelines(int scale) const;

    void clear();

   private:
    // Key: scale, Value: pipeline structure
    mutable std::unordered_map<int, std::shared_ptr<SuperResolutionPipelines>> pipelines;

    // Factory function for creating pipelines
    std::function<std::shared_ptr<SuperResolutionPipelines>(int)> pipeline_factory;
};

// Net cache
// This is required since some engines use different models for different scales
class NetCache {
   public:
    NetCache(std::function<std::shared_ptr<ncnn::Net>(int)> factory);

    std::shared_ptr<ncnn::Net> get_net(int scale) const;

    void clear();

   private:
    // Key: scale, Value: net structure
    mutable std::unordered_map<int, std::shared_ptr<ncnn::Net>> nets;

    // Factory function for creating nets
    std::function<std::shared_ptr<ncnn::Net>(int)> net_factory;
};

// Base class for super-resolution engines
class SuperResolutionEngine {
   public:
    // Constructor and destructor
    SuperResolutionEngine(const SuperResolutionEngineConfig& config);
    virtual ~SuperResolutionEngine();

    // Image processing
    virtual int process(const ncnn::Mat& in, ncnn::Mat& out, const ProcessConfig& config) const;

    virtual ProcessConfig create_default_process_config() const;

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
    virtual std::shared_ptr<ncnn::Net> create_net(int scale, const NetCache& net_cache) const = 0;
    virtual std::shared_ptr<SuperResolutionPipelines> create_pipelines(int scale, const PipelineCache& pipeline_cache) const = 0;

    // Process implementation methods
    virtual int process_gpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const = 0;
    virtual int process_cpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const;

    // Helper methods
    virtual void prepare_net_options(ncnn::Option& options) const;
    virtual int handle_alpha_channel_gpu(const ncnn::VkMat& in_alpha_tile, ncnn::VkMat& out_alpha_tile, int scale, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

    static int net_load_model(ncnn::Net& net, const std::filesystem::path& path);
    static int net_load_param(ncnn::Net& net, const std::filesystem::path& path);

    static int net_load_model_and_param(ncnn::Net& net, const std::filesystem::path& path);
};

#endif  // SUPERRESOLUTION_BASE_HPP
