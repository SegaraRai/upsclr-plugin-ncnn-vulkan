// Super-resolution base class implementation
#include "base.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <vector>

//------------------------------------------------------------------------------
// VkAllocators implementation
//------------------------------------------------------------------------------

VkAllocators::VkAllocators(ncnn::VulkanDevice* vkdev) {
    if (vkdev == nullptr) {
        return;
    }

    this->blob_vkallocator_ptr = std::unique_ptr<ncnn::VkAllocator, std::function<void(ncnn::VkAllocator*)>>(
        vkdev->acquire_blob_allocator(),
        [vkdev](ncnn::VkAllocator* allocator) {
            if (allocator != nullptr) {
                vkdev->reclaim_blob_allocator(allocator);
            }
        });

    this->staging_vkallocator_ptr = std::unique_ptr<ncnn::VkAllocator, std::function<void(ncnn::VkAllocator*)>>(
        vkdev->acquire_staging_allocator(),
        [vkdev](ncnn::VkAllocator* allocator) {
            if (allocator != nullptr) {
                vkdev->reclaim_staging_allocator(allocator);
            }
        });
}

ncnn::VkAllocator* VkAllocators::get_blob_allocator() const {
    return this->blob_vkallocator_ptr.get();
}

ncnn::VkAllocator* VkAllocators::get_staging_allocator() const {
    return this->staging_vkallocator_ptr.get();
}

//------------------------------------------------------------------------------
// BicubicLayers implementation
//------------------------------------------------------------------------------

BicubicLayers::BicubicLayers(ncnn::VulkanDevice* _vkdev, const ncnn::Option& _opt)
    : vkdev(_vkdev), opt(_opt) {
    // FIXME: support CPU?
    if (this->vkdev == nullptr) {
        return;
    }

    // Create bicubic layers for scales 2, 3, and 4
    for (const int scale : {2, 3, 4}) {
        ncnn::Layer* layer = ncnn::create_layer("Interp");
        layer->vkdev = this->vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 3);  // bicubic
        pd.set(1, float(scale));
        pd.set(2, float(scale));
        layer->load_param(pd);

        layer->create_pipeline(this->opt);

        this->bicubics.emplace(scale, std::shared_ptr<ncnn::Layer>(layer, [this](ncnn::Layer* l) {
                                   if (l != nullptr) {
                                       l->destroy_pipeline(this->opt);
                                       delete l;
                                   }
                               }));
    }
}

std::shared_ptr<ncnn::Layer> BicubicLayers::get_bicubic(int scale) const {
    const auto it = this->bicubics.find(scale);
    if (it != this->bicubics.end()) {
        return it->second;
    }

    return nullptr;
}

//------------------------------------------------------------------------------
// SuperResolutionPipelines implementation
//------------------------------------------------------------------------------

SuperResolutionPipelines::SuperResolutionPipelines(const ncnn::VulkanDevice* vkdev)
    : preprocess_rgb(vkdev),
      preprocess_bgr(vkdev),
      postprocess_rgb(vkdev),
      postprocess_bgr(vkdev) {
}

//------------------------------------------------------------------------------
// PipelineCache implementation
//------------------------------------------------------------------------------

PipelineCache::PipelineCache(std::function<std::shared_ptr<SuperResolutionPipelines>(int)> factory)
    : pipeline_factory(factory) {
}

std::shared_ptr<SuperResolutionPipelines> PipelineCache::get_pipelines(int scale) const {
    std::lock_guard lk(this->mutex);

    const auto it = this->pipelines.find(scale);
    if (it != this->pipelines.end()) {
        return it->second;
    }

    // Create new pipelines
    const auto result = this->pipelines.emplace(scale, this->pipeline_factory(scale));
    return result.first->second;
}

void PipelineCache::clear() {
    std::lock_guard lk(this->mutex);

    this->pipelines.clear();
}

//------------------------------------------------------------------------------
// NetCache implementation
//------------------------------------------------------------------------------

NetCache::NetCache(std::function<std::shared_ptr<ncnn::Net>(int)> factory)
    : net_factory(factory) {
}

std::shared_ptr<ncnn::Net> NetCache::get_net(int scale) const {
    std::lock_guard lk(this->mutex);

    const auto it = this->nets.find(scale);
    if (it != this->nets.end()) {
        return it->second;
    }

    // Create new net
    const auto result = this->nets.emplace(scale, this->net_factory(scale));
    return result.first->second;
}

void NetCache::clear() {
    std::lock_guard lk(this->mutex);

    this->nets.clear();
}

//------------------------------------------------------------------------------
// SuperResolutionEngine implementation
//------------------------------------------------------------------------------

ncnn::VulkanDevice* SuperResolutionEngine::get_gpu_device_or_throw(int gpu_id, const std::shared_ptr<spdlog::logger>& logger_error) {
    if (gpu_id < 0) {
        return nullptr;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_id);
    if (vkdev == nullptr || !vkdev->is_valid()) {
        logger_error->error("[{}] Failed to acquire Vulkan device with ID {}", __func__, gpu_id);
        throw std::runtime_error("Failed to acquire Vulkan device");
    }

    return vkdev;
}

SuperResolutionEngine::SuperResolutionEngine(const SuperResolutionEngineInfo& _engine_info, const SuperResolutionEngineConfig& _config)
    : engine_info(_engine_info),
      config(_config),
      vkdev(get_gpu_device_or_throw(config.gpu_id, config.logger_error)),
      net_cache([this](int scale) -> std::shared_ptr<ncnn::Net> {
          auto net = this->create_net(scale, this->net_cache);
          if (net == nullptr) {
              this->config.logger_error->error("create_net returned nullptr for scale {}", scale);
              return nullptr;
          }
          return net;
      }),
      pipeline_cache([this](int scale) -> std::shared_ptr<SuperResolutionPipelines> {
          auto pipelines = this->create_pipelines(scale, this->pipeline_cache);
          if (pipelines == nullptr) {
              this->config.logger_error->error("create_pipelines returned nullptr for scale {}", scale);
              return nullptr;
          }
          return pipelines;
      }),
      bicubic_layers(this->vkdev, ([this]() {
                         ncnn::Option opt;
                         this->prepare_net_options(opt);
                         return opt;
                     })()) {
    if (this->vkdev == nullptr && !this->engine_info.supports(SuperResolutionFeatureFlags::CPU)) {
        this->config.logger_error->error("[{}] This engine requires a Vulkan device to run", __func__);
        throw std::runtime_error("This engine requires a Vulkan device to run");
    }

    if (!this->engine_info.is_compatible_config(config)) {
        this->config.logger_error->warn("[{}] Configuration may not be fully compatible with RealCUGAN engine", __func__);
    }
}

SuperResolutionEngine::~SuperResolutionEngine() {
    // Clean up resources
    this->pipeline_cache.clear();
    this->net_cache.clear();
}

const SuperResolutionEngineInfo& SuperResolutionEngine::get_engine_info() const {
    return this->engine_info;
}

int SuperResolutionEngine::process(const ncnn::Mat& in, ncnn::Mat& out, const ProcessConfig& config) const {
    if (!this->engine_info.supports_scale(config.scale)) {
        this->config.logger_error->error("[{}] Unsupported scale {}", __func__, config.scale);
        return -1;
    }

    if (this->vkdev != nullptr) {
        return this->process_gpu(in, config.input_format, out, config.output_format, config);
    } else {
        return this->process_cpu(in, config.input_format, out, config.output_format, config);
    }
}

std::shared_ptr<ncnn::Net> SuperResolutionEngine::create_net_base() const {
    // FIXME: support CPU?
    if (this->vkdev == nullptr) {
        return nullptr;
    }

    auto net = std::make_shared<ncnn::Net>();
    this->prepare_net_options(net->opt);
    net->set_vulkan_device(this->vkdev);
    return net;
}

int SuperResolutionEngine::preload(int scale) const {
    if (!this->engine_info.supports_scale(scale)) {
        this->config.logger_error->error("[{}] Unsupported scale {}", __func__, scale);
        return -1;
    }

    // Get the network for the current scale
    const auto ptr_net = net_cache.get_net(scale);
    if (ptr_net == nullptr) {
        this->config.logger_error->error("[{}] Failed to get net for scale {}", __func__, scale);
        return -1;
    }

    // Get pipelines for the current scale
    const auto ptr_pipelines = pipeline_cache.get_pipelines(scale);
    if (ptr_pipelines == nullptr) {
        this->config.logger_error->error("[{}] Failed to get pipelines for scale {}", __func__, scale);
        return -1;
    }

    return 0;
}

int SuperResolutionEngine::process_cpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const {
    // Base class provides an empty implementation
    // Derived classes should override this method
    this->config.logger_error->error("[{}] process_cpu not implemented", __func__);
    return -1;
}

void SuperResolutionEngine::prepare_net_options(ncnn::Option& options) const {
    // Set basic options
    options.num_threads = config.num_threads;

    // Set Vulkan compute options if available
    if (this->vkdev != nullptr) {
        options.use_vulkan_compute = true;
    } else {
        options.use_vulkan_compute = false;
    }
}

int SuperResolutionEngine::handle_alpha_channel_gpu(const ncnn::VkMat& in_alpha_tile, ncnn::VkMat& out_alpha_tile, int scale, ncnn::VkCompute& cmd, const ncnn::Option& opt) const {
    if (scale == 1) {
        out_alpha_tile = in_alpha_tile;
        return 0;
    }

    const auto ptr_bicubic = this->bicubic_layers.get_bicubic(scale);
    if (ptr_bicubic == nullptr) {
        this->config.logger_error->error("[{}] Failed to get bicubic layer for scale {}", __func__, scale);
        return -1;
    }
    ptr_bicubic->forward(in_alpha_tile, out_alpha_tile, cmd, opt);
    return 0;
}

int SuperResolutionEngine::net_load_model(ncnn::Net& net, const std::filesystem::path& path) const {
#if _WIN32
    FILE* fp = nullptr;
    if (const auto result = _wfopen_s(&fp, path.wstring().c_str(), L"rb"); result != 0) {
        this->config.logger_error->error(L"[net_load_model] Failed to open model file: {}", path.wstring());
        return result;
    }

    const auto result = net.load_model(fp);
    fclose(fp);

    return result;
#else
    const auto result = net.load_model(path.c_str());
    if (result != 0) {
        this->config.logger_error->error("[{}] Failed to load model file: {}", __func__, path.string());
    }
    return result;
#endif
}

int SuperResolutionEngine::net_load_param(ncnn::Net& net, const std::filesystem::path& path) const {
#if _WIN32
    FILE* fp = nullptr;
    if (const auto result = _wfopen_s(&fp, path.wstring().c_str(), L"rb"); result != 0) {
        this->config.logger_error->error(L"[net_load_param] Failed to open param file: {}", path.wstring());
        return result;
    }

    const auto result = net.load_param(fp);
    fclose(fp);

    return result;
#else
    const auto result = net.load_param(path.c_str());
    if (result != 0) {
        this->config.logger_error->error("[{}] Failed to load param file: {}", __func__, path.string());
    }
    return result;
#endif
}

int SuperResolutionEngine::net_load_model_and_param(ncnn::Net& net, const std::filesystem::path& path) const {
    auto p = path;

    if (const auto result = SuperResolutionEngine::net_load_param(net, p.replace_extension(".param")); result != 0) {
        return result;
    }

    if (const auto result = SuperResolutionEngine::net_load_model(net, p.replace_extension(".bin")); result != 0) {
        return result;
    }

    return 0;
}
