// RealESRGAN implemented with ncnn library - refactored version
#ifndef REALESRGAN_REFACTOR_HPP
#define REALESRGAN_REFACTOR_HPP

#include "../base.hpp"

// RealESRGAN class derived from SuperResolutionEngine
class RealESRGAN final : public SuperResolutionEngine {
   public:
    // Constructor and destructor
    RealESRGAN(const SuperResolutionEngineConfig& config);
    virtual ~RealESRGAN() = default;

    ProcessConfig create_default_process_config() const override;

   protected:
    void prepare_net_options(ncnn::Option& options) const override;

    std::shared_ptr<ncnn::Net> create_net(int scale, const NetCache& net_cache) const override;

    std::shared_ptr<SuperResolutionPipelines> create_pipelines(int scale, const PipelineCache&) const override;

    // Override GPU processing method
    int process_gpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const override;

    // RealESRGAN doesn't support CPU processing
    int process_cpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const override;
};

#endif  // REALESRGAN_REFACTOR_HPP
