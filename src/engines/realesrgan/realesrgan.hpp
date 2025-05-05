// RealESRGAN implemented with ncnn library - refactored version
#ifndef REALESRGAN_HPP
#define REALESRGAN_HPP

#include "../base.hpp"

// RealESRGAN class derived from SuperResolutionEngine
class RealESRGAN final : public SuperResolutionEngine {
   public:
    // Constructor and destructor
    RealESRGAN(const SuperResolutionEngineConfig& config);
    virtual ~RealESRGAN() = default;

    ProcessConfig create_default_process_config() const override;

    // Engine information
    static const SuperResolutionEngineInfo& get_engine_info() {
        static const SuperResolutionEngineInfo info{
            .engine_name = "realesrgan",
            .supported_features = FeatureFlags(
                SuperResolutionFeatureFlags::TTA_MODE,
                SuperResolutionFeatureFlags::ALPHA,
                SuperResolutionFeatureFlags::TILESIZE),
            .supported_scales = ScaleFlags(
                SuperResolutionScale::X2,
                SuperResolutionScale::X3,
                SuperResolutionScale::X4),
            .model_names = {"realesrgan-x4plus", "realesrnet-x4plus", "realesrgan-x4plus-anime", "realesr-animevideov3"},
            .default_model = "realesrgan-x4plus",
            .description = "Enhanced ESRGAN implementation with pure-CNN architecture",
            .version = "1.0.0"};
        return info;
    }

   protected:
    void prepare_net_options(ncnn::Option& options) const override;

    std::shared_ptr<ncnn::Net> create_net(int scale, const NetCache& net_cache) const override;

    std::shared_ptr<SuperResolutionPipelines> create_pipelines(int scale, const PipelineCache&) const override;

    // Override GPU processing method
    int process_gpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const override;

    // RealESRGAN doesn't support CPU processing
    int process_cpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const override;
};

#endif  // REALESRGAN_HPP
