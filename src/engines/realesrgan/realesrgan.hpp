// RealESRGAN implemented with ncnn library - refactored version
#ifndef REALESRGAN_HPP
#define REALESRGAN_HPP

#include "../base.hpp"

// RealESRGAN class derived from SuperResolutionEngine
class RealESRGAN final : public SuperResolutionEngine {
   public:
    // Engine information
    static const SuperResolutionEngineInfo& get_engine_info() {
        static const SuperResolutionEngineInfo info{
            .engine_name = u8"realesrgan",
            .supported_features = FeatureFlags(
                SuperResolutionFeatureFlags::TTA_MODE,
                SuperResolutionFeatureFlags::ALPHA,
                SuperResolutionFeatureFlags::TILE_SIZE),
            .supported_scales = ScaleFlags(
                SuperResolutionScale::X2,
                SuperResolutionScale::X3,
                SuperResolutionScale::X4),
            .model_names = {
                u8"realesrgan-x4plus",
                u8"realesrnet-x4plus",
                u8"realesrgan-x4plus-anime",
                u8"realesr-animevideov3",
            },
            .default_model = u8"realesrgan-x4plus",
            .description = u8"Enhanced ESRGAN implementation with pure-CNN architecture",
            .version = u8"1.0.0",
        };
        return info;
    }

    // Constructor and destructor
    RealESRGAN(const SuperResolutionEngineConfig& config);
    virtual ~RealESRGAN() = default;

    int get_default_tile_size() const override;

   protected:
    void prepare_net_options(ncnn::Option& options) const override;

    std::shared_ptr<ncnn::Net> create_net(int scale, const NetCache& net_cache) const override;

    std::shared_ptr<SuperResolutionPipelines> create_pipelines(int scale, const PipelineCache&) const override;

    // Override GPU processing method
    int process_gpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const override;
};

#endif  // REALESRGAN_HPP
