#ifndef REALCUGAN_HPP
#define REALCUGAN_HPP

#include <functional>

#include "../base.hpp"

class RealCUGANSyncGapGPU;

// RealCUGAN class derived from SuperResolutionEngine
class RealCUGAN final : public SuperResolutionEngine {
   public:
    // Engine information
    static const SuperResolutionEngineInfo& get_engine_info() {
        static const SuperResolutionEngineInfo info{
            .engine_name = u8"realcugan",
            .supported_features = FeatureFlags(
                SuperResolutionFeatureFlags::TTA_MODE,
                SuperResolutionFeatureFlags::NOISE,
                SuperResolutionFeatureFlags::SYNC_GAP,
                SuperResolutionFeatureFlags::ALPHA,
                SuperResolutionFeatureFlags::TILE_SIZE),
            .supported_scales = ScaleFlags(
                SuperResolutionScale::X2,
                SuperResolutionScale::X3,
                SuperResolutionScale::X4),
            .model_names = {
                u8"models-se",
                u8"models-nose",
                u8"models-pro",
            },
            .default_model = u8"models-se",
            .default_noise = 0,
            .description = u8"Real-CUGAN: Real-world Cartoon Image Super-Resolution",
            .version = u8"1.0.0",
        };
        return info;
    }

    // Constructor and destructor
    RealCUGAN(const SuperResolutionEngineConfig& config);
    virtual ~RealCUGAN() = default;

    const SuperResolutionEngineInfo& engine_info() const override;

    int get_default_tile_size() const override;

   protected:
    void prepare_net_options(ncnn::Option& options) const override;

    std::shared_ptr<ncnn::Net> create_net(int scale, const NetCache& net_cache) const override;

    std::shared_ptr<SuperResolutionPipelines> create_pipelines(int scale, const PipelineCache&) const override;

    // Override GPU processing method
    int process_gpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const override;

   private:
    std::unique_ptr<RealCUGANSyncGapGPU> create_sync_gap_gpu(const ncnn::Mat& in, ColorFormat in_format, const ProcessConfig& config) const;

    // Helper methods for SyncGAP processing with different levels
    // Level 0: No SyncGAP processing
    int process_gpu_nose(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const;
    // Level 1: Detailed SyncGAP processing
    int process_gpu_se(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const;
    // Level 2: Rough SyncGAP processing
    int process_gpu_se_rough(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const;
    // Level 3: Very rough SyncGAP processing
    int process_gpu_se_very_rough(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const;

    // Helper methods for loading models
    std::u8string get_model_path(const std::u8string& model_type, int scale, int noise) const;
};

#endif  // REALCUGAN_HPP
