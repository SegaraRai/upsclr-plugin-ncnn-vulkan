#include "realcugan.hpp"

#include "../../encoding_utils.hpp"

#include "ncnn/cpu.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace {

const uint32_t realcugan_preproc_spv_data[] = {
#include "shaders/realcugan_preproc.spv.hex.h"
};
const uint32_t realcugan_preproc_fp16s_spv_data[] = {
#include "shaders/realcugan_preproc_fp16s.spv.hex.h"
};
const uint32_t realcugan_preproc_int8s_spv_data[] = {
#include "shaders/realcugan_preproc_int8s.spv.hex.h"
};
const uint32_t realcugan_postproc_spv_data[] = {
#include "shaders/realcugan_postproc.spv.hex.h"
};
const uint32_t realcugan_postproc_fp16s_spv_data[] = {
#include "shaders/realcugan_postproc_fp16s.spv.hex.h"
};
const uint32_t realcugan_postproc_int8s_spv_data[] = {
#include "shaders/realcugan_postproc_int8s.spv.hex.h"
};

const uint32_t realcugan_preproc_tta_spv_data[] = {
#include "shaders/realcugan_preproc_tta.spv.hex.h"
};
const uint32_t realcugan_preproc_tta_fp16s_spv_data[] = {
#include "shaders/realcugan_preproc_tta_fp16s.spv.hex.h"
};
const uint32_t realcugan_preproc_tta_int8s_spv_data[] = {
#include "shaders/realcugan_preproc_tta_int8s.spv.hex.h"
};
const uint32_t realcugan_postproc_tta_spv_data[] = {
#include "shaders/realcugan_postproc_tta.spv.hex.h"
};
const uint32_t realcugan_postproc_tta_fp16s_spv_data[] = {
#include "shaders/realcugan_postproc_tta_fp16s.spv.hex.h"
};
const uint32_t realcugan_postproc_tta_int8s_spv_data[] = {
#include "shaders/realcugan_postproc_tta_int8s.spv.hex.h"
};

const uint32_t realcugan_4x_postproc_spv_data[] = {
#include "shaders/realcugan_4x_postproc.spv.hex.h"
};
const uint32_t realcugan_4x_postproc_fp16s_spv_data[] = {
#include "shaders/realcugan_4x_postproc_fp16s.spv.hex.h"
};
const uint32_t realcugan_4x_postproc_int8s_spv_data[] = {
#include "shaders/realcugan_4x_postproc_int8s.spv.hex.h"
};
const uint32_t realcugan_4x_postproc_tta_spv_data[] = {
#include "shaders/realcugan_4x_postproc_tta.spv.hex.h"
};
const uint32_t realcugan_4x_postproc_tta_fp16s_spv_data[] = {
#include "shaders/realcugan_4x_postproc_tta_fp16s.spv.hex.h"
};
const uint32_t realcugan_4x_postproc_tta_int8s_spv_data[] = {
#include "shaders/realcugan_4x_postproc_tta_int8s.spv.hex.h"
};

constexpr int get_prepadding_for_scale(int scale) {
    switch (scale) {
        case 2:
            return 18;
        case 3:
            return 14;
        case 4:
            return 19;
    }

    // Unsupported scale
    return 0;
}

enum class StorageMode {
    FP16_INT8,
    FP16,
    OTHER,
};

StorageMode get_storage_mode(const ncnn::Option& options) {
    if (options.use_fp16_storage && options.use_int8_storage) {
        return StorageMode::FP16_INT8;
    } else if (options.use_fp16_storage) {
        return StorageMode::FP16;
    } else {
        return StorageMode::OTHER;
    }
}

struct Tile {
    size_t elemsize;

    int size_x;
    int size_y;

    int w_nopad;
    int h_nopad;

    int xi;
    int yi;

    int x0;
    int y0;
    int x1;
    int y1;
};

template <typename T>
class FeatureCache {
    std::unordered_map<std::string, T> cache;

    static inline std::string make_key(int yi, int xi, int ti, const std::string& name) {
        return std::format("{}-{}-{}-{}", yi, xi, ti, name);
    }

   public:
    void clear() {
        cache.clear();
    }

    void load(int yi, int xi, int ti, const std::string& name, T& feat) {
        feat = cache[make_key(yi, xi, ti, name)];
    }

    void save(int yi, int xi, int ti, const std::string& name, T& feat) {
        cache[make_key(yi, xi, ti, name)] = feat;
    }
};

void extract_features(ncnn::Net& net, const ncnn::Option& options, const ncnn::VkMat& in_tile, ncnn::VkMat& out_tile, ncnn::VkCompute& cmd);
void preprocess_tile_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, ncnn::VkMat& in_tile_gpu, ncnn::VkMat& in_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);
void preprocess_tile_tta_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, ncnn::VkMat in_tile_gpu[8], ncnn::VkMat& in_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);
void postprocess_tile_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& out_gpu_row, ncnn::VkMat& out_tile_gpu, ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);
void postprocess_tile_tta_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& out_gpu_row, ncnn::VkMat out_tile_gpu[8], ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);
void postprocess_tile_4x_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, const ncnn::VkMat& out_gpu_row, ncnn::VkMat& out_tile_gpu, ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);
void postprocess_tile_4x_tta_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, const ncnn::VkMat& out_gpu_row, ncnn::VkMat out_tile_gpu[8], ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);

}  // namespace

// SyncGAP GPU processing class

class RealCUGANSyncGapGPU {
    static constexpr int VERY_ROUGH_TILE_SIZE = 32;

    ncnn::VulkanDevice* vkdev;
    const ncnn::Net& net;
    const SuperResolutionPipelines& pipelines;
    ncnn::Option opt;
    std::shared_ptr<spdlog::logger> logger_error;
    const ProcessConfig& config;
    int prepadding;
    const ncnn::Mat& in;  // Reference to input image Mat
    ColorFormat in_format;
    bool tta_mode;
    std::function<int(const ncnn::VkMat& in_alpha_tile, ncnn::VkMat& out_alpha_tile, int scale, ncnn::VkCompute& cmd, const ncnn::Option& opt)> handle_alpha_channel_gpu;

    FeatureCache<ncnn::VkMat> cache;

    // Helper to generate feature names like "gap0", "gap1" etc.
    static std::vector<std::string> generate_feature_names(int begin, int end) {
        std::vector<std::string> names;
        names.reserve(end - begin);
        for (int i = begin; i < end; ++i) {
            names.push_back(std::format("gap{}", i));
        }
        return names;
    }

    void extract_features_sync_gap(const ncnn::Net& net, const ncnn::Option& options, const ncnn::VkMat& in_tile, const std::vector<std::string>& in_names, const std::vector<std::string>& out_names, FeatureCache<ncnn::VkMat>& cache, int yi, int xi, int ti, ncnn::VkCompute& cmd) {
        ncnn::Extractor ex = net.create_extractor();

        ex.set_blob_vkallocator(options.blob_vkallocator);
        ex.set_workspace_vkallocator(options.workspace_vkallocator);
        ex.set_staging_vkallocator(options.staging_vkallocator);

        ex.input("in0", in_tile);

        // Input cached features
        for (const auto& name : in_names) {
            ncnn::VkMat feat;
            cache.load(yi, xi, ti, name, feat);
            if (!feat.empty()) {  // Only input if feature exists in cache
                ex.input(name.c_str(), feat);
            } else {
                // Handle missing input features if necessary (e.g., log warning)
                this->logger_error->warn("[{}] Missing input feature '{}' for tile ({}, {}, {})", __func__, name, yi, xi, ti);
            }
        }

        // Extract and save output features
        for (const auto& name : out_names) {
            ncnn::VkMat feat;
            ex.extract(name.c_str(), feat, cmd);
            cache.save(yi, xi, ti, name, feat);  // Save extracted feature to cache
        }
    }

    struct Stage2Options {
        ncnn::VkMat& out_gpu_row;
        ncnn::Mat& out;
        ColorFormat out_format;
    };

    // Helper function similar to parts of process_se_stage0/stage2
    void process_tile_stage0_or_2(
        int yi,
        const ncnn::VkMat& in_gpu_row,
        const std::vector<std::string>& in_names,
        const std::vector<std::string>& out_names_or_final,  // "out0" for stage2, gap names for stage0
        std::optional<Stage2Options> stage2_options          // Options for stage2 processing
    ) {
        const auto* ptr_stage2_options = stage2_options.has_value() ? &stage2_options.value() : nullptr;

        const int w = in.w;
        const int h = in.h;
        const int channels = in.elempack;
        const int scale = config.scale;
        const int prepadding = this->prepadding;
        const int tile_size = config.tile_size;  // Use config.tile_size for standard stages

        const int TILE_SIZE_X = tile_size;
        const int TILE_SIZE_Y = tile_size;
        const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;

        const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

        ncnn::VkCompute cmd(vkdev);

        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;  // Need the outer loop's prepadding_bottom
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        } else if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;

            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            } else if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            const Tile tile{
                .elemsize = in_out_tile_elemsize,
                .size_x = TILE_SIZE_X,
                .size_y = TILE_SIZE_Y,
                .w_nopad = tile_w_nopad,
                .h_nopad = tile_h_nopad,  // Need tile_h_nopad from outer loop
                .xi = xi,
                .yi = yi,
                .x0 = xi * TILE_SIZE_X - prepadding,
                .y0 = yi * TILE_SIZE_Y - prepadding,
                .x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right,
                .y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom,
            };

            if (this->tta_mode) {
                // TTA Mode
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                preprocess_tile_tta_gpu(pipelines, in_gpu_row, in_tile_gpu, in_alpha_tile_gpu, prepadding, channels, in_format, tile, cmd, opt);

                ncnn::VkMat out_tile_gpu[8];  // Only first element used if !is_stage2
                for (int ti = 0; ti < 8; ti++) {
                    this->extract_features_sync_gap(net, opt, in_tile_gpu[ti], in_names, out_names_or_final, cache, yi, xi, ti, cmd);

                    if (ptr_stage2_options != nullptr) {
                        // Load the final output "out0" which was just saved inside extract_features_sync_gap
                        cache.load(yi, xi, ti, "out0", out_tile_gpu[ti]);
                    }
                }

                if (ptr_stage2_options != nullptr) {
                    ncnn::VkMat out_alpha_tile_gpu;
                    if (channels == 4) {
                        this->handle_alpha_channel_gpu(in_alpha_tile_gpu, out_alpha_tile_gpu, scale, cmd, opt);
                    }
                    if (scale == 4) {
                        postprocess_tile_4x_tta_gpu(pipelines, in_gpu_row, ptr_stage2_options->out_gpu_row, out_tile_gpu, out_alpha_tile_gpu, prepadding, channels, ptr_stage2_options->out_format, scale, tile, cmd, opt);
                    } else {
                        postprocess_tile_tta_gpu(pipelines, ptr_stage2_options->out_gpu_row, out_tile_gpu, out_alpha_tile_gpu, prepadding, channels, ptr_stage2_options->out_format, scale, tile, cmd, opt);
                    }
                }

            } else {
                // Standard Mode
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                preprocess_tile_gpu(pipelines, in_gpu_row, in_tile_gpu, in_alpha_tile_gpu, prepadding, channels, in_format, tile, cmd, opt);

                this->extract_features_sync_gap(net, opt, in_tile_gpu, in_names, out_names_or_final, cache, yi, xi, 0, cmd);

                if (ptr_stage2_options != nullptr) {
                    ncnn::VkMat out_tile_gpu;  // Only used if is_stage2

                    // Load the final output "out0" which was just saved inside extract_features_sync_gap
                    cache.load(yi, xi, 0, "out0", out_tile_gpu);

                    ncnn::VkMat out_alpha_tile_gpu;
                    if (channels == 4) {
                        this->handle_alpha_channel_gpu(in_alpha_tile_gpu, out_alpha_tile_gpu, scale, cmd, opt);
                    }
                    if (scale == 4) {
                        postprocess_tile_4x_gpu(pipelines, in_gpu_row, ptr_stage2_options->out_gpu_row, out_tile_gpu, out_alpha_tile_gpu, prepadding, channels, ptr_stage2_options->out_format, scale, tile, cmd, opt);
                    } else {
                        postprocess_tile_gpu(pipelines, ptr_stage2_options->out_gpu_row, out_tile_gpu, out_alpha_tile_gpu, prepadding, channels, ptr_stage2_options->out_format, scale, tile, cmd, opt);
                    }
                }
            }

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }  // End xi loop

        // Stage 2 needs download step after processing all xi for a yi
        if (ptr_stage2_options != nullptr) {
            // Download output tile
            const auto storage_mode = get_storage_mode(opt);
            unsigned char* out_data = static_cast<unsigned char*>(ptr_stage2_options->out.data);
            ncnn::Mat out_tile_cpu;

            if (storage_mode == StorageMode::FP16_INT8) {
                out_tile_cpu = ncnn::Mat(ptr_stage2_options->out_gpu_row.w, ptr_stage2_options->out_gpu_row.h, out_data + yi * scale * TILE_SIZE_Y * w * scale * channels, (size_t)channels, 1);
            }

            cmd.record_clone(ptr_stage2_options->out_gpu_row, out_tile_cpu, opt);
            cmd.submit_and_wait();

            // Copy to output image if not directly written
            if (storage_mode != StorageMode::FP16_INT8) {
                const auto out_pixel_format = ptr_stage2_options->out_format == ColorFormat::RGB ? (channels == 3 ? ncnn::Mat::PIXEL_RGB : ncnn::Mat::PIXEL_RGBA) : (channels == 3 ? ncnn::Mat::PIXEL_RGB2BGR : ncnn::Mat::PIXEL_RGBA2BGRA);
                out_tile_cpu.to_pixels(out_data + yi * scale * TILE_SIZE_Y * w * scale * channels, out_pixel_format);
            }
        } else {
            // Stage 0 just needs to ensure commands are finished for this yi row
            cmd.submit_and_wait();
        }

    }  // End process_tile_stage0_or_2

   public:
    RealCUGANSyncGapGPU(ncnn::VulkanDevice* vkdev, const ncnn::Net& net, const SuperResolutionPipelines& pipelines, const ncnn::Option& opt, std::shared_ptr<spdlog::logger> logger_error, bool tta_mode, const ncnn::Mat& in, ColorFormat in_format, const ProcessConfig& config, std::function<int(const ncnn::VkMat& in_alpha_tile, ncnn::VkMat& out_alpha_tile, int scale, ncnn::VkCompute& cmd, const ncnn::Option& opt)> handle_alpha_channel_gpu)
        : vkdev(vkdev), net(net), pipelines(pipelines), opt(opt), logger_error(logger_error), config(config), prepadding(get_prepadding_for_scale(config.scale)), in(in), in_format(in_format), tta_mode(tta_mode), handle_alpha_channel_gpu(handle_alpha_channel_gpu) {}

    // --- Standard SE Methods ---

    int stage0(int in_begin, int in_end, int out_begin, int out_end) {
        const int w = in.w;
        const int h = in.h;
        const int channels = in.elempack;
        const int scale = config.scale;
        const int prepadding = this->prepadding;
        const int tile_size = config.tile_size;

        const int TILE_SIZE_X = tile_size;
        const int TILE_SIZE_Y = tile_size;
        const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        const unsigned char* in_data = static_cast<const unsigned char*>(in.data);

        std::vector<std::string> in_names = RealCUGANSyncGapGPU::generate_feature_names(in_begin, in_end);
        std::vector<std::string> out_names = RealCUGANSyncGapGPU::generate_feature_names(out_begin, out_end);

        const auto storage_mode = get_storage_mode(opt);
        const auto in_pixel_format = in_format == ColorFormat::RGB ? (channels == 3 ? ncnn::Mat::PIXEL_RGB : ncnn::Mat::PIXEL_RGBA) : (channels == 3 ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGRA2RGBA);

        for (int yi = 0; yi < ytiles; yi++) {
            const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;
            int prepadding_bottom = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
            }

            const int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
            const int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

            ncnn::Mat in_cpu_row;
            if (storage_mode == StorageMode::FP16_INT8) {
                in_cpu_row = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), const_cast<unsigned char*>(in_data) + in_tile_y0 * w * channels, (size_t)channels, 1);
            } else {
                in_cpu_row = ncnn::Mat::from_pixels(in_data + in_tile_y0 * w * channels, in_pixel_format, w, (in_tile_y1 - in_tile_y0));
            }

            ncnn::VkCompute cmd_upload(vkdev);
            ncnn::VkMat in_gpu_row;
            cmd_upload.record_clone(in_cpu_row, in_gpu_row, opt);
            cmd_upload.submit_and_wait();  // Submit upload for the row

            // Process tiles in the row
            this->process_tile_stage0_or_2(yi, in_gpu_row, in_names, out_names, std::nullopt);  // Stage 0, dummy args for stage2 parts

        }  // End yi loop

        return 0;
    }

    int sync_gap(int gap_begin, int gap_end) {
        const int w = in.w;
        const int h = in.h;
        const int tile_size = this->config.tile_size;

        const int TILE_SIZE_X = tile_size;
        const int TILE_SIZE_Y = tile_size;
        const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
        const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        const int t_count = this->tta_mode ? 8 : 1;

        std::vector<std::string> names = generate_feature_names(gap_begin, gap_end);
        if (names.empty()) {
            return 0;  // Nothing to sync
        }

        std::vector<std::vector<ncnn::VkMat>> feats_gpu(names.size());
        int tile_count = 0;

        // 1. Load all feature maps from cache
        for (int yi = 0; yi < ytiles; yi++) {
            for (int xi = 0; xi < xtiles; xi++) {
                for (size_t i = 0; i < names.size(); i++) {
                    for (int ti = 0; ti < t_count; ti++) {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);
                        if (!feat.empty()) {  // Check if load was successful
                            feats_gpu[i].push_back(feat);
                        }
                    }
                }
                if (!names.empty()) {
                    tile_count++;  // Count valid tiles processed
                }
            }
        }
        const int total_tiles_to_average = tile_count * t_count;
        if (total_tiles_to_average == 0 || feats_gpu.empty() || feats_gpu[0].empty()) {
            this->logger_error->warn("[{}] No features found in cache to sync for gaps {} to {}.", __func__, gap_begin, gap_end);
            return 0;  // Nothing to sync
        }

        ncnn::VkCompute cmd(vkdev);

        // 2. Download features to CPU
        std::vector<std::vector<ncnn::Mat>> feats_cpu(names.size());
        for (size_t i = 0; i < names.size(); i++) {
            if (feats_gpu[i].empty()) {
                continue;  // Skip if no features for this name
            }
            feats_cpu[i].resize(feats_gpu[i].size());
            for (size_t j = 0; j < feats_gpu[i].size(); j++) {
                if (!feats_gpu[i][j].empty()) {
                    cmd.record_download(feats_gpu[i][j], feats_cpu[i][j], opt);
                }
            }
        }
        cmd.submit_and_wait();
        cmd.reset();

        // 3. Calculate global average on CPU
        std::vector<ncnn::VkMat> avgfeats_gpu(names.size());
        for (size_t i = 0; i < names.size(); i++) {
            if (feats_cpu[i].empty())
                continue;  // Skip if no features

            // Ensure all mats are valid before proceeding
            size_t valid_mats = 0;
            for (const auto& mat : feats_cpu[i]) {
                if (!mat.empty())
                    valid_mats++;
            }
            if (valid_mats == 0) {
                continue;  // Skip if no valid mats downloaded
            }

            // Pre-process CPU features (FP16->FP32, Unpack)
            for (size_t j = 0; j < feats_cpu[i].size(); j++) {
                if (feats_cpu[i][j].empty()) {
                    continue;
                }
                if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && feats_cpu[i][j].elembits() == 16) {
                    ncnn::Mat feat_fp32;
                    ncnn::cast_float16_to_float32(feats_cpu[i][j], feat_fp32, opt);
                    feats_cpu[i][j] = feat_fp32;
                }
                if (opt.use_packing_layout && feats_cpu[i][j].elempack != 1) {
                    ncnn::Mat feat_cpu_unpacked;
                    ncnn::convert_packing(feats_cpu[i][j], feat_cpu_unpacked, 1, opt);
                    feats_cpu[i][j] = feat_cpu_unpacked;
                }
            }

            // Find the first valid mat to create the average mat
            ncnn::Mat first_valid_mat;
            for (const auto& mat : feats_cpu[i]) {
                if (!mat.empty()) {
                    first_valid_mat = mat;
                    break;
                }
            }
            if (first_valid_mat.empty()) {
                continue;  // Should not happen if valid_mats > 0
            }

            ncnn::Mat avgfeat;
            avgfeat.create_like(first_valid_mat);  // Use a valid mat for template
            avgfeat.fill(0.f);
            int len = avgfeat.total();

            // Sum valid features
            for (size_t j = 0; j < feats_cpu[i].size(); j++) {
                if (feats_cpu[i][j].empty() || feats_cpu[i][j].total() != len) {
                    continue;  // Skip empty or mismatched
                }
                const float* f = feats_cpu[i][j];
                float* avg = avgfeat;
                for (int k = 0; k < len; k++) {
                    avg[k] += f[k];
                }
            }

            // Divide by the number of valid mats averaged
            float* avg = avgfeat;
            for (int k = 0; k < len; k++) {
                if (valid_mats > 0) {  // Avoid division by zero
                    avg[k] /= static_cast<float>(valid_mats);
                }
            }

            // 4. Upload average feature to GPU
            cmd.record_upload(avgfeat, avgfeats_gpu[i], opt);
        }
        cmd.submit_and_wait();
        cmd.reset();

        // 5. Save average feature back to cache for all tiles
        for (int yi = 0; yi < ytiles; yi++) {
            for (int xi = 0; xi < xtiles; xi++) {
                for (size_t i = 0; i < names.size(); i++) {
                    if (avgfeats_gpu[i].empty()) {
                        continue;  // Skip if average wasn't calculated
                    }
                    for (int ti = 0; ti < t_count; ti++) {
                        cache.save(yi, xi, ti, names[i], avgfeats_gpu[i]);
                    }
                }
            }
        }

        return 0;
    }

    int stage2(int in_begin, int in_end, ncnn::Mat& out, ColorFormat out_format) {
        const int w = in.w;
        const int h = in.h;
        const int channels = in.elempack;
        const int scale = config.scale;
        const int prepadding = this->prepadding;
        const int tile_size = config.tile_size;

        const int TILE_SIZE_X = tile_size;
        const int TILE_SIZE_Y = tile_size;
        const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
        const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;  // Need xtiles for output allocation

        const unsigned char* in_data = static_cast<const unsigned char*>(in.data);

        std::vector<std::string> in_names = RealCUGANSyncGapGPU::generate_feature_names(in_begin, in_end);
        std::vector<std::string> out_names = {"out0"};  // Final output name

        const auto storage_mode = get_storage_mode(opt);
        const auto in_pixel_format = in_format == ColorFormat::RGB ? (channels == 3 ? ncnn::Mat::PIXEL_RGB : ncnn::Mat::PIXEL_RGBA) : (channels == 3 ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGRA2RGBA);

        for (int yi = 0; yi < ytiles; yi++) {
            const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;
            int prepadding_bottom = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
            }

            const int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
            const int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

            ncnn::Mat in_cpu_row;
            if (storage_mode == StorageMode::FP16_INT8) {
                in_cpu_row = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), const_cast<unsigned char*>(in_data) + in_tile_y0 * w * channels, (size_t)channels, 1);
            } else {
                in_cpu_row = ncnn::Mat::from_pixels(in_data + in_tile_y0 * w * channels, in_pixel_format, w, (in_tile_y1 - in_tile_y0));
            }

            ncnn::VkCompute cmd_upload(vkdev);
            ncnn::VkMat in_gpu_row;
            cmd_upload.record_clone(in_cpu_row, in_gpu_row, opt);
            cmd_upload.submit_and_wait();  // Submit upload for the row

            // Create output GPU mat for the row
            const int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
            const int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);
            ncnn::VkMat out_gpu_row;
            if (storage_mode == StorageMode::FP16_INT8) {
                out_gpu_row.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels, 1, opt.blob_vkallocator);
            } else {
                out_gpu_row.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)4u, 1, opt.blob_vkallocator);
            }

            // Process tiles in the row and download
            this->process_tile_stage0_or_2(yi, in_gpu_row, in_names, out_names, std::make_optional(Stage2Options{
                                                                                    .out_gpu_row = out_gpu_row,
                                                                                    .out = out,
                                                                                    .out_format = out_format,
                                                                                }));

        }  // End yi loop
        return 0;
    }

    // --- Very Rough SE Methods ---

    int very_rough_stage0(int in_begin, int in_end, int out_begin, int out_end) {
        // Similar to stage0 but with smaller tile_size and different loop steps
        const int w = in.w;
        const int h = in.h;
        const int channels = in.elempack;
        const int scale = config.scale;
        const int prepadding = this->prepadding;
        const int tile_size = RealCUGANSyncGapGPU::VERY_ROUGH_TILE_SIZE;  // Fixed for very_rough

        const int TILE_SIZE_X = tile_size;
        const int TILE_SIZE_Y = tile_size;
        const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
        const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        const unsigned char* in_data = static_cast<const unsigned char*>(in.data);

        std::vector<std::string> in_names = RealCUGANSyncGapGPU::generate_feature_names(in_begin, in_end);
        std::vector<std::string> out_names = RealCUGANSyncGapGPU::generate_feature_names(out_begin, out_end);

        const auto storage_mode = get_storage_mode(opt);
        const auto in_pixel_format = in_format == ColorFormat::RGB ? (channels == 3 ? ncnn::Mat::PIXEL_RGB : ncnn::Mat::PIXEL_RGBA) : (channels == 3 ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGRA2RGBA);
        const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

        for (int yi = 0; yi + 2 < ytiles; yi += 3) {  // Step by 3
            const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;
            int prepadding_bottom = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
            }

            const int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
            const int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);  // Note: only processes yi row's input

            ncnn::Mat in_cpu_row;
            if (storage_mode == StorageMode::FP16_INT8) {
                in_cpu_row = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), const_cast<unsigned char*>(in_data) + in_tile_y0 * w * channels, (size_t)channels, 1);
            } else {
                in_cpu_row = ncnn::Mat::from_pixels(in_data + in_tile_y0 * w * channels, in_pixel_format, w, (in_tile_y1 - in_tile_y0));
            }

            ncnn::VkCompute cmd_upload(vkdev);
            ncnn::VkMat in_gpu_row;
            cmd_upload.record_clone(in_cpu_row, in_gpu_row, opt);
            cmd_upload.submit_and_wait();

            ncnn::VkCompute cmd_process(vkdev);
            // Inner loop for processing xi with step 3
            for (int xi = 0; xi + 2 < xtiles; xi += 3) {  // Step by 3
                const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;
                int prepadding_right = prepadding;
                if (scale == 1 || scale == 3) {
                    prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
                }
                if (scale == 2 || scale == 4) {
                    prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
                }

                const Tile tile{
                    .elemsize = in_out_tile_elemsize,
                    .size_x = TILE_SIZE_X,
                    .size_y = TILE_SIZE_Y,
                    .w_nopad = tile_w_nopad,
                    .h_nopad = tile_h_nopad,
                    .xi = xi,
                    .yi = yi,
                    .x0 = xi * TILE_SIZE_X - prepadding,
                    .y0 = yi * TILE_SIZE_Y - prepadding,
                    .x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right,
                    .y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom,  // Using yi's bottom padding
                };

                // Process only the (yi, xi) tile
                if (this->tta_mode) {
                    ncnn::VkMat in_tile_gpu[8];
                    ncnn::VkMat in_alpha_tile_gpu;
                    preprocess_tile_tta_gpu(pipelines, in_gpu_row, in_tile_gpu, in_alpha_tile_gpu, prepadding, channels, in_format, tile, cmd_process, opt);
                    for (int ti = 0; ti < 8; ti++) {
                        this->extract_features_sync_gap(net, opt, in_tile_gpu[ti], in_names, out_names, cache, yi, xi, ti, cmd_process);
                    }
                } else {
                    ncnn::VkMat in_tile_gpu;
                    ncnn::VkMat in_alpha_tile_gpu;
                    preprocess_tile_gpu(pipelines, in_gpu_row, in_tile_gpu, in_alpha_tile_gpu, prepadding, channels, in_format, tile, cmd_process, opt);
                    this->extract_features_sync_gap(net, opt, in_tile_gpu, in_names, out_names, cache, yi, xi, 0, cmd_process);
                }

                if (xtiles > 1) {  // Submit per processed tile in very rough
                    cmd_process.submit_and_wait();
                    cmd_process.reset();
                }

            }  // End xi loop (step 3)
            cmd_process.submit_and_wait();  // Final submit for the row if needed

        }  // End yi loop (step 3)

        return 0;
    }

    int very_rough_sync_gap(int gap_begin, int gap_end) {
        // Similar to sync gap but loads/averages only from processed tiles (step 3)
        // and saves the average to all 9 tiles in the 3x3 block.
        const int w = in.w;
        const int h = in.h;
        const int tile_size = RealCUGANSyncGapGPU::VERY_ROUGH_TILE_SIZE;  // Fixed for very_rough

        const int TILE_SIZE_X = tile_size;
        const int TILE_SIZE_Y = tile_size;
        const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
        const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        const int t_count = this->tta_mode ? 8 : 1;

        std::vector<std::string> names = RealCUGANSyncGapGPU::generate_feature_names(gap_begin, gap_end);
        if (names.empty()) {
            return 0;  // Nothing to sync
        }

        std::vector<std::vector<ncnn::VkMat>> feats_gpu(names.size());
        int processed_tile_count = 0;  // Count only tiles processed in very_rough_stage0

        // 1. Load features only from processed tiles
        for (int yi = 0; yi + 2 < ytiles; yi += 3) {      // Step 3
            for (int xi = 0; xi + 2 < xtiles; xi += 3) {  // Step 3
                for (size_t i = 0; i < names.size(); i++) {
                    for (int ti = 0; ti < t_count; ti++) {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);  // Load only from (yi, xi)
                        if (!feat.empty()) {
                            feats_gpu[i].push_back(feat);
                        }
                    }
                }
                if (!names.empty()) {
                    processed_tile_count++;  // Count valid tiles processed
                }
            }
        }

        const int total_tiles_to_average = processed_tile_count * t_count;
        if (total_tiles_to_average == 0 || feats_gpu.empty() || feats_gpu[0].empty()) {
            this->logger_error->warn("[{}] No features found in cache to sync for gaps {} to {}.", __func__, gap_begin, gap_end);
            return 0;  // Nothing to sync
        }

        ncnn::VkCompute cmd(vkdev);

        // 2. Download features to CPU (Same as sync_gap)
        std::vector<std::vector<ncnn::Mat>> feats_cpu(names.size());
        for (size_t i = 0; i < names.size(); i++) {
            if (feats_gpu[i].empty()) {
                continue;
            }
            feats_cpu[i].resize(feats_gpu[i].size());
            for (size_t j = 0; j < feats_gpu[i].size(); j++) {
                if (!feats_gpu[i][j].empty()) {
                    cmd.record_download(feats_gpu[i][j], feats_cpu[i][j], opt);
                }
            }
        }
        cmd.submit_and_wait();
        cmd.reset();

        // 3. Calculate global average on CPU (Same as sync_gap)
        std::vector<ncnn::VkMat> avgfeats_gpu(names.size());
        for (size_t i = 0; i < names.size(); i++) {
            if (feats_cpu[i].empty()) {
                continue;  // Skip if no features
            }

            // Ensure all mats are valid before proceeding
            size_t valid_mats = 0;
            for (const auto& mat : feats_cpu[i]) {
                if (!mat.empty()) {
                    valid_mats++;
                }
            }
            if (valid_mats == 0) {
                continue;  // Skip if no valid mats downloaded
            }

            // Pre-process CPU features (FP16->FP32, Unpack)
            for (size_t j = 0; j < feats_cpu[i].size(); j++) {
                if (feats_cpu[i][j].empty()) {
                    continue;
                }
                if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && feats_cpu[i][j].elembits() == 16) {
                    ncnn::Mat feat_fp32;
                    ncnn::cast_float16_to_float32(feats_cpu[i][j], feat_fp32, opt);
                    feats_cpu[i][j] = feat_fp32;
                }
                if (opt.use_packing_layout && feats_cpu[i][j].elempack != 1) {
                    ncnn::Mat feat_cpu_unpacked;
                    ncnn::convert_packing(feats_cpu[i][j], feat_cpu_unpacked, 1, opt);
                    feats_cpu[i][j] = feat_cpu_unpacked;
                }
            }

            // Find the first valid mat to create the average mat
            ncnn::Mat first_valid_mat;
            for (const auto& mat : feats_cpu[i]) {
                if (!mat.empty()) {
                    first_valid_mat = mat;
                    break;
                }
            }
            if (first_valid_mat.empty()) {
                continue;  // Should not happen if valid_mats > 0
            }

            ncnn::Mat avgfeat;
            avgfeat.create_like(first_valid_mat);  // Use a valid mat for template
            avgfeat.fill(0.f);
            int len = avgfeat.total();

            // Sum valid features
            for (size_t j = 0; j < feats_cpu[i].size(); j++) {
                if (feats_cpu[i][j].empty() || feats_cpu[i][j].total() != len) {
                    continue;  // Skip empty or mismatched
                }
                const float* f = feats_cpu[i][j];
                float* avg = avgfeat;
                for (int k = 0; k < len; k++) {
                    avg[k] += f[k];
                }
            }

            // Divide by the number of valid mats averaged
            float* avg = avgfeat;
            for (int k = 0; k < len; k++) {
                if (valid_mats > 0) {  // Avoid division by zero
                    avg[k] /= static_cast<float>(valid_mats);
                }
            }

            // 4. Upload average feature to GPU
            cmd.record_upload(avgfeat, avgfeats_gpu[i], opt);
        }
        cmd.submit_and_wait();
        cmd.reset();

        // 5. Save average feature back to cache for *all* 9 tiles in each 3x3 block
        for (int yi = 0; yi + 2 < ytiles; yi += 3) {
            for (int xi = 0; xi + 2 < xtiles; xi += 3) {
                for (size_t i = 0; i < names.size(); i++) {
                    if (avgfeats_gpu[i].empty()) {
                        continue;  // Skip if average wasn't calculated
                    }
                    for (int ti = 0; ti < t_count; ti++) {
                        for (int dy = 0; dy < 3; dy++) {
                            for (int dx = 0; dx < 3; dx++) {
                                cache.save(yi + dy, xi + dx, ti, names[i], avgfeats_gpu[i]);
                            }
                        }
                    }
                }
            }
        }

        return 0;
    }
};

RealCUGAN::RealCUGAN(const SuperResolutionEngineConfig& config)
    : SuperResolutionEngine(config) {
    // Validate config against engine info
    const auto& info = get_engine_info();
    if (!info.is_compatible_config(config)) {
        this->config.logger_error->warn("[{}] Configuration may not be fully compatible with RealCUGAN engine", __func__);
    }
}

const SuperResolutionEngineInfo& RealCUGAN::engine_info() const {
    return RealCUGAN::get_engine_info();
}

int RealCUGAN::get_default_tile_size() const {
    // Determine tile size based on available GPU memory
    const uint32_t heap_budget = this->vkdev != nullptr ? this->vkdev->get_heap_budget() : 2000;
    if (heap_budget > 1900) {
        return 400;
    }
    if (heap_budget > 550) {
        return 200;
    }
    if (heap_budget > 190) {
        return 100;
    }
    return 32;
}

void RealCUGAN::prepare_net_options(ncnn::Option& options) const {
    SuperResolutionEngine::prepare_net_options(options);

    // Set network options
    options.use_fp16_packed = true;
    options.use_fp16_storage = this->vkdev != nullptr;
    options.use_fp16_arithmetic = false;
    options.use_int8_storage = true;
    options.use_int8_arithmetic = false;
}

std::u8string RealCUGAN::get_model_path(const std::u8string& model_type, int scale, int noise) const {
    constexpr const char* NOISE_SUFFIXES_DEFAULT[] = {
        "conservative",
        "no-denoise",
        "denoise1x",
        "denoise2x",
        "denoise3x",
    };

    constexpr const char* NOISE_SUFFIXES_NOSE[] = {
        "conservative-no-denoise",
        "no-denoise",
    };

    if (scale < 2 || scale > 4) {
        this->config.logger_error->warn("[{}] Scale {} is not officially supported by RealCUGAN", __func__, scale);
        return u8"";
    }

    std::u8string model_path;

    std::u8string current_type = model_type;
    if (current_type == u8"models-nose" && (scale != 2)) {
        this->config.logger_error->warn("[{}] models-nose does not support scale {}, falling back to models-se", __func__, scale);
        current_type = u8"models-se";
    } else if (current_type == u8"models-pro" && (scale != 2 && scale != 3)) {
        this->config.logger_error->warn("[{}] models-pro does not support scale {}, falling back to models-se", __func__, scale);
        current_type = u8"models-se";
    }

    if (current_type == u8"models-se") {
        model_path = ascii_to_utf8(std::format("realcugan-se-up{}x-{}", scale, NOISE_SUFFIXES_DEFAULT[std::clamp(noise + 1, 0, 4)]));
    } else if (current_type == u8"models-nose") {
        model_path = ascii_to_utf8(std::format("realcugan-nose-up{}x-{}", scale, NOISE_SUFFIXES_NOSE[std::clamp(noise + 1, 0, 1)]));
    } else if (current_type == u8"models-pro") {
        model_path = ascii_to_utf8(std::format("realcugan-pro-up{}x-{}", scale, NOISE_SUFFIXES_DEFAULT[std::clamp(noise + 1, 0, 4)]));
    }

    return model_path;
}

std::shared_ptr<ncnn::Net> RealCUGAN::create_net(int scale, const NetCache& net_cache) const {
    // Get engine info
    const auto& info = get_engine_info();

    // Check if scale is supported
    if (!info.supports_scale(scale)) {
        this->config.logger_error->warn("[{}] Scale {} is not officially supported by RealCUGAN", __func__, scale);
    }

    // Use default model if none specified
    std::u8string model_name = this->config.model.empty() ? info.default_model : this->config.model;

    // Check if model is supported
    if (!info.supports_model(model_name)) {
        this->config.logger_error->warn("[{}] Model '{}' is not officially supported by RealCUGAN", __func__, utf8_to_ascii(model_name));
    }

    // Get model path based on model type, scale and noise
    std::u8string model_path = get_model_path(model_name, scale, this->config.noise);

    auto net = this->create_net_base();
    this->net_load_model_and_param(*net, this->config.model_dir / (model_path + u8".param"));

    return net;
}

std::shared_ptr<SuperResolutionPipelines> RealCUGAN::create_pipelines(int scale, const PipelineCache&) const {
    if (this->vkdev == nullptr) {
        return nullptr;
    }

    auto sp = std::make_shared<SuperResolutionPipelines>(this->vkdev);
    auto& pipelines = *sp;

    // Get the storage mode from the current network options
    const auto ptr_net = this->net_cache.get_net(scale);
    if (ptr_net == nullptr) {
        return nullptr;
    }
    const auto storage_mode = get_storage_mode(ptr_net->opt);

    // Create specializations for RGB and BGR
    std::vector<ncnn::vk_specialization_type> specializations_rgb(1);
    specializations_rgb[0].i = 0;  // RGB mode

    std::vector<ncnn::vk_specialization_type> specializations_bgr(1);
    specializations_bgr[0].i = 1;  // BGR mode

    // Create preprocessing and postprocessing pipelines
    pipelines.preprocess_rgb.set_optimal_local_size_xyz(8, 8, 3);
    pipelines.preprocess_bgr.set_optimal_local_size_xyz(8, 8, 3);
    pipelines.postprocess_rgb.set_optimal_local_size_xyz(8, 8, 3);
    pipelines.postprocess_bgr.set_optimal_local_size_xyz(8, 8, 3);

    if (!this->config.tta_mode) {
        // Standard preprocessing and postprocessing
        switch (storage_mode) {
            case StorageMode::FP16_INT8:
                pipelines.preprocess_rgb.create(realcugan_preproc_int8s_spv_data, sizeof(realcugan_preproc_int8s_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realcugan_preproc_int8s_spv_data, sizeof(realcugan_preproc_int8s_spv_data), specializations_bgr);

                if (scale == 4) {
                    pipelines.postprocess_rgb.create(realcugan_4x_postproc_int8s_spv_data, sizeof(realcugan_4x_postproc_int8s_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_4x_postproc_int8s_spv_data, sizeof(realcugan_4x_postproc_int8s_spv_data), specializations_bgr);
                } else {
                    pipelines.postprocess_rgb.create(realcugan_postproc_int8s_spv_data, sizeof(realcugan_postproc_int8s_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_postproc_int8s_spv_data, sizeof(realcugan_postproc_int8s_spv_data), specializations_bgr);
                }
                break;

            case StorageMode::FP16:
                pipelines.preprocess_rgb.create(realcugan_preproc_fp16s_spv_data, sizeof(realcugan_preproc_fp16s_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realcugan_preproc_fp16s_spv_data, sizeof(realcugan_preproc_fp16s_spv_data), specializations_bgr);

                if (scale == 4) {
                    pipelines.postprocess_rgb.create(realcugan_4x_postproc_fp16s_spv_data, sizeof(realcugan_4x_postproc_fp16s_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_4x_postproc_fp16s_spv_data, sizeof(realcugan_4x_postproc_fp16s_spv_data), specializations_bgr);
                } else {
                    pipelines.postprocess_rgb.create(realcugan_postproc_fp16s_spv_data, sizeof(realcugan_postproc_fp16s_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_postproc_fp16s_spv_data, sizeof(realcugan_postproc_fp16s_spv_data), specializations_bgr);
                }
                break;

            default:
                pipelines.preprocess_rgb.create(realcugan_preproc_spv_data, sizeof(realcugan_preproc_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realcugan_preproc_spv_data, sizeof(realcugan_preproc_spv_data), specializations_bgr);

                if (scale == 4) {
                    pipelines.postprocess_rgb.create(realcugan_4x_postproc_spv_data, sizeof(realcugan_4x_postproc_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_4x_postproc_spv_data, sizeof(realcugan_4x_postproc_spv_data), specializations_bgr);
                } else {
                    pipelines.postprocess_rgb.create(realcugan_postproc_spv_data, sizeof(realcugan_postproc_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_postproc_spv_data, sizeof(realcugan_postproc_spv_data), specializations_bgr);
                }
                break;
        }
    } else {
        // TTA preprocessing and postprocessing
        switch (storage_mode) {
            case StorageMode::FP16_INT8:
                pipelines.preprocess_rgb.create(realcugan_preproc_tta_int8s_spv_data, sizeof(realcugan_preproc_tta_int8s_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realcugan_preproc_tta_int8s_spv_data, sizeof(realcugan_preproc_tta_int8s_spv_data), specializations_bgr);

                if (scale == 4) {
                    pipelines.postprocess_rgb.create(realcugan_4x_postproc_tta_int8s_spv_data, sizeof(realcugan_4x_postproc_tta_int8s_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_4x_postproc_tta_int8s_spv_data, sizeof(realcugan_4x_postproc_tta_int8s_spv_data), specializations_bgr);
                } else {
                    pipelines.postprocess_rgb.create(realcugan_postproc_tta_int8s_spv_data, sizeof(realcugan_postproc_tta_int8s_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_postproc_tta_int8s_spv_data, sizeof(realcugan_postproc_tta_int8s_spv_data), specializations_bgr);
                }
                break;

            case StorageMode::FP16:
                pipelines.preprocess_rgb.create(realcugan_preproc_tta_fp16s_spv_data, sizeof(realcugan_preproc_tta_fp16s_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realcugan_preproc_tta_fp16s_spv_data, sizeof(realcugan_preproc_tta_fp16s_spv_data), specializations_bgr);

                if (scale == 4) {
                    pipelines.postprocess_rgb.create(realcugan_4x_postproc_tta_fp16s_spv_data, sizeof(realcugan_4x_postproc_tta_fp16s_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_4x_postproc_tta_fp16s_spv_data, sizeof(realcugan_4x_postproc_tta_fp16s_spv_data), specializations_bgr);
                } else {
                    pipelines.postprocess_rgb.create(realcugan_postproc_tta_fp16s_spv_data, sizeof(realcugan_postproc_tta_fp16s_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_postproc_tta_fp16s_spv_data, sizeof(realcugan_postproc_tta_fp16s_spv_data), specializations_bgr);
                }
                break;

            default:
                pipelines.preprocess_rgb.create(realcugan_preproc_tta_spv_data, sizeof(realcugan_preproc_tta_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realcugan_preproc_tta_spv_data, sizeof(realcugan_preproc_tta_spv_data), specializations_bgr);

                if (scale == 4) {
                    pipelines.postprocess_rgb.create(realcugan_4x_postproc_tta_spv_data, sizeof(realcugan_4x_postproc_tta_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_4x_postproc_tta_spv_data, sizeof(realcugan_4x_postproc_tta_spv_data), specializations_bgr);
                } else {
                    pipelines.postprocess_rgb.create(realcugan_postproc_tta_spv_data, sizeof(realcugan_postproc_tta_spv_data), specializations_rgb);
                    pipelines.postprocess_bgr.create(realcugan_postproc_tta_spv_data, sizeof(realcugan_postproc_tta_spv_data), specializations_bgr);
                }
                break;
        }
    }

    return sp;
}

int RealCUGAN::process_gpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const {
    // Check if sync gap is enabled and use appropriate processing method
    if (this->config.sync_gap > 0 && config.tile_size < std::max(in.w, in.h)) {
        if (this->config.sync_gap == 1) {
            return process_gpu_se(in, in_format, out, out_format, config);
        } else if (this->config.sync_gap == 2) {
            return process_gpu_se_rough(in, in_format, out, out_format, config);
        } else if (this->config.sync_gap == 3) {
            return process_gpu_se_very_rough(in, in_format, out, out_format, config);
        }
    }
    return process_gpu_nose(in, in_format, out, out_format, config);
}

int RealCUGAN::process_gpu_nose(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const {
    const unsigned char* in_data = static_cast<const unsigned char*>(in.data);
    const int w = in.w;
    const int h = in.h;
    const int channels = in.elempack;

    unsigned char* out_data = static_cast<unsigned char*>(out.data);

    // Get parameters from config
    const int scale = config.scale;
    const int tile_size = config.tile_size;
    const int prepadding = get_prepadding_for_scale(scale);

    if (w < 1 || h < 1 || (channels != 3 && channels != 4) || tile_size < 1 || prepadding < 0 || scale < 1) {
        this->config.logger_error->error("[{}] Invalid input parameters: w={}, h={}, channels={}, tile_size={}, prepadding={}, scale={}", __func__, w, h, channels, tile_size, prepadding, scale);
        return -1;
    }

    // Check if output dimensions are correct
    if (out.w != w * scale || out.h != h * scale || out.elempack != channels) {
        this->config.logger_error->error("[{}] Output dimensions do not match input dimensions: expected (w={}, h={}, channels={}), got (w={}, h={}, channels={})", __func__, w * scale, h * scale, channels, out.w, out.h, out.elempack);
        return -1;
    }

    const auto in_pixel_format = in_format == ColorFormat::RGB ? (channels == 3 ? ncnn::Mat::PIXEL_RGB : ncnn::Mat::PIXEL_RGBA) : (channels == 3 ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGRA2RGBA);
    const auto out_pixel_format = out_format == ColorFormat::RGB ? (channels == 3 ? ncnn::Mat::PIXEL_RGB : ncnn::Mat::PIXEL_RGBA) : (channels == 3 ? ncnn::Mat::PIXEL_RGB2BGR : ncnn::Mat::PIXEL_RGBA2BGRA);

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
    const auto& pipelines = *ptr_pipelines;

    // Create allocators
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    if (blob_vkallocator == nullptr) {
        this->config.logger_error->error("[{}] Failed to acquire blob allocator", __func__);
        return -1;
    }
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    if (staging_vkallocator == nullptr) {
        this->config.logger_error->error("[{}] Failed to acquire staging allocator", __func__);
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        return -1;
    }

    ncnn::Option opt = ptr_net->opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    const auto storage_mode = get_storage_mode(opt);

    // Calculate tiles
    const int TILE_SIZE_X = tile_size;
    const int TILE_SIZE_Y = tile_size;

    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    // Process tiles
    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        const int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        const int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        ncnn::Mat in_cpu_row;
        if (storage_mode == StorageMode::FP16_INT8) {
            // Use source pixel data directly for FP16_INT8 storage mode
            in_cpu_row = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), const_cast<unsigned char*>(in_data) + in_tile_y0 * w * channels, (size_t)channels, 1);
        } else {
            // Copy pixel data to ncnn::Mat for other storage modes
            in_cpu_row = ncnn::Mat::from_pixels(in_data + in_tile_y0 * w * channels, in_pixel_format, w, (in_tile_y1 - in_tile_y0));
        }

        ncnn::VkCompute cmd(vkdev);

        // Upload input tile
        ncnn::VkMat in_gpu_row;

        cmd.record_clone(in_cpu_row, in_gpu_row, opt);
        if (xtiles > 1) {
            cmd.submit_and_wait();
            cmd.reset();
        }

        const int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        const int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        // Create output GPU mat
        ncnn::VkMat out_gpu_row;
        if (storage_mode == StorageMode::FP16_INT8) {
            out_gpu_row.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels, 1, blob_vkallocator);
        } else {
            out_gpu_row.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)4u, 1, blob_vkallocator);
        }

        // Process each tile
        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            } else if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            // Calculate tile coordinates
            const Tile tile{
                .elemsize = in_out_tile_elemsize,

                .size_x = TILE_SIZE_X,
                .size_y = TILE_SIZE_Y,

                .w_nopad = tile_w_nopad,
                .h_nopad = tile_h_nopad,

                .xi = xi,
                .yi = yi,

                .x0 = xi * TILE_SIZE_X - prepadding,
                .y0 = yi * TILE_SIZE_Y - prepadding,
                .x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right,
                .y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom,
            };

            if (!this->config.tta_mode) {
                // Standard mode processing
                // Preprocess
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;

                // Preprocess
                preprocess_tile_gpu(pipelines, in_gpu_row, in_tile_gpu, in_alpha_tile_gpu,
                                    prepadding, channels, in_format, tile, cmd, opt);

                // Process with neural network
                ncnn::VkMat out_tile_gpu;
                extract_features(*ptr_net, opt, in_tile_gpu, out_tile_gpu, cmd);

                // Handle alpha channel
                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    this->handle_alpha_channel_gpu(in_alpha_tile_gpu, out_alpha_tile_gpu, scale, cmd, opt);
                }

                // Postprocess
                if (scale == 4) {
                    postprocess_tile_4x_gpu(pipelines, in_gpu_row, out_gpu_row, out_tile_gpu, out_alpha_tile_gpu,
                                            prepadding, channels, out_format, scale, tile, cmd, opt);
                } else {
                    postprocess_tile_gpu(pipelines, out_gpu_row, out_tile_gpu, out_alpha_tile_gpu,
                                         prepadding, channels, out_format, scale, tile, cmd, opt);
                }
            } else {
                // TTA mode processing
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;

                // Preprocess with TTA
                preprocess_tile_tta_gpu(pipelines, in_gpu_row, in_tile_gpu, in_alpha_tile_gpu,
                                        prepadding, channels, in_format, tile, cmd, opt);

                // Process with neural network
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++) {
                    extract_features(*ptr_net, opt, in_tile_gpu[ti], out_tile_gpu[ti], cmd);
                }

                // Handle alpha channel
                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    this->handle_alpha_channel_gpu(in_alpha_tile_gpu, out_alpha_tile_gpu, scale, cmd, opt);
                }

                // Postprocess with TTA
                if (scale == 4) {
                    postprocess_tile_4x_tta_gpu(pipelines, in_gpu_row, out_gpu_row, out_tile_gpu, out_alpha_tile_gpu,
                                                prepadding, channels, out_format, scale, tile, cmd, opt);
                } else {
                    postprocess_tile_tta_gpu(pipelines, out_gpu_row, out_tile_gpu, out_alpha_tile_gpu,
                                             prepadding, channels, out_format, scale, tile, cmd, opt);
                }
            }

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        // Download output tile
        ncnn::Mat out_cpu_row;
        if (storage_mode == StorageMode::FP16_INT8) {
            // Output directly to the output mat if using FP16_INT8 storage
            out_cpu_row = ncnn::Mat(out_gpu_row.w, out_gpu_row.h, out_data + yi * scale * TILE_SIZE_Y * w * scale * channels, (size_t)channels, 1);
        }

        cmd.record_clone(out_gpu_row, out_cpu_row, opt);
        cmd.submit_and_wait();

        // Copy to output image
        if (storage_mode != StorageMode::FP16_INT8) {
            // Copy output tile to the output mat
            out_cpu_row.to_pixels(out_data + yi * scale * TILE_SIZE_Y * w * scale * channels,
                                  out_pixel_format);
        }
    }

    // Reclaim allocators
    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

std::unique_ptr<RealCUGANSyncGapGPU, std::function<void(RealCUGANSyncGapGPU*)>> RealCUGAN::create_sync_gap_gpu(const ncnn::Mat& in, ColorFormat in_format, const ProcessConfig& config) const {
    // Get the network for the current scale
    const auto ptr_net = this->net_cache.get_net(config.scale);
    if (ptr_net == nullptr) {
        this->config.logger_error->error("[{}] Failed to get net for scale {}", __func__, config.scale);
        return nullptr;
    }
    const auto& net = *ptr_net;

    // Get pipelines for the current scale
    const auto ptr_pipelines = this->pipeline_cache.get_pipelines(config.scale);
    if (ptr_pipelines == nullptr) {
        this->config.logger_error->error("[{}] Failed to get pipelines for scale {}", __func__, config.scale);
        return nullptr;
    }
    const auto& pipelines = *ptr_pipelines;

    const auto vkdev = this->vkdev;

    // Create allocators
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    if (blob_vkallocator == nullptr) {
        this->config.logger_error->error("[{}] Failed to acquire blob allocator", __func__);
        return nullptr;
    }
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    if (staging_vkallocator == nullptr) {
        this->config.logger_error->error("[{}] Failed to acquire staging allocator", __func__);
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        return nullptr;
    }

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    return std::unique_ptr<RealCUGANSyncGapGPU, std::function<void(RealCUGANSyncGapGPU*)>>(
        new RealCUGANSyncGapGPU(vkdev, net, pipelines, opt, this->config.logger_error, this->config.tta_mode, in, in_format, config, std::bind(&RealCUGAN::handle_alpha_channel_gpu, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5)),
        [vkdev, blob_vkallocator, staging_vkallocator](RealCUGANSyncGapGPU* ptr) {
            delete ptr;

            // Reclaim allocators
            vkdev->reclaim_blob_allocator(blob_vkallocator);
            vkdev->reclaim_staging_allocator(staging_vkallocator);
        });
}

int RealCUGAN::process_gpu_se(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const {
    const auto sg = this->create_sync_gap_gpu(in, in_format, config);
    if (sg == nullptr) {
        this->config.logger_error->error("[{}] Failed to create GPU sync gap processor", __func__);
        return -1;
    }

    // Stage 0: Extract features (first pass)
    sg->stage0(0, 0, 0, 1);  // [] -> [gap0]

    // Sync GAP (first pass)
    sg->sync_gap(0, 1);  // [gap0]

    // Stage 0: Extract features (second pass)
    sg->stage0(0, 1, 1, 2);  // [gap0] -> [gap1]

    // Sync GAP (second pass)
    sg->sync_gap(1, 2);  // [gap1]

    // Stage 0: Extract features (third pass)
    sg->stage0(0, 2, 2, 3);  // [gap0, gap1] -> [gap2]

    // Sync GAP (third pass)
    sg->sync_gap(2, 3);  // [gap2]

    // Stage 0: Extract features (fourth pass)
    sg->stage0(0, 3, 3, 4);  // [gap0, gap1, gap2] -> [gap3]

    // Sync GAP (fourth pass)
    sg->sync_gap(3, 4);  // [gap3]

    // Stage 2: Process with features
    const auto ret = sg->stage2(0, 4, out, out_format);

    return ret;
}

int RealCUGAN::process_gpu_se_rough(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const {
    const auto sg = this->create_sync_gap_gpu(in, in_format, config);
    if (sg == nullptr) {
        this->config.logger_error->error("[{}] Failed to create GPU sync gap processor", __func__);
        return -1;
    }

    // Stage 0: Extract features
    sg->stage0(0, 0, 0, 4);  // [] -> [gap0, gap1, gap2, gap3]

    // Sync GAP
    sg->sync_gap(0, 4);  // [gap0, gap1, gap2, gap3]

    // Stage 2: Process with features
    const auto ret = sg->stage2(0, 4, out, out_format);

    return ret;
}

int RealCUGAN::process_gpu_se_very_rough(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const {
    const auto sg = this->create_sync_gap_gpu(in, in_format, config);
    if (sg == nullptr) {
        this->config.logger_error->error("[{}] Failed to create GPU sync gap processor", __func__);
        return -1;
    }

    // Very rough stage 0: Extract features
    sg->very_rough_stage0(0, 0, 0, 4);  // [] -> [gap0, gap1, gap2, gap3]

    // Very rough sync GAP
    sg->very_rough_sync_gap(0, 4);  // [gap0, gap1, gap2, gap3]

    // Stage 2: Process with features
    const auto ret = sg->stage2(0, 4, out, out_format);

    return ret;
}

namespace {

void extract_features(ncnn::Net& net, const ncnn::Option& options, const ncnn::VkMat& in_tile, ncnn::VkMat& out_tile, ncnn::VkCompute& cmd) {
    ncnn::Extractor ex = net.create_extractor();

    ex.set_blob_vkallocator(options.blob_vkallocator);
    ex.set_workspace_vkallocator(options.workspace_vkallocator);
    ex.set_staging_vkallocator(options.staging_vkallocator);

    ex.input("in0", in_tile);
    ex.extract("out0", out_tile, cmd);
}

void preprocess_tile_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, ncnn::VkMat& in_tile_gpu, ncnn::VkMat& in_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt) {
    in_tile_gpu.create(tile.x1 - tile.x0, tile.y1 - tile.y0, 3, tile.elemsize, 1, opt.blob_vkallocator);

    if (channels == 4) {
        in_alpha_tile_gpu.create(tile.w_nopad, tile.h_nopad, 1, tile.elemsize, 1, opt.blob_vkallocator);
    }

    std::vector<ncnn::VkMat> bindings(3);
    bindings[0] = in_gpu_row;
    bindings[1] = in_tile_gpu;
    bindings[2] = in_alpha_tile_gpu;

    std::vector<ncnn::vk_constant_type> constants(13);
    constants[0].i = in_gpu_row.w;
    constants[1].i = in_gpu_row.h;
    constants[2].i = in_gpu_row.cstep;
    constants[3].i = in_tile_gpu.w;
    constants[4].i = in_tile_gpu.h;
    constants[5].i = in_tile_gpu.cstep;
    constants[6].i = prepadding;
    constants[7].i = prepadding;
    constants[8].i = tile.xi * tile.size_x;
    constants[9].i = std::min(tile.yi * tile.size_y, prepadding);
    constants[10].i = channels;
    constants[11].i = in_alpha_tile_gpu.w;
    constants[12].i = in_alpha_tile_gpu.h;

    ncnn::VkMat dispatcher;
    dispatcher.w = in_tile_gpu.w;
    dispatcher.h = in_tile_gpu.h;
    dispatcher.c = channels;

    cmd.record_pipeline(format == ColorFormat::RGB ? &pipelines.preprocess_rgb : &pipelines.preprocess_bgr, bindings, constants, dispatcher);
}

void preprocess_tile_tta_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, ncnn::VkMat in_tile_gpu[8], ncnn::VkMat& in_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt) {
    in_tile_gpu[0].create(tile.x1 - tile.x0, tile.y1 - tile.y0, 3, tile.elemsize, 1, opt.blob_vkallocator);
    in_tile_gpu[1].create(tile.x1 - tile.x0, tile.y1 - tile.y0, 3, tile.elemsize, 1, opt.blob_vkallocator);
    in_tile_gpu[2].create(tile.x1 - tile.x0, tile.y1 - tile.y0, 3, tile.elemsize, 1, opt.blob_vkallocator);
    in_tile_gpu[3].create(tile.x1 - tile.x0, tile.y1 - tile.y0, 3, tile.elemsize, 1, opt.blob_vkallocator);
    in_tile_gpu[4].create(tile.y1 - tile.y0, tile.x1 - tile.x0, 3, tile.elemsize, 1, opt.blob_vkallocator);
    in_tile_gpu[5].create(tile.y1 - tile.y0, tile.x1 - tile.x0, 3, tile.elemsize, 1, opt.blob_vkallocator);
    in_tile_gpu[6].create(tile.y1 - tile.y0, tile.x1 - tile.x0, 3, tile.elemsize, 1, opt.blob_vkallocator);
    in_tile_gpu[7].create(tile.y1 - tile.y0, tile.x1 - tile.x0, 3, tile.elemsize, 1, opt.blob_vkallocator);

    if (channels == 4) {
        in_alpha_tile_gpu.create(tile.w_nopad, tile.h_nopad, 1, tile.elemsize, 1, opt.blob_vkallocator);
    }

    std::vector<ncnn::VkMat> bindings(10);
    bindings[0] = in_gpu_row;
    bindings[1] = in_tile_gpu[0];
    bindings[2] = in_tile_gpu[1];
    bindings[3] = in_tile_gpu[2];
    bindings[4] = in_tile_gpu[3];
    bindings[5] = in_tile_gpu[4];
    bindings[6] = in_tile_gpu[5];
    bindings[7] = in_tile_gpu[6];
    bindings[8] = in_tile_gpu[7];
    bindings[9] = in_alpha_tile_gpu;

    std::vector<ncnn::vk_constant_type> constants(13);
    constants[0].i = in_gpu_row.w;
    constants[1].i = in_gpu_row.h;
    constants[2].i = in_gpu_row.cstep;
    constants[3].i = in_tile_gpu[0].w;
    constants[4].i = in_tile_gpu[0].h;
    constants[5].i = in_tile_gpu[0].cstep;
    constants[6].i = prepadding;
    constants[7].i = prepadding;
    constants[8].i = tile.xi * tile.size_x;
    constants[9].i = std::min(tile.yi * tile.size_y, prepadding);
    constants[10].i = channels;
    constants[11].i = in_alpha_tile_gpu.w;
    constants[12].i = in_alpha_tile_gpu.h;

    ncnn::VkMat dispatcher;
    dispatcher.w = in_tile_gpu[0].w;
    dispatcher.h = in_tile_gpu[0].h;
    dispatcher.c = channels;

    cmd.record_pipeline(format == ColorFormat::RGB ? &pipelines.preprocess_rgb : &pipelines.preprocess_bgr, bindings, constants, dispatcher);
}

void postprocess_tile_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& out_gpu_row, ncnn::VkMat& out_tile_gpu, ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt) {
    std::vector<ncnn::VkMat> bindings(3);
    bindings[0] = out_tile_gpu;
    bindings[1] = out_alpha_tile_gpu;
    bindings[2] = out_gpu_row;

    std::vector<ncnn::vk_constant_type> constants(11);
    constants[0].i = out_tile_gpu.w;
    constants[1].i = out_tile_gpu.h;
    constants[2].i = out_tile_gpu.cstep;
    constants[3].i = out_gpu_row.w;
    constants[4].i = out_gpu_row.h;
    constants[5].i = out_gpu_row.cstep;
    constants[6].i = tile.xi * tile.size_x * scale;
    constants[7].i = std::min(tile.size_x * scale, out_gpu_row.w - tile.xi * tile.size_x * scale);
    constants[8].i = channels;
    constants[9].i = out_alpha_tile_gpu.w;
    constants[10].i = out_alpha_tile_gpu.h;

    ncnn::VkMat dispatcher;
    dispatcher.w = std::min(tile.size_x * scale, out_gpu_row.w - tile.xi * tile.size_x * scale);
    dispatcher.h = out_gpu_row.h;
    dispatcher.c = channels;

    cmd.record_pipeline(format == ColorFormat::RGB ? &pipelines.postprocess_rgb : &pipelines.postprocess_bgr, bindings, constants, dispatcher);
}

void postprocess_tile_tta_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& out_gpu_row, ncnn::VkMat out_tile_gpu[8], ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt) {
    std::vector<ncnn::VkMat> bindings(10);
    bindings[0] = out_tile_gpu[0];
    bindings[1] = out_tile_gpu[1];
    bindings[2] = out_tile_gpu[2];
    bindings[3] = out_tile_gpu[3];
    bindings[4] = out_tile_gpu[4];
    bindings[5] = out_tile_gpu[5];
    bindings[6] = out_tile_gpu[6];
    bindings[7] = out_tile_gpu[7];
    bindings[8] = out_alpha_tile_gpu;
    bindings[9] = out_gpu_row;

    std::vector<ncnn::vk_constant_type> constants(13);
    constants[0].i = out_tile_gpu[0].w;
    constants[1].i = out_tile_gpu[0].h;
    constants[2].i = out_tile_gpu[0].cstep;
    constants[3].i = out_gpu_row.w;
    constants[4].i = out_gpu_row.h;
    constants[5].i = out_gpu_row.cstep;
    constants[6].i = tile.xi * tile.size_x * scale;
    constants[7].i = std::min(tile.size_x * scale, out_gpu_row.w - tile.xi * tile.size_x * scale);
    constants[8].i = prepadding * scale;
    constants[9].i = prepadding * scale;
    constants[10].i = channels;
    constants[11].i = out_alpha_tile_gpu.w;
    constants[12].i = out_alpha_tile_gpu.h;

    ncnn::VkMat dispatcher;
    dispatcher.w = std::min(tile.size_x * scale, out_gpu_row.w - tile.xi * tile.size_x * scale);
    dispatcher.h = out_gpu_row.h;
    dispatcher.c = channels;

    cmd.record_pipeline(format == ColorFormat::RGB ? &pipelines.postprocess_rgb : &pipelines.postprocess_bgr, bindings, constants, dispatcher);
}

void postprocess_tile_4x_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, const ncnn::VkMat& out_gpu_row, ncnn::VkMat& out_tile_gpu, ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt) {
    std::vector<ncnn::VkMat> bindings(4);
    bindings[0] = in_gpu_row;
    bindings[1] = out_tile_gpu;
    bindings[2] = out_alpha_tile_gpu;
    bindings[3] = out_gpu_row;

    std::vector<ncnn::vk_constant_type> constants(16);
    constants[0].i = in_gpu_row.w;
    constants[1].i = in_gpu_row.h;
    constants[2].i = in_gpu_row.cstep;
    constants[3].i = out_tile_gpu.w;
    constants[4].i = out_tile_gpu.h;
    constants[5].i = out_tile_gpu.cstep;
    constants[6].i = out_gpu_row.w;
    constants[7].i = out_gpu_row.h;
    constants[8].i = out_gpu_row.cstep;
    constants[9].i = tile.xi * tile.size_x;
    constants[10].i = std::min(tile.yi * tile.size_y, prepadding);
    constants[11].i = tile.xi * tile.size_x * scale;
    constants[12].i = std::min(tile.size_x * scale, out_gpu_row.w - tile.xi * tile.size_x * scale);
    constants[13].i = channels;
    constants[14].i = out_alpha_tile_gpu.w;
    constants[15].i = out_alpha_tile_gpu.h;

    ncnn::VkMat dispatcher;
    dispatcher.w = std::min(tile.size_x * scale, out_gpu_row.w - tile.xi * tile.size_x * scale);
    dispatcher.h = out_gpu_row.h;
    dispatcher.c = channels;

    cmd.record_pipeline(format == ColorFormat::RGB ? &pipelines.postprocess_rgb : &pipelines.postprocess_bgr, bindings, constants, dispatcher);
}

void postprocess_tile_4x_tta_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, const ncnn::VkMat& out_gpu_row, ncnn::VkMat out_tile_gpu[8], ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt) {
    std::vector<ncnn::VkMat> bindings(11);
    bindings[0] = in_gpu_row;
    bindings[1] = out_tile_gpu[0];
    bindings[2] = out_tile_gpu[1];
    bindings[3] = out_tile_gpu[2];
    bindings[4] = out_tile_gpu[3];
    bindings[5] = out_tile_gpu[4];
    bindings[6] = out_tile_gpu[5];
    bindings[7] = out_tile_gpu[6];
    bindings[8] = out_tile_gpu[7];
    bindings[9] = out_alpha_tile_gpu;
    bindings[10] = out_gpu_row;

    std::vector<ncnn::vk_constant_type> constants(16);
    constants[0].i = in_gpu_row.w;
    constants[1].i = in_gpu_row.h;
    constants[2].i = in_gpu_row.cstep;
    constants[3].i = out_tile_gpu[0].w;
    constants[4].i = out_tile_gpu[0].h;
    constants[5].i = out_tile_gpu[0].cstep;
    constants[6].i = out_gpu_row.w;
    constants[7].i = out_gpu_row.h;
    constants[8].i = out_gpu_row.cstep;
    constants[9].i = tile.xi * tile.size_x;
    constants[10].i = std::min(tile.yi * tile.size_y, prepadding);
    constants[11].i = tile.xi * tile.size_x * scale;
    constants[12].i = std::min(tile.size_x * scale, out_gpu_row.w - tile.xi * tile.size_x * scale);
    constants[13].i = channels;
    constants[14].i = out_alpha_tile_gpu.w;
    constants[15].i = out_alpha_tile_gpu.h;

    ncnn::VkMat dispatcher;
    dispatcher.w = std::min(tile.size_x * scale, out_gpu_row.w - tile.xi * tile.size_x * scale);
    dispatcher.h = out_gpu_row.h;
    dispatcher.c = channels;

    cmd.record_pipeline(format == ColorFormat::RGB ? &pipelines.postprocess_rgb : &pipelines.postprocess_bgr, bindings, constants, dispatcher);
}

}  // namespace
