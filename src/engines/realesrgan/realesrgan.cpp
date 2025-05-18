#include "realesrgan.hpp"

#include "../../encoding_utils.hpp"
#include "../check_result.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <vector>

namespace {

const uint32_t realesrgan_preproc_spv_data[] = {
#include "shaders/realesrgan_preproc.spv.hex.h"
};
const uint32_t realesrgan_preproc_fp16s_spv_data[] = {
#include "shaders/realesrgan_preproc_fp16s.spv.hex.h"
};
const uint32_t realesrgan_preproc_int8s_spv_data[] = {
#include "shaders/realesrgan_preproc_int8s.spv.hex.h"
};
const uint32_t realesrgan_postproc_spv_data[] = {
#include "shaders/realesrgan_postproc.spv.hex.h"
};
const uint32_t realesrgan_postproc_fp16s_spv_data[] = {
#include "shaders/realesrgan_postproc_fp16s.spv.hex.h"
};
const uint32_t realesrgan_postproc_int8s_spv_data[] = {
#include "shaders/realesrgan_postproc_int8s.spv.hex.h"
};

const uint32_t realesrgan_preproc_tta_spv_data[] = {
#include "shaders/realesrgan_preproc_tta.spv.hex.h"
};
const uint32_t realesrgan_preproc_tta_fp16s_spv_data[] = {
#include "shaders/realesrgan_preproc_tta_fp16s.spv.hex.h"
};
const uint32_t realesrgan_preproc_tta_int8s_spv_data[] = {
#include "shaders/realesrgan_preproc_tta_int8s.spv.hex.h"
};
const uint32_t realesrgan_postproc_tta_spv_data[] = {
#include "shaders/realesrgan_postproc_tta.spv.hex.h"
};
const uint32_t realesrgan_postproc_tta_fp16s_spv_data[] = {
#include "shaders/realesrgan_postproc_tta_fp16s.spv.hex.h"
};
const uint32_t realesrgan_postproc_tta_int8s_spv_data[] = {
#include "shaders/realesrgan_postproc_tta_int8s.spv.hex.h"
};

static constexpr int REALESRGAN_PREPADDING = 10;

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

int extract_features(const std::shared_ptr<spdlog::logger>& logger_error, ncnn::Net& net, const ncnn::Option& options, const ncnn::VkMat& in_tile, ncnn::VkMat& out_tile, ncnn::VkCompute& cmd);
void preprocess_tile_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, ncnn::VkMat& in_tile_gpu, ncnn::VkMat& in_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);
void preprocess_tile_tta_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& in_gpu_row, ncnn::VkMat in_tile_gpu[8], ncnn::VkMat& in_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);
void postprocess_tile_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& out_gpu_row, ncnn::VkMat& out_tile_gpu, ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);
void postprocess_tile_tta_gpu(const SuperResolutionPipelines& pipelines, const ncnn::VkMat& out_gpu_row, ncnn::VkMat out_tile_gpu[8], ncnn::VkMat& out_alpha_tile_gpu, int prepadding, int channels, ColorFormat format, int scale, const Tile& tile, ncnn::VkCompute& cmd, const ncnn::Option& opt);

}  // namespace

#define CHECK_NCNN_RESULT(expr) CHECK_NCNN_RESULT_AND_LOG(this->config.logger_error, expr)

RealESRGAN::RealESRGAN(const SuperResolutionEngineConfig& config)
    : SuperResolutionEngine(RealESRGAN::get_engine_info(), config) {
}

int RealESRGAN::get_default_tile_size() const {
    // Determine tile size based on available GPU memory
    const uint32_t heap_budget = this->vkdev != nullptr ? this->vkdev->get_heap_budget() : 2000;
    if (heap_budget > 1900) {
        return 200;
    }
    if (heap_budget > 550) {
        return 100;
    }
    if (heap_budget > 190) {
        return 64;
    }
    return 32;
}

void RealESRGAN::prepare_net_options(ncnn::Option& options) const {
    SuperResolutionEngine::prepare_net_options(options);

    // Set network options
    options.use_fp16_packed = true;
    options.use_fp16_storage = true;
    options.use_fp16_arithmetic = false;
    options.use_int8_storage = true;
    options.use_int8_arithmetic = false;
}

std::shared_ptr<ncnn::Net> RealESRGAN::create_net(int scale, const NetCache& net_cache) const {
    // Use default model if none specified
    std::u8string model_name = this->config.model.empty() ? this->engine_info.default_model : this->config.model;

    // Check if model is supported
    if (!this->engine_info.supports_model(model_name)) {
        this->config.logger_error->error("Model '{}' is not supported by RealESRGAN", utf8_to_ascii(model_name));
    }

    std::u8string basename;
    if (model_name == u8"realesr-animevideov3") {
        // For `realesr-animevideov3`, the model depends on the scale
        basename = ascii_to_utf8(std::format("{}-x{}", utf8_to_ascii(model_name), scale));
    } else {
        // For other models, the scale is not part of the model name
        // Use scale 1 for the representative model for convenience
        if (scale != 1) {
            // Reuse the cached model for other scales
            return net_cache.get_net(1);
        }

        basename = model_name;
    }

    auto net = this->create_net_base();
    if (this->net_load_model_and_param(*net, this->config.model_dir / (basename + u8".param")) != 0) {
        return nullptr;
    }

    return net;
}

std::shared_ptr<SuperResolutionPipelines> RealESRGAN::create_pipelines(int scale, const PipelineCache&) const {
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
    pipelines.preprocess_rgb.set_optimal_local_size_xyz(32, 32, 3);
    pipelines.preprocess_bgr.set_optimal_local_size_xyz(32, 32, 3);
    pipelines.postprocess_rgb.set_optimal_local_size_xyz(32, 32, 3);
    pipelines.postprocess_bgr.set_optimal_local_size_xyz(32, 32, 3);

    if (!this->config.tta_mode) {
        // Standard preprocessing and postprocessing
        switch (storage_mode) {
            case StorageMode::FP16_INT8:
                pipelines.preprocess_rgb.create(realesrgan_preproc_int8s_spv_data, sizeof(realesrgan_preproc_int8s_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realesrgan_preproc_int8s_spv_data, sizeof(realesrgan_preproc_int8s_spv_data), specializations_bgr);
                pipelines.postprocess_rgb.create(realesrgan_postproc_int8s_spv_data, sizeof(realesrgan_postproc_int8s_spv_data), specializations_rgb);
                pipelines.postprocess_bgr.create(realesrgan_postproc_int8s_spv_data, sizeof(realesrgan_postproc_int8s_spv_data), specializations_bgr);
                break;

            case StorageMode::FP16:
                pipelines.preprocess_rgb.create(realesrgan_preproc_fp16s_spv_data, sizeof(realesrgan_preproc_fp16s_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realesrgan_preproc_fp16s_spv_data, sizeof(realesrgan_preproc_fp16s_spv_data), specializations_bgr);
                pipelines.postprocess_rgb.create(realesrgan_postproc_fp16s_spv_data, sizeof(realesrgan_postproc_fp16s_spv_data), specializations_rgb);
                pipelines.postprocess_bgr.create(realesrgan_postproc_fp16s_spv_data, sizeof(realesrgan_postproc_fp16s_spv_data), specializations_bgr);
                break;

            default:
                pipelines.preprocess_rgb.create(realesrgan_preproc_spv_data, sizeof(realesrgan_preproc_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realesrgan_preproc_spv_data, sizeof(realesrgan_preproc_spv_data), specializations_bgr);
                pipelines.postprocess_rgb.create(realesrgan_postproc_spv_data, sizeof(realesrgan_postproc_spv_data), specializations_rgb);
                pipelines.postprocess_bgr.create(realesrgan_postproc_spv_data, sizeof(realesrgan_postproc_spv_data), specializations_bgr);
                break;
        }
    } else {
        // TTA preprocessing and postprocessing
        switch (storage_mode) {
            case StorageMode::FP16_INT8:
                pipelines.preprocess_rgb.create(realesrgan_preproc_tta_int8s_spv_data, sizeof(realesrgan_preproc_tta_int8s_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realesrgan_preproc_tta_int8s_spv_data, sizeof(realesrgan_preproc_tta_int8s_spv_data), specializations_bgr);
                pipelines.postprocess_rgb.create(realesrgan_postproc_tta_int8s_spv_data, sizeof(realesrgan_postproc_tta_int8s_spv_data), specializations_rgb);
                pipelines.postprocess_bgr.create(realesrgan_postproc_tta_int8s_spv_data, sizeof(realesrgan_postproc_tta_int8s_spv_data), specializations_bgr);
                break;

            case StorageMode::FP16:
                pipelines.preprocess_rgb.create(realesrgan_preproc_tta_fp16s_spv_data, sizeof(realesrgan_preproc_tta_fp16s_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realesrgan_preproc_tta_fp16s_spv_data, sizeof(realesrgan_preproc_tta_fp16s_spv_data), specializations_bgr);
                pipelines.postprocess_rgb.create(realesrgan_postproc_tta_fp16s_spv_data, sizeof(realesrgan_postproc_tta_fp16s_spv_data), specializations_rgb);
                pipelines.postprocess_bgr.create(realesrgan_postproc_tta_fp16s_spv_data, sizeof(realesrgan_postproc_tta_fp16s_spv_data), specializations_bgr);
                break;

            default:
                pipelines.preprocess_rgb.create(realesrgan_preproc_tta_spv_data, sizeof(realesrgan_preproc_tta_spv_data), specializations_rgb);
                pipelines.preprocess_bgr.create(realesrgan_preproc_tta_spv_data, sizeof(realesrgan_preproc_tta_spv_data), specializations_bgr);
                pipelines.postprocess_rgb.create(realesrgan_postproc_tta_spv_data, sizeof(realesrgan_postproc_tta_spv_data), specializations_rgb);
                pipelines.postprocess_bgr.create(realesrgan_postproc_tta_spv_data, sizeof(realesrgan_postproc_tta_spv_data), specializations_bgr);
                break;
        }
    }

    return sp;
}

int RealESRGAN::process_gpu(const ncnn::Mat& in, ColorFormat in_format, ncnn::Mat& out, ColorFormat out_format, const ProcessConfig& config) const {
    const unsigned char* in_data = static_cast<const unsigned char*>(in.data);
    const int w = in.w;
    const int h = in.h;
    const int channels = in.elempack;

    unsigned char* out_data = static_cast<unsigned char*>(out.data);

    // Get parameters from config
    const int scale = config.scale;
    const int tile_size = config.tile_size;
    const int prepadding = REALESRGAN_PREPADDING;

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
    const VkAllocators allocators(this->vkdev);

    const auto blob_vkallocator = allocators.get_blob_allocator();
    if (blob_vkallocator == nullptr) {
        this->config.logger_error->error("[{}] Failed to acquire blob allocator", __func__);
        return -1;
    }

    const auto staging_vkallocator = allocators.get_staging_allocator();
    if (staging_vkallocator == nullptr) {
        this->config.logger_error->error("[{}] Failed to acquire staging allocator", __func__);
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

        const int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        const int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, h);

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
            CHECK_NCNN_RESULT(cmd.submit_and_wait());
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

            // Calculate tile coordinates
            const int tile_x0 = xi * TILE_SIZE_X - prepadding;
            const int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding;
            const int tile_y0 = yi * TILE_SIZE_Y - prepadding;
            const int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding;

            const Tile tile{
                .elemsize = in_out_tile_elemsize,

                .size_x = TILE_SIZE_X,
                .size_y = TILE_SIZE_Y,

                .w_nopad = tile_w_nopad,
                .h_nopad = tile_h_nopad,

                .xi = xi,
                .yi = yi,

                .x0 = tile_x0,
                .y0 = tile_y0,
                .x1 = tile_x1,
                .y1 = tile_y1,
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
                CHECK_NCNN_RESULT(extract_features(this->config.logger_error, *ptr_net, opt, in_tile_gpu, out_tile_gpu, cmd));

                // Handle alpha channel
                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    CHECK_NCNN_RESULT(this->handle_alpha_channel_gpu(in_alpha_tile_gpu, out_alpha_tile_gpu, scale, cmd, opt));
                }

                // Postprocess
                postprocess_tile_gpu(pipelines, out_gpu_row, out_tile_gpu, out_alpha_tile_gpu,
                                     prepadding, channels, out_format, scale, tile, cmd, opt);
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
                    CHECK_NCNN_RESULT(extract_features(this->config.logger_error, *ptr_net, opt, in_tile_gpu[ti], out_tile_gpu[ti], cmd));

                    CHECK_NCNN_RESULT(cmd.submit_and_wait());
                    cmd.reset();
                }

                // Handle alpha channel
                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    CHECK_NCNN_RESULT(this->handle_alpha_channel_gpu(in_alpha_tile_gpu, out_alpha_tile_gpu, scale, cmd, opt));
                }

                // Postprocess with TTA
                postprocess_tile_tta_gpu(pipelines, out_gpu_row, out_tile_gpu, out_alpha_tile_gpu,
                                         prepadding, channels, out_format, scale, tile, cmd, opt);
            }

            if (xtiles > 1) {
                CHECK_NCNN_RESULT(cmd.submit_and_wait());
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
        CHECK_NCNN_RESULT(cmd.submit_and_wait());

        // Copy to output image
        if (storage_mode != StorageMode::FP16_INT8) {
            // Copy output tile to the output mat
            out_cpu_row.to_pixels(out_data + yi * scale * TILE_SIZE_Y * w * scale * channels,
                                  out_pixel_format);
        }
    }

    return 0;
}

namespace {

int extract_features(const std::shared_ptr<spdlog::logger>& logger_error, ncnn::Net& net, const ncnn::Option& options, const ncnn::VkMat& in_tile, ncnn::VkMat& out_tile, ncnn::VkCompute& cmd) {
    ncnn::Extractor ex = net.create_extractor();

    ex.set_blob_vkallocator(options.blob_vkallocator);
    ex.set_workspace_vkallocator(options.workspace_vkallocator);
    ex.set_staging_vkallocator(options.staging_vkallocator);

    CHECK_NCNN_RESULT_AND_LOG(logger_error, ex.input("data", in_tile));
    CHECK_NCNN_RESULT_AND_LOG(logger_error, ex.extract("output", out_tile, cmd));

    return 0;
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

    std::vector<ncnn::vk_constant_type> constants(13);
    constants[0].i = out_tile_gpu.w;
    constants[1].i = out_tile_gpu.h;
    constants[2].i = out_tile_gpu.cstep;
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

}  // namespace
