#include <algorithm>
#include <chrono>
#include <clocale>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <ranges>
#include <regex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// Third-party libraries
#include "cxxopts.hpp"
#include "glaze/glaze.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "../encoding_utils.hpp"
#include "../plugin/upsclr_plugin.h"

// Image processing
#if _WIN32
#    include "wic_image.h"
#else
#    define STB_IMAGE_IMPLEMENTATION
#    define STBI_NO_PSD
#    define STBI_NO_TGA
#    define STBI_NO_GIF
#    define STBI_NO_HDR
#    define STBI_NO_PIC
#    define STBI_NO_STDIO
#    include "stb_image.h"
#    define STB_IMAGE_WRITE_IMPLEMENTATION
#    include "stb_image_write.h"
#endif
#include "webp_image.h"

namespace {

constexpr int DEFAULT_GPU_DEVICE_ID_PLACEHOLDER = -999;
constexpr int MAX_QUEUE_SIZE = 8;

#if _WIN32
#    define PATHSTR(X) L##X

std::wstring p2s(const std::filesystem::path& path) {
    return path.wstring();
}
#else
#    define PATHSTR(X) X

std::string p2s(const std::filesystem::path& path) {
    return path.string();
}
#endif

int list_directory(const std::filesystem::path& dir, std::vector<std::filesystem::path>& filepaths) {
    filepaths.clear();

    try {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                filepaths.push_back(entry.path().filename());
            }
        }
        std::ranges::sort(filepaths);
        return 0;
    } catch (const std::filesystem::filesystem_error& e) {
#if _WIN32
        fwprintf(stderr, L"opendir failed %ls: %hs\n", dir.wstring().c_str(), e.what());
#else
        fprintf(stderr, "opendir failed %s: %s\n", dir.string().c_str(), e.what());
#endif
        return -1;
    }
}

// Structure to store image data
struct ImageData final {
    int w = 0;
    int h = 0;
    int c = 0;
    void* data = nullptr;
    UpsclrColorFormat color_format = UPSCLR_COLOR_FORMAT_RGB;
};

// Class to manage plugin instance
class PluginManager final {
    std::unique_ptr<UpsclrEngineInstance, std::function<void(UpsclrEngineInstance*)>> instance;
    size_t engine_index;
    std::u8string config_json;
    std::shared_ptr<spdlog::logger> logger_info;
    std::shared_ptr<spdlog::logger> logger_error;

   public:
    PluginManager(size_t engine_idx, const std::u8string& config, std::shared_ptr<spdlog::logger> info_logger, std::shared_ptr<spdlog::logger> error_logger)
        : instance(nullptr), engine_index(engine_idx), config_json(config), logger_info(info_logger), logger_error(error_logger) {
        instance = std::unique_ptr<UpsclrEngineInstance, std::function<void(UpsclrEngineInstance*)>>(
            upsclr_plugin_create_engine_instance(
                engine_idx,
                config_json.data(),
                config_json.length()),
            [](UpsclrEngineInstance* instance) {
                if (instance == nullptr) {
                    return;
                }

                upsclr_plugin_destroy_engine_instance(instance);
            });

        if (instance == nullptr) {
            logger_error->error("Failed to create engine instance (engine index {})", engine_idx);
        }
    }

    ~PluginManager() = default;

    UpsclrErrorCode preload(int scale) {
        if (instance == nullptr) {
            return UPSCLR_ERROR_ENGINE_NOT_FOUND;
        }

        return upsclr_preload_upscale(instance.get(), scale);
    }

    UpsclrErrorCode process(void* in_data, int in_width, int in_height, int in_channels, void* out_data, int out_width, int out_height, int out_channels, int scale, UpsclrColorFormat in_format, UpsclrColorFormat out_format) {
        if (instance == nullptr) {
            return UPSCLR_ERROR_ENGINE_NOT_FOUND;
        }

        return upsclr_upscale(
            instance.get(),
            scale,
            static_cast<const unsigned char*>(in_data),
            in_width * in_height * in_channels,
            in_width,
            in_height,
            in_channels,
            in_format,
            static_cast<unsigned char*>(out_data),
            out_width * out_height * out_channels,
            out_format);
    }

    static void print_available_engines() {
        const size_t engine_count = upsclr_plugin_count_engines();

        fprintf(stderr, "\nAvailable engines:\n");
        for (size_t i = 0; i < engine_count; ++i) {
            const auto* info = upsclr_plugin_get_engine_info(i);
            if (info == nullptr) {
                continue;
            }

            fprintf(stderr, "  %s: %s\n", utf8_to_ascii(info->name).c_str(), utf8_to_ascii(info->description).c_str());

            // Parse JSON schema to extract model names
            fprintf(stderr, "    Models: ");

            // This is a simplified approach - in a real implementation,
            // you would parse the JSON schema properly
            std::u8string schema(info->config_json_schema);
            size_t model_pos = schema.find(u8"\"model\"");
            size_t enum_pos = schema.find(u8"\"enumeration\"", model_pos);

            if (model_pos != std::u8string::npos && enum_pos != std::u8string::npos) {
                size_t start = schema.find(u8'[', enum_pos);
                size_t end = schema.find(u8']', start);

                if (start != std::u8string::npos && end != std::u8string::npos) {
                    std::string models = utf8_to_ascii(schema.substr(start + 1, end - start - 1));
                    // Very simple parsing - would need improvement for real use
                    models = std::regex_replace(models, std::regex("\""), "");
                    models = std::regex_replace(models, std::regex(","), ", ");
                    fprintf(stderr, "%s", models.c_str());
                }
            }

            fprintf(stderr, "\n\n");
        }
    }

    static size_t find_engine_by_name(const std::u8string& name) {
        const size_t engine_count = upsclr_plugin_count_engines();

        for (size_t i = 0; i < engine_count; ++i) {
            const auto* info = upsclr_plugin_get_engine_info(i);
            if (info == nullptr) {
                continue;
            }

            if (name == info->name) {
                return i;
            }
        }

        return static_cast<size_t>(-1);
    }
};

// Function to generate engine configuration JSON
std::u8string generate_engine_config(const std::u8string& engine_name,
                                     const std::u8string& model_dir,
                                     const std::u8string& model_name,
                                     int gpu_id,
                                     int tile_size,
                                     bool tta_mode,
                                     int noise = 0,
                                     int sync_gap = 0) {
    std::map<std::string, std::variant<std::string, int, bool>> data;

    if (gpu_id != DEFAULT_GPU_DEVICE_ID_PLACEHOLDER) {
        data["gpu_id"] = gpu_id;
    }

    if (tile_size > 0) {
        data["tile_size"] = tile_size;
    }

    data["tta_mode"] = tta_mode;

    if (engine_name == u8"realcugan") {
        data["noise"] = noise;
        data["sync_gap"] = sync_gap;
    }

    data["model_dir"] = as_string(model_dir);

    if (model_name != u8"") {
        data["model"] = as_string(model_name);
    }

    return as_utf8(glz::write_json(data).value_or("{}"));
}

struct Task final {
    int id;
    int webp;

    std::chrono::microseconds elapsed;

    std::filesystem::path in_path;
    std::filesystem::path out_path;

    ImageData in_image;
    ImageData out_image;

    int scale;
};

class TaskQueue final {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<Task> tasks;

   public:
    TaskQueue() = default;

    void put(const Task& v) {
        std::unique_lock lock(mutex);

        cv.wait(lock, [&] {
            return tasks.size() < MAX_QUEUE_SIZE;
        });

        tasks.push(v);

        cv.notify_all();
    }

    void get(Task& v) {
        std::unique_lock lock(mutex);

        cv.wait(lock, [&] {
            return tasks.size() > 0;
        });

        v = tasks.front();
        tasks.pop();

        cv.notify_all();
    }
};

TaskQueue to_proc_queue;
TaskQueue to_save_queue;

struct LoadThreadParams final {
    int scale;
    int jobs_load;

    // session data
    std::vector<std::filesystem::path> input_files;
    std::vector<std::filesystem::path> output_files;

    std::shared_ptr<spdlog::logger> logger_info;
    std::shared_ptr<spdlog::logger> logger_error;
};

void load(const LoadThreadParams& ltp) {
    const int count = static_cast<int>(ltp.input_files.size());
    const int scale = ltp.scale;

#pragma omp parallel for schedule(static, 1) num_threads(ltp.jobs_load)
    for (int i = 0; i < count; i++) {
        const auto& image_path = ltp.input_files[i];
        int webp = 0;

        unsigned char* in_image_data = nullptr;
        int w;
        int h;
        int c;

#if _WIN32
        FILE* fp = _wfopen(image_path.wstring().c_str(), L"rb");
#else
        FILE* fp = fopen(image_path.string().c_str(), "rb");
#endif
        if (fp == nullptr) {
            continue;
        }

        // read whole file
        std::unique_ptr<unsigned char[]> file_data;
        int length = 0;
        fseek(fp, 0, SEEK_END);
        length = ftell(fp);
        rewind(fp);
        file_data = std::make_unique<unsigned char[]>(length);
        fread(file_data.get(), 1, length, fp);
        fclose(fp);

        // decode image
        if (length > 12 && file_data[0] == 'R' && file_data[8] == 'W') {
            // RIFF____WEBP
            in_image_data = webp_load(file_data.get(), length, &w, &h, &c);
            if (in_image_data != nullptr) {
                webp = 1;
            }
        }
        if (in_image_data == nullptr) {
            // not webp, try jpg png etc.
#if _WIN32
            in_image_data = wic_decode_image(image_path.wstring().c_str(), &w, &h, &c);
#else   // _WIN32
            in_image_data = stbi_load_from_memory(file_data.get(), length, &w, &h, &c, 0);
            if (in_image_data != nullptr) {
                // stb_image auto channel
                if (c == 1) {
                    // grayscale -> rgb
                    stbi_image_free(in_image_data);
                    in_image_data = stbi_load_from_memory(file_data.get(), length, &w, &h, &c, 3);
                    c = 3;
                } else if (c == 2) {
                    // grayscale + alpha -> rgba
                    stbi_image_free(in_image_data);
                    in_image_data = stbi_load_from_memory(file_data.get(), length, &w, &h, &c, 4);
                    c = 4;
                }
            }
#endif  // _WIN32
        }

        if (in_image_data == nullptr) {
            ltp.logger_error->error(PATHSTR("Decode failed: {}"), p2s(image_path));
            continue;
        }

        Task v;
        v.id = i;
        v.in_path = image_path;
        v.out_path = ltp.output_files[i];
        v.webp = webp;

        // Store image data
        v.in_image.w = w;
        v.in_image.h = h;
        v.in_image.c = c;
        v.in_image.data = in_image_data;

        // Allocate memory for output image
        v.out_image.w = w * scale;
        v.out_image.h = h * scale;
        v.out_image.c = c;
        v.out_image.data = std::malloc(w * scale * h * scale * c);

        v.scale = scale;

#if _WIN32
        v.in_image.color_format = UPSCLR_COLOR_FORMAT_BGR;
        v.out_image.color_format = UPSCLR_COLOR_FORMAT_BGR;
#else
        v.in_image.color_format = UPSCLR_COLOR_FORMAT_RGB;
        v.out_image.color_format = UPSCLR_COLOR_FORMAT_RGB;
#endif

        if (const auto ext = v.out_path.extension(); c == 4 && (ext == ".jpg" || ext == ".JPG" || ext == ".jpeg" || ext == ".JPEG")) {
            std::filesystem::path out_path_renamed = ltp.output_files[i];
            out_path_renamed += ".png";
            v.out_path = out_path_renamed;
            ltp.logger_error->warn(PATHSTR("Image {} has alpha channel! {} will output {}"), p2s(image_path), p2s(image_path), p2s(out_path_renamed));
        }

        to_proc_queue.put(v);
    }
}

struct ProcThreadParams final {
    PluginManager* plugin_manager;
    std::atomic<long long>* total_elapsed;
};

void proc(ProcThreadParams ptp) {
    const auto plugin_manager = ptp.plugin_manager;

    for (;;) {
        Task v;

        to_proc_queue.get(v);

        if (v.id == -233) {
            break;
        }

        const auto tp_begin = std::chrono::steady_clock::now();

        UpsclrErrorCode result = plugin_manager->process(
            v.in_image.data, v.in_image.w, v.in_image.h, v.in_image.c,
            v.out_image.data, v.out_image.w, v.out_image.h, v.out_image.c,
            v.scale, v.in_image.color_format, v.out_image.color_format);

        const auto tp_end = std::chrono::steady_clock::now();

        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_begin);
        v.elapsed = elapsed;
        ptp.total_elapsed->fetch_add(elapsed.count() * 1'000'000 / (v.in_image.w * v.in_image.h));

        to_save_queue.put(v);
    }
}

struct SaveThreadParams final {
    bool verbose;

    std::shared_ptr<spdlog::logger> logger_info;
    std::shared_ptr<spdlog::logger> logger_error;
};

void save(const SaveThreadParams& stp) {
    const bool verbose = stp.verbose;

    for (;;) {
        Task v;

        to_save_queue.get(v);

        if (v.id == -233) {
            break;
        }

        // Free input pixel data
        {
            if (v.webp == 1) {
                std::free(v.in_image.data);
            } else {
#if _WIN32
                std::free(v.in_image.data);
#else
                stbi_image_free(v.in_image.data);
#endif
            }
        }

        const auto ext = v.out_path.extension().u8string();

        // Create folder if not exists
        const auto fs_path = std::filesystem::absolute(v.out_path);
        const auto parent_path = fs_path.parent_path();
        if (!std::filesystem::exists(parent_path)) {
            stp.logger_info->info(PATHSTR("Create folder: {}"), p2s(parent_path));
            std::filesystem::create_directories(parent_path);
        }

        int success = 0;

        if (ext == u8".webp" || ext == u8".WEBP") {
            success = webp_save(v.out_path.c_str(), v.out_image.w, v.out_image.h, v.out_image.c, static_cast<const unsigned char*>(v.out_image.data));
        } else if (ext == u8".png" || ext == u8".PNG") {
#if _WIN32
            success = wic_encode_image(v.out_path.c_str(), v.out_image.w, v.out_image.h, v.out_image.c, v.out_image.data);
#else
            success = stbi_write_png(v.out_path.c_str(), v.out_image.w, v.out_image.h, v.out_image.c, v.out_image.data, 0);
#endif
        } else if (ext == u8".jpg" || ext == u8".JPG" || ext == u8".jpeg" || ext == u8".JPEG") {
#if _WIN32
            success = wic_encode_jpeg_image(v.out_path.c_str(), v.out_image.w, v.out_image.h, v.out_image.c, v.out_image.data);
#else
            success = stbi_write_jpg(v.out_path.c_str(), v.out_image.w, v.out_image.h, v.out_image.c, v.out_image.data, 100);
#endif
        }

        // Free output image memory
        std::free(v.out_image.data);

        if (success) {
            if (verbose) {
                stp.logger_info->info(PATHSTR("{} -> {} done, took {}us"), p2s(v.in_path), p2s(v.out_path), v.elapsed.count());
            }
        } else {
            stp.logger_error->error(PATHSTR("Encode failed: {}"), p2s(v.out_path));
        }
    }
}

void print_usage() {
    fprintf(stderr, "Usage: upsclr-ncnn-vulkan-2 -i infile -o outfile [options]...\n\n");
    fprintf(stderr, "  -h,--help                   show this help\n");
    fprintf(stderr, "  -i,--input INPUT-PATH       input image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -o,--output OUTPUT-PATH     output image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -s,--scale SCALE            upscale ratio (can be 2, 3, 4. default=4)\n");
    fprintf(stderr, "  -t,--tile-size TILE-SIZE    tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu\n");
    fprintf(stderr, "  -m,--model-path MODEL-PATH  folder path to the pre-trained models. default=models\n");
    fprintf(stderr, "  -n,--model-name MODEL-NAME  model name (default depends on engine)\n");
    fprintf(stderr, "  -e,--engine ENGINE          engine to use (realesrgan, realcugan, default=realesrgan)\n");
    fprintf(stderr, "  -g,--gpu-id GPU-ID          gpu device to use (default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stderr, "  -j,--threads THREADS        thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stderr, "  -x,--tta-mode               enable tta mode\n");
    fprintf(stderr, "  -f,--format FORMAT          output image format (jpg/png/webp, default=ext/png)\n");
    fprintf(stderr, "  -v,--verbose                verbose output\n");
    fprintf(stderr, "  -y,--noise NOISE            noise level (-1/0/1/2/3, default=0, only for realcugan)\n");
    fprintf(stderr, "  -z,--sync-gap SYNCGAP       sync gap mode (0/1/2/3, default=0, only for realcugan)\n");

    // Print available engines
    PluginManager::print_available_engines();
}

// Parse comma-separated values into vector of integers
std::vector<int> parse_int_list(const std::string& str) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            try {
                result.push_back(std::stoi(item));
            } catch (const std::exception&) {
                // Skip invalid values
            }
        }
    }

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    const auto logger_info = spdlog::stdout_color_mt("console");
    const auto logger_error = spdlog::stderr_color_mt("stderr");

    // Parse command line arguments
    cxxopts::Options options("upsclr-ncnn-vulkan", "Image super-resolution using upsclr-plugin-ncnn-vulkan");

    options.add_options()                                                                                                    //
        ("h,help", "Show this help")                                                                                         //
        ("i,input", "Input image path (jpg/png/webp) or directory", cxxopts::value<std::string>())                           //
        ("o,output", "Output image path (jpg/png/webp) or directory", cxxopts::value<std::string>())                         //
        ("s,scale", "Upscale ratio (2, 3, 4)", cxxopts::value<int>()->default_value("4"))                                    //
        ("t,tile-size", "Tile size (>=32/0=auto)", cxxopts::value<std::string>()->default_value("0"))                        //
        ("m,model-dir", "Directory path to the pre-trained models", cxxopts::value<std::string>()->default_value("models"))  //
        ("n,model-name", "Model name", cxxopts::value<std::string>())                                                        //
        ("e,engine", "Engine to use (realesrgan, realcugan)", cxxopts::value<std::string>()->default_value("realesrgan"))    //
        ("g,gpu-id", "GPU device to use", cxxopts::value<std::string>())                                                     //
        ("j,threads", "Thread count for load/proc/save", cxxopts::value<std::string>()->default_value("1:2:2"))              //
        ("x,tta-mode", "Enable TTA mode", cxxopts::value<bool>()->default_value("false"))                                    //
        ("f,format", "Output image format (jpg/png/webp)", cxxopts::value<std::string>()->default_value("png"))              //
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))                                      //
        ("y,noise", "Noise level (-1/0/1/2/3, only for realcugan)", cxxopts::value<int>()->default_value("0"))               //
        ("z,sync-gap", "Sync gap mode (0/1/2/3, only for realcugan)", cxxopts::value<int>()->default_value("0"));

    const auto result = options.parse(argc, argv);

    if (result.count("help")) {
        print_usage();
        return 0;
    }

    std::filesystem::path input_path;
    std::filesystem::path output_path;
    int scale = result["scale"].as<int>();
    std::vector<int> tile_size;
    std::filesystem::path model_dir = u8"models";
    std::u8string model_name = u8"";
    std::u8string engine_name = as_utf8(result["engine"].as<std::string>());
    int noise = result["noise"].as<int>();
    int sync_gap = result["sync-gap"].as<int>();
    std::vector<int> gpu_id;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 2;
    bool verbose = result["verbose"].as<bool>();
    int tta_mode = result["tta-mode"].as<bool>() ? 1 : 0;
    std::u8string format = u8"png";

    // Get input and output paths
    if (result.count("input")) {
        input_path = as_utf8(result["input"].as<std::string>());
    } else {
        print_usage();
        return -1;
    }

    if (result.count("output")) {
        output_path = as_utf8(result["output"].as<std::string>());
    } else {
        print_usage();
        return -1;
    }

    // Parse tile size
    if (result.count("tile-size")) {
        std::string tile_str = result["tile-size"].as<std::string>();
        tile_size = parse_int_list(tile_str);
    }

    // Parse model path
    if (result.count("model-dir")) {
        model_dir = as_utf8(result["model-dir"].as<std::string>());
    }

    // Parse model name
    if (result.count("model-name")) {
        model_name = as_utf8(result["model-name"].as<std::string>());
    }

    // Parse GPU ID
    if (result.count("gpu-id")) {
        std::string gpu_str = result["gpu-id"].as<std::string>();
        gpu_id = parse_int_list(gpu_str);
    }

    // Parse thread count
    if (result.count("threads")) {
        std::string threads_str = result["threads"].as<std::string>();
        size_t pos1 = threads_str.find(':');
        if (pos1 != std::string::npos) {
            jobs_load = std::stoi(threads_str.substr(0, pos1));

            size_t pos2 = threads_str.find(':', pos1 + 1);
            if (pos2 != std::string::npos) {
                std::string proc_str = threads_str.substr(pos1 + 1, pos2 - pos1 - 1);
                jobs_proc = parse_int_list(proc_str);
                jobs_save = std::stoi(threads_str.substr(pos2 + 1));
            } else {
                std::string proc_str = threads_str.substr(pos1 + 1);
                jobs_proc = parse_int_list(proc_str);
            }
        }
    }

    // Parse output format
    if (result.count("format")) {
        format = as_utf8(result["format"].as<std::string>());
    }

    if (input_path.empty() || output_path.empty()) {
        print_usage();
        return -1;
    }

    if (gpu_id.empty()) {
        if (verbose) {
            logger_info->info("No GPU specified, using default GPU device");
        }
        gpu_id.push_back(DEFAULT_GPU_DEVICE_ID_PLACEHOLDER);
    }

    const auto expected_gpu_related_item_count = gpu_id.size() > 1 ? std::format("1 or {}", gpu_id.size()) : "1";

    if (tile_size.empty()) {
        if (verbose) {
            logger_info->info("No tile_size specified, using default tile_size 0");
        }
        tile_size.push_back(0);
    }

    if (jobs_proc.empty()) {
        if (verbose) {
            logger_info->info("No jobs_proc specified, using default jobs_proc 2");
        }
        jobs_proc.push_back(2);
    }

    if (scale < 2 || scale > 4) {
        logger_error->error("Invalid scale");
        return -1;
    }

    if (tile_size.size() != 1 && tile_size.size() != gpu_id.size()) {
        logger_error->error("Invalid tile_size: expected {} tile_size, got {}", expected_gpu_related_item_count, tile_size.size());
        return -1;
    }

    for (const auto tile : tile_size) {
        if (tile != 0 && tile < 32) {
            logger_error->error("Invalid tile_size: expected >=32 or 0, got {}", tile);
            return -1;
        }
    }

    if (noise < -1 || noise > 3) {
        logger_error->error("Invalid noise: expected -1/0/1/2/3, got {}", noise);
        return -1;
    }

    if (sync_gap < 0 || sync_gap > 3) {
        logger_error->error("Invalid sync_gap: expected 0/1/2/3, got {}", sync_gap);
        return -1;
    }

    if (jobs_proc.size() != 1 && jobs_proc.size() != gpu_id.size()) {
        logger_error->error("Invalid jobs_proc: expected {} jobs_proc, got {}", expected_gpu_related_item_count, jobs_proc.size());
        return -1;
    }

    for (const auto count : jobs_proc) {
        if (count < 1) {
            logger_error->error("Invalid jobs_proc: expected >=1, got {}", count);
            return -1;
        }
    }

    if (!std::filesystem::is_directory(output_path)) {
        // guess format from output_path no matter what format argument specified
        const auto ext = output_path.extension().u8string();

        if (ext == u8".png" || ext == u8".PNG") {
            format = u8"png";
        } else if (ext == u8".webp" || ext == u8".WEBP") {
            format = u8"webp";
        } else if (ext == u8".jpg" || ext == u8".JPG" || ext == u8".jpeg" || ext == u8".JPEG") {
            format = u8"jpg";
        } else {
            logger_error->error("Invalid output path extension: '{}'", utf8_to_ascii(ext));
            return -1;
        }
    }

    if (format != u8"png" && format != u8"webp" && format != u8"jpg") {
        logger_error->error("Invalid output format: '{}'", utf8_to_ascii(format));
        return -1;
    }

    // Normalize config
    if (jobs_proc.size() != gpu_id.size()) {
        jobs_proc.resize(gpu_id.size(), jobs_proc[0]);
    }

    if (tile_size.size() != gpu_id.size()) {
        tile_size.resize(gpu_id.size(), tile_size[0]);
    }

    // Count total proc thread count
    int total_jobs_proc = 0;
    for (const auto count : jobs_proc) {
        total_jobs_proc += count;
    }

    // collect input and output filepath
    std::vector<std::filesystem::path> input_files;
    std::vector<std::filesystem::path> output_files;
    {
        if (std::filesystem::is_directory(input_path) && std::filesystem::is_directory(output_path)) {
            std::vector<std::filesystem::path> filenames;
            if (list_directory(input_path, filenames) != 0) {
                return -1;
            }

            const size_t count = filenames.size();
            input_files.resize(count);
            output_files.resize(count);

            std::filesystem::path last_filename;
            std::filesystem::path last_filename_noext;
            for (size_t i = 0; i < count; i++) {
                std::filesystem::path filename = filenames[i];
                std::filesystem::path filename_noext = filename.stem();
                std::filesystem::path output_filename = filename;
                output_filename.replace_extension(format);

                // filename list is sorted, check if output image path conflicts
                if (filename_noext == last_filename_noext) {
                    std::filesystem::path output_filename2 = filename;
                    output_filename2 += u8"." + format;
                    logger_error->warn(PATHSTR("Image {} has same name with {}! {} will output {}"), p2s(filename), p2s(last_filename), p2s(filename), p2s(output_filename2));
                    output_filename = output_filename2;
                } else {
                    last_filename = filename;
                    last_filename_noext = filename_noext;
                }

                input_files[i] = input_path / filename;
                output_files[i] = output_path / output_filename;
            }
        } else if (!std::filesystem::is_directory(input_path) && !std::filesystem::is_directory(output_path)) {
            input_files.push_back(input_path);
            output_files.push_back(output_path);
        } else {
            logger_error->error("input path and output path must be either file or directory at the same time");
            return -1;
        }
    }

    // Find engine by name
    const size_t engine_index = PluginManager::find_engine_by_name(engine_name);
    if (engine_index == static_cast<size_t>(-1)) {
        logger_error->error("Unknown engine: '{}'", utf8_to_ascii(engine_name));
        return -1;
    }

    const auto* engine_info = upsclr_plugin_get_engine_info(engine_index);
    if (engine_info == nullptr) {
        logger_error->error("Unknown engine: '{}'", utf8_to_ascii(engine_name));
        return -1;
    }

    std::vector<std::unique_ptr<PluginManager>> plugin_managers;
    for (size_t i = 0; i < gpu_id.size(); i++) {
        // Configure plugin
        const std::u8string config_json = generate_engine_config(
            engine_name, model_dir.u8string(), model_name, gpu_id[i], tile_size[i], tta_mode, noise, sync_gap);

        logger_info->info("Config JSON: {}", utf8_to_ascii(config_json));

        // Create plugin instance
        auto plugin_manager = std::make_unique<PluginManager>(engine_index, config_json, logger_info, logger_error);

        // Preload
        UpsclrErrorCode preload_result = plugin_manager->preload(scale);
        if (preload_result != UPSCLR_SUCCESS) {
            logger_error->error("Failed to preload engine: {}", static_cast<int>(preload_result));
            return -1;
        }

        plugin_managers.emplace_back(std::move(plugin_manager));
    }

    // Main processing
    {
        std::atomic<long long> total_elapsed(0);

        // Set load thread parameters
        LoadThreadParams ltp;
        ltp.scale = scale;
        ltp.jobs_load = jobs_load;
        ltp.input_files = input_files;
        ltp.output_files = output_files;
        ltp.logger_info = logger_info;
        ltp.logger_error = logger_error;

        // Create load thread
        std::thread load_thread(load, ltp);

        // Create processing threads
        std::vector<std::thread> proc_threads;
        for (int i = 0; i < gpu_id.size(); i++) {
            ProcThreadParams ptp;
            ptp.plugin_manager = plugin_managers[i].get();
            ptp.total_elapsed = &total_elapsed;

            for (int j = 0; j < jobs_proc[i]; j++) {
                proc_threads.emplace_back(std::thread(proc, ptp));
            }
        }

        // Set save thread parameters
        SaveThreadParams stp;
        stp.verbose = verbose;
        stp.logger_info = logger_info;
        stp.logger_error = logger_error;

        // Create save threads
        std::vector<std::thread> save_threads;
        for (int i = 0; i < jobs_save; i++) {
            save_threads.emplace_back(std::thread(save, stp));
        }

        // Wait for load thread to finish
        load_thread.join();

        // Send end task
        Task end;
        end.id = -233;

        for (int i = 0; i < total_jobs_proc; i++) {
            to_proc_queue.put(end);
        }

        // Wait for processing threads to finish
        for (auto& thread : proc_threads) {
            thread.join();
        }

        // Send end task to save threads
        for (int i = 0; i < jobs_save; i++) {
            to_save_queue.put(end);
        }

        // Wait for save threads to finish
        for (auto& thread : save_threads) {
            thread.join();
        }

        logger_info->info(std::format("Average processing time: {:.2f}ms/MP", static_cast<double>(total_elapsed) / 1000 / input_files.size()));
    }

    plugin_managers.clear();

    return 0;
}
