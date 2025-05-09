#include <algorithm>
#include <chrono>
#include <clocale>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <mutex>
#include <queue>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "engines/encoding_utils.hpp"
#include "engines/engine_factory.hpp"

namespace fs = std::filesystem;

#if _WIN32
// image decoder and encoder with wic
#    include "wic_image.h"
#else  // _WIN32
// image decoder and encoder with stb
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
#endif  // _WIN32
#include "webp_image.h"

#if _WIN32
#    include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring) {
    if (optind >= argc || argv[optind][0] != L'-') {
        return -1;
    }

    wchar_t opt = argv[optind][1];
    const wchar_t* p = wcschr(optstring, opt);
    if (p == NULL) {
        return L'?';
    }

    optarg = NULL;

    if (p[1] == L':') {
        optind++;
        if (optind >= argc) {
            return L'?';
        }

        optarg = argv[optind];
    }

    optind++;

    return opt;
}

static std::vector<int> parse_optarg_int_array(const wchar_t* optarg) {
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t* p = wcschr(optarg, L',');
    while (p) {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}
#else                    // _WIN32
#    include <unistd.h>  // getopt()

static std::vector<int> parse_optarg_int_array(const char* optarg) {
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p) {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}
#endif                   // _WIN32

#if _WIN32
using path_t = std::wstring;
#    define PATHSTR(X) L##X
#else
using path_t = std::string;
#    define PATHSTR(X) X
#endif

namespace {

#if _WIN32
std::wstring tow(std::string_view ascii_str) {
    return std::wstring(ascii_str.begin(), ascii_str.end());
}

std::wstring tow(std::u8string_view ascii_str) {
    return std::wstring(ascii_str.begin(), ascii_str.end());
}
#endif

path_t get_file_extension(const path_t& path) {
#if _WIN32
    return fs::path(path).extension().wstring();
#else
    return fs::path(path).extension().string();
#endif
}

path_t get_file_name_without_extension(const path_t& path) {
#if _WIN32
    return fs::path(path).stem().wstring();
#else
    return fs::path(path).stem().string();
#endif
}

bool path_is_directory(const path_t& path) {
    return fs::is_directory(fs::path(path));
}

int list_directory(const path_t& dirpath, std::vector<path_t>& imagepaths) {
    imagepaths.clear();

    try {
        for (const auto& entry : fs::directory_iterator(fs::path(dirpath))) {
            if (entry.is_regular_file()) {
                imagepaths.push_back(entry.path().filename().string<path_t::value_type, std::char_traits<path_t::value_type>, std::allocator<path_t::value_type>>());
            }
        }
        std::ranges::sort(imagepaths);
        return 0;
    } catch (const fs::filesystem_error& e) {
#if _WIN32
        fwprintf(stderr, L"opendir failed %ls: %hs\n", dirpath.c_str(), e.what());
#else
        fprintf(stderr, "opendir failed %s: %s\n", dirpath.c_str(), e.what());
#endif
        return -1;
    }
}

path_t get_executable_directory() {
#if _WIN32
    return fs::path(fs::current_path()).wstring();
#else
    return fs::path(fs::current_path()).string();
#endif
}

bool filepath_is_readable(const path_t& path) {
    try {
        return fs::exists(fs::path(path)) && fs::is_regular_file(fs::path(path));
    } catch (const fs::filesystem_error&) {
        return false;
    }
}

path_t sanitize_filepath(const path_t& path) {
    if (filepath_is_readable(path)) {
        return path;
    }

    return get_executable_directory() + path;
}

void print_usage() {
    fprintf(stderr, "Usage: realesrgan-ncnn-vulkan -i infile -o outfile [options]...\n\n");
    fprintf(stderr, "  -h                   show this help\n");
    fprintf(stderr, "  -i input-path        input image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -o output-path       output image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -s scale             upscale ratio (can be 2, 3, 4. default=4)\n");
    fprintf(stderr, "  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu\n");
    fprintf(stderr, "  -m model-path        folder path to the pre-trained models. default=models\n");
    fprintf(stderr, "  -n model-name        model name (default depends on engine)\n");
    fprintf(stderr, "  -e engine            engine to use (realesrgan, realcugan, default=realesrgan)\n");
    fprintf(stderr, "  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stderr, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stderr, "  -x                   enable tta mode\n");
    fprintf(stderr, "  -f format            output image format (jpg/png/webp, default=ext/png)\n");
    fprintf(stderr, "  -v                   verbose output\n");
    fprintf(stderr, "  -y noise             noise level (-1/0/1/2/3, default=0, only for realcugan)\n");
    fprintf(stderr, "  -z sync-gap          sync gap mode (0/1/2/3, default=0, only for realcugan)\n");

    // Print available engines
    fprintf(stderr, "\nAvailable engines:\n");
    auto engines = SuperResolutionEngineFactory::get_available_engines();
    for (const auto& engine : engines) {
        const auto* info = SuperResolutionEngineFactory::get_engine_info(engine);
        fprintf(stderr, "  %s: %s\n", utf8_to_ascii(engine).c_str(), utf8_to_ascii(info->description).c_str());
        fprintf(stderr, "    Models: ");
        for (size_t i = 0; i < info->model_names.size(); ++i) {
            fprintf(stderr, "%s", utf8_to_ascii(info->model_names[i]).c_str());
            if (i < info->model_names.size() - 1) {
                fprintf(stderr, ", ");
            }
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "    Default model: %s\n", utf8_to_ascii(info->default_model).c_str());
        if (info->supports(SuperResolutionFeatureFlags::NOISE)) {
            fprintf(stderr, "    Default noise: %d\n", info->default_noise);
        }
        fprintf(stderr, "\n");
    }
}

class Task {
   public:
    int id;
    int webp;

    std::chrono::microseconds elapsed;

    path_t inpath;
    path_t outpath;

    ncnn::Mat inimage;
    ncnn::Mat outimage;

    ProcessConfig config;
};

class TaskQueue {
   public:
    TaskQueue() = default;

    void put(const Task& v) {
        std::unique_lock lock(mutex);

        while (tasks.size() >= 8)  // FIXME hardcode queue length
        {
            cv.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        cv.notify_one();
    }

    void get(Task& v) {
        std::unique_lock lock(mutex);

        while (tasks.size() == 0) {
            cv.wait(lock);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        cv.notify_one();
    }

   private:
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<Task> tasks;
};

TaskQueue toproc;
TaskQueue tosave;

class LoadThreadParams {
   public:
    int scale;
    int jobs_load;

    // session data
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;

    std::shared_ptr<spdlog::logger> logger_info;
    std::shared_ptr<spdlog::logger> logger_error;
};

void* load(void* args) {
    const LoadThreadParams* ltp = static_cast<const LoadThreadParams*>(args);
    const int count = ltp->input_files.size();
    const int scale = ltp->scale;

#pragma omp parallel for schedule(static, 1) num_threads(ltp->jobs_load)
    for (int i = 0; i < count; i++) {
        const path_t& imagepath = ltp->input_files[i];

        int webp = 0;

        unsigned char* pixeldata = nullptr;
        int w;
        int h;
        int c;

#if _WIN32
        FILE* fp = _wfopen(imagepath.c_str(), L"rb");
#else
        FILE* fp = fopen(imagepath.c_str(), "rb");
#endif
        if (!fp) {
            continue;
        }

        // read whole file
        std::unique_ptr<unsigned char[]> filedata;
        int length = 0;
        fseek(fp, 0, SEEK_END);
        length = ftell(fp);
        rewind(fp);
        filedata = std::make_unique<unsigned char[]>(length);
        fread(filedata.get(), 1, length, fp);
        fclose(fp);

        // decode image
        if (length > 12 && filedata[0] == 'R' && filedata[8] == 'W') {
            // RIFF____WEBP
            pixeldata = webp_load(filedata.get(), length, &w, &h, &c);
            if (pixeldata) {
                webp = 1;
            }
        }
        if (!pixeldata) {
            // not webp, try jpg png etc.
#if _WIN32
            pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else   // _WIN32
            pixeldata = stbi_load_from_memory(filedata.get(), length, &w, &h, &c, 0);
            if (pixeldata) {
                // stb_image auto channel
                if (c == 1) {
                    // grayscale -> rgb
                    stbi_image_free(pixeldata);
                    pixeldata = stbi_load_from_memory(filedata.get(), length, &w, &h, &c, 3);
                    c = 3;
                } else if (c == 2) {
                    // grayscale + alpha -> rgba
                    stbi_image_free(pixeldata);
                    pixeldata = stbi_load_from_memory(filedata.get(), length, &w, &h, &c, 4);
                    c = 4;
                }
            }
#endif  // _WIN32
        }

        if (!pixeldata) {
            ltp->logger_error->error(PATHSTR("Decode failed: {}"), imagepath);
            continue;
        }

        Task v;
        v.id = i;
        v.inpath = imagepath;
        v.outpath = ltp->output_files[i];

        v.inimage = ncnn::Mat(w, h, static_cast<void*>(pixeldata), (size_t)c, c);
        v.outimage = ncnn::Mat(w * scale, h * scale, (size_t)c, c);

        v.config = ProcessConfig{
            .scale = scale,

#if _WIN32
            .input_format = ColorFormat::BGR,
            .output_format = ColorFormat::BGR,
#else
            .input_format = ColorFormat::RGB,
            .output_format = ColorFormat::RGB,
#endif

            .tile_size = 200,
        };

        if (const auto ext = get_file_extension(v.outpath); c == 4 && (ext == PATHSTR(".jpg") || ext == PATHSTR(".JPG") || ext == PATHSTR(".jpeg") || ext == PATHSTR(".JPEG"))) {
            const path_t output_filename2 = ltp->output_files[i] + PATHSTR(".png");
            v.outpath = output_filename2;
            ltp->logger_error->warn(PATHSTR("Image {} has alpha channel! {} will output {}"), imagepath, imagepath, output_filename2);
        }

        toproc.put(v);
    }

    return 0;
}

class ProcThreadParams {
   public:
    const SuperResolutionEngine* engine;
    std::atomic<long long>* total_elapsed;
    ProcessConfig default_config;
};

void* proc(void* args) {
    const ProcThreadParams* ptp = static_cast<const ProcThreadParams*>(args);
    const SuperResolutionEngine* engine = ptp->engine;
    const auto tile_size = ptp->default_config.tile_size;

    for (;;) {
        Task v;

        toproc.get(v);

        if (v.id == -233) {
            break;
        }

        v.config.tile_size = tile_size;

        const auto tp_begin = std::chrono::steady_clock::now();
        engine->process(v.inimage, v.outimage, v.config);
        const auto tp_end = std::chrono::steady_clock::now();

        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_begin);
        v.elapsed = elapsed;
        ptp->total_elapsed->fetch_add(elapsed.count() * 1'000'000 / (v.inimage.w * v.inimage.h));

        tosave.put(v);
    }

    return 0;
}

class SaveThreadParams {
   public:
    int verbose;

    std::shared_ptr<spdlog::logger> logger_info;
    std::shared_ptr<spdlog::logger> logger_error;
};

void* save(void* args) {
    const SaveThreadParams* stp = static_cast<const SaveThreadParams*>(args);
    const int verbose = stp->verbose;

    for (;;) {
        Task v;

        tosave.get(v);

        if (v.id == -233) {
            break;
        }

        // free input pixel data
        {
            unsigned char* pixeldata = static_cast<unsigned char*>(v.inimage.data);
            if (v.webp == 1) {
                free(pixeldata);
            } else {
#if _WIN32
                free(pixeldata);
#else
                stbi_image_free(pixeldata);
#endif
            }
        }

        const auto ext = get_file_extension(v.outpath);

        // create folder if not exists
        const fs::path fs_path = fs::absolute(v.outpath);
        const std::string parent_path = fs_path.parent_path().string();
        if (fs::exists(parent_path) != 1) {
            stp->logger_info->info("Create folder: {}", parent_path);
            fs::create_directories(parent_path);
        }

        int success = 0;

        if (ext == PATHSTR(".webp") || ext == PATHSTR(".WEBP")) {
            success = webp_save(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, static_cast<const unsigned char*>(v.outimage.data));
        } else if (ext == PATHSTR(".png") || ext == PATHSTR(".PNG")) {
#if _WIN32
            success = wic_encode_image(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data);
#else
            success = stbi_write_png(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data, 0);
#endif
        } else if (ext == PATHSTR(".jpg") || ext == PATHSTR(".JPG") || ext == PATHSTR(".jpeg") || ext == PATHSTR(".JPEG")) {
#if _WIN32
            success = wic_encode_jpeg_image(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data);
#else
            success = stbi_write_jpg(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data, 100);
#endif
        }

        if (success) {
            if (verbose) {
                stp->logger_info->info(PATHSTR("{} -> {} done, took {}us"), v.inpath, v.outpath, v.elapsed.count());
            }
        } else {
            stp->logger_error->error(PATHSTR("Encode failed: {}"), v.outpath);
        }
    }

    return 0;
}

}  // namespace

#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    auto logger_info = spdlog::stdout_color_mt("console");
    auto logger_error = spdlog::stderr_color_mt("stderr");

    path_t inputpath;
    path_t outputpath;
    int scale = 4;
    std::vector<int> tile_size;
    path_t model = PATHSTR("models");
    std::string modelname = "";
    std::string engine_name = "realesrgan";
    int noise = 0;
    int sync_gap = 0;
    std::vector<int> gpu_id;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 2;
    int verbose = 0;
    int tta_mode = 0;
    path_t format = PATHSTR("png");

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:o:s:t:m:n:e:g:j:f:y:z:vxh")) != (wchar_t)-1) {
        switch (opt) {
            case L'i':
                inputpath = optarg;
                break;
            case L'o':
                outputpath = optarg;
                break;
            case L's':
                scale = _wtoi(optarg);
                break;
            case L't':
                tile_size = parse_optarg_int_array(optarg);
                break;
            case L'm':
                model = optarg;
                break;
            case L'n':
                modelname.resize(wcslen(optarg) + 1);
                sprintf_s(&modelname[0], modelname.size(), "%ls", optarg);
                modelname.resize(wcslen(optarg));
                break;
            case L'e':
                engine_name.resize(wcslen(optarg) + 1);
                sprintf_s(&engine_name[0], engine_name.size(), "%ls", optarg);
                engine_name.resize(wcslen(optarg));
                break;
            case L'g':
                gpu_id = parse_optarg_int_array(optarg);
                break;
            case L'j':
                swscanf(optarg, L"%d:%*[^:]:%d", &jobs_load, &jobs_save);
                jobs_proc = parse_optarg_int_array(wcschr(optarg, L':') + 1);
                break;
            case L'f':
                format = optarg;
                break;
            case L'v':
                verbose = 1;
                break;
            case L'x':
                tta_mode = 1;
                break;
            case L'y':
                noise = _wtoi(optarg);
                break;
            case L'z':
                sync_gap = _wtoi(optarg);
                break;
            case L'h':
            default:
                print_usage();
                return -1;
        }
    }
#else   // _WIN32
    int opt;
    while ((opt = getopt(argc, argv, "i:o:s:t:m:n:e:g:j:f:y:z:vxh")) != -1) {
        switch (opt) {
            case 'i':
                inputpath = optarg;
                break;
            case 'o':
                outputpath = optarg;
                break;
            case 's':
                scale = atoi(optarg);
                break;
            case 't':
                tile_size = parse_optarg_int_array(optarg);
                break;
            case 'm':
                model = optarg;
                break;
            case 'n':
                modelname = optarg;
                break;
            case 'e':
                engine_name = optarg;
                break;
            case 'g':
                gpu_id = parse_optarg_int_array(optarg);
                break;
            case 'j':
                sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
                jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
                break;
            case 'f':
                format = optarg;
                break;
            case 'v':
                verbose = 1;
                break;
            case 'x':
                tta_mode = 1;
                break;
            case 'y':
                noise = atoi(optarg);
                break;
            case 'z':
                sync_gap = atoi(optarg);
                break;
            case 'h':
            default:
                print_usage();
                return -1;
        }
    }
#endif  // _WIN32

    if (inputpath.empty() || outputpath.empty()) {
        print_usage();
        return -1;
    }

    if (gpu_id.empty()) {
        if (verbose) {
            logger_info->info("No GPU specified, using default GPU device {}", ncnn::get_default_gpu_index());
        }
        gpu_id.push_back(ncnn::get_default_gpu_index());
    }

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
        logger_error->error("Invalid tile_size: expected 1 or {} tile_size, got {}", gpu_id.size(), tile_size.size());
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
        logger_error->error("Invalid jobs_proc: expected 1 or {} jobs_proc, got {}", gpu_id.size(), jobs_proc.size());
        return -1;
    }

    for (const auto count : jobs_proc) {
        if (count < 1) {
            logger_error->error("Invalid jobs_proc: expected >=1, got {}", count);
            return -1;
        }
    }

    if (!path_is_directory(outputpath)) {
        // guess format from outputpath no matter what format argument specified
        const auto ext = get_file_extension(outputpath);

        if (ext == PATHSTR(".png") || ext == PATHSTR(".PNG")) {
            format = PATHSTR("png");
        } else if (ext == PATHSTR(".webp") || ext == PATHSTR(".WEBP")) {
            format = PATHSTR("webp");
        } else if (ext == PATHSTR(".jpg") || ext == PATHSTR(".JPG") || ext == PATHSTR(".jpeg") || ext == PATHSTR(".JPEG")) {
            format = PATHSTR("jpg");
        } else {
            logger_error->error(PATHSTR("Invalid output path extension: '{}'"), ext);
            return -1;
        }
    }

    if (format != PATHSTR("png") && format != PATHSTR("webp") && format != PATHSTR("jpg")) {
        logger_error->error(PATHSTR("Invalid output format: '{}'"), format);
        return -1;
    }

    // collect input and output filepath
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
    {
        if (path_is_directory(inputpath) && path_is_directory(outputpath)) {
            std::vector<path_t> filenames;
            int lr = list_directory(inputpath, filenames);
            if (lr != 0) {
                return -1;
            }

            const int count = filenames.size();
            input_files.resize(count);
            output_files.resize(count);

            path_t last_filename;
            path_t last_filename_noext;
            for (int i = 0; i < count; i++) {
                path_t filename = filenames[i];
                path_t filename_noext = get_file_name_without_extension(filename);
                path_t output_filename = filename_noext + PATHSTR(".") + format;

                // filename list is sorted, check if output image path conflicts
                if (filename_noext == last_filename_noext) {
                    path_t output_filename2 = filename + PATHSTR(".") + format;
                    logger_error->warn(PATHSTR("Image {} has same name with {}! {} will output {}"), filename, last_filename, filename, output_filename2);
                    output_filename = output_filename2;
                } else {
                    last_filename = filename;
                    last_filename_noext = filename_noext;
                }

                input_files[i] = inputpath + PATHSTR('/') + filename;
                output_files[i] = outputpath + PATHSTR('/') + output_filename;
            }
        } else if (!path_is_directory(inputpath) && !path_is_directory(outputpath)) {
            input_files.push_back(inputpath);
            output_files.push_back(outputpath);
        } else {
            logger_error->error("inputpath and outputpath must be either file or directory at the same time");
            return -1;
        }
    }

    const auto* engine_info = SuperResolutionEngineFactory::get_engine_info(ascii_to_utf8(engine_name));
    if (!engine_info) {
        logger_error->error("Unknown engine: '{}'", engine_name);
        return -1;
    }

    if (modelname.empty()) {
        modelname = utf8_to_ascii(engine_info->default_model);
    }

    if (!std::ranges::any_of(engine_info->model_names, [&modelname](const auto& name) {
            return name == ascii_to_utf8(modelname);
        })) {
        logger_error->error("Unknown model: '{}'", modelname);
        return -1;
    }

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    const auto use_gpu_count = gpu_id.size();

    if (jobs_proc.size() != use_gpu_count) {
        jobs_proc.resize(use_gpu_count, jobs_proc[0]);
    }

    if (tile_size.size() != use_gpu_count) {
        tile_size.resize(use_gpu_count, tile_size[0]);
    }

    const int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    int gpu_count = ncnn::get_gpu_count();
    for (std::size_t i = 0; i < use_gpu_count; i++) {
        if (gpu_id[i] < 0 || gpu_id[i] >= gpu_count) {
            logger_error->error("Invalid gpu device: {} (expected: 0-{})", gpu_id[i], gpu_count - 1);

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (std::size_t i = 0; i < use_gpu_count; i++) {
        const int gpu_queue_count = ncnn::get_gpu_info(gpu_id[i]).compute_queue_count();
        jobs_proc[i] = std::min(jobs_proc[i], gpu_queue_count);
        total_jobs_proc += jobs_proc[i];
    }

    {
        std::vector<std::unique_ptr<SuperResolutionEngine>> engines;

        for (std::size_t i = 0; i < use_gpu_count; i++) {
            SuperResolutionEngineConfig config{
                .model_dir = fs::path(model),
                .model = ascii_to_utf8(modelname),

                .gpu_id = gpu_id[i],
                .tta_mode = tta_mode != 0,
                .num_threads = 1,

                .noise = noise,
                .sync_gap = sync_gap,

                .logger_error = logger_error,
            };

#if _WIN32
            logger_info->info(L"Creating {} instance for gpu_id={}, model={}, model_dir={}", tow(engine_name), config.gpu_id, tow(config.model), config.model_dir.wstring());
#else
            logger_info->info("Creating {} instance for gpu_id={}, model={}, model_dir={}", engine_name, config.gpu_id, config.model, config.model_dir.string());
#endif

            auto engine = SuperResolutionEngineFactory::create_engine(ascii_to_utf8(engine_name), config);

            // Check if the instance was created successfully
            if (engine == nullptr) {
                logger_error->error("Failed to create {} instance for gpu_id={}", engine_name, config.gpu_id);
                return -1;
            }

            engine->preload(scale);

            engines.emplace_back(std::move(engine));
        }

        // main routine
        {
            std::atomic<long long> total_elapsed(0);

            // load image
            LoadThreadParams ltp;
            ltp.scale = scale;
            ltp.jobs_load = jobs_load;
            ltp.input_files = input_files;
            ltp.output_files = output_files;
            ltp.logger_info = logger_info;
            ltp.logger_error = logger_error;

            ncnn::Thread load_thread(load, static_cast<void*>(&ltp));

            // engine proc
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (std::size_t i = 0; i < use_gpu_count; i++) {
                ptp[i].engine = engines[i].get();
                ptp[i].default_config = ProcessConfig();
                ptp[i].default_config.tile_size = engines[i]->get_default_tile_size();
                ptp[i].total_elapsed = &total_elapsed;

                // Override scale if specified
                if (scale > 0) {
                    ptp[i].default_config.scale = scale;
                }

                // Override tile_size if specified
                if (!tile_size.empty() && tile_size[i] > 0) {
                    ptp[i].default_config.tile_size = tile_size[i];
                }
            }

            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (std::size_t i = 0; i < use_gpu_count; i++) {
                    for (int j = 0; j < jobs_proc[i]; j++) {
                        proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, static_cast<void*>(&ptp[i]));
                    }
                }
            }

            // save image
            SaveThreadParams stp;
            stp.verbose = verbose;
            stp.logger_info = logger_info;
            stp.logger_error = logger_error;

            std::vector<ncnn::Thread*> save_threads(jobs_save);
            for (int i = 0; i < jobs_save; i++) {
                save_threads[i] = new ncnn::Thread(save, static_cast<void*>(&stp));
            }

            // end
            load_thread.join();

            Task end;
            end.id = -233;

            for (int i = 0; i < total_jobs_proc; i++) {
                toproc.put(end);
            }

            for (int i = 0; i < total_jobs_proc; i++) {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

            for (int i = 0; i < jobs_save; i++) {
                tosave.put(end);
            }

            for (int i = 0; i < jobs_save; i++) {
                save_threads[i]->join();
                delete save_threads[i];
            }

            logger_info->info(std::format("Average processing time: {:.2f}ms/MP", static_cast<double>(total_elapsed) / 1000 / input_files.size()));
        }

        engines.clear();
    }

    ncnn::destroy_gpu_instance();

    return 0;
}
