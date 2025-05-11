/**
 * @file upsclr_plugin.h
 * @brief Header file defining the API for the plugin DLL
 *
 * This header file defines the plugin API for image upscaling.
 * All string literals are UTF-8 encoded.
 */

#ifndef UPSCLR_PLUGIN_H
#define UPSCLR_PLUGIN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <uchar.h>

#if defined(_WIN32) || defined(_WIN64)
#    ifdef UPSCLR_EXPORTS
#        define UPSCLR_API __declspec(dllexport)
#    else
#        define UPSCLR_API __declspec(dllimport)
#    endif
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#    ifdef UPSCLR_EXPORTS
#        define UPSCLR_API __attribute__((visibility("default")))
#    else
#        define UPSCLR_API
#    endif
#else
#    error "Unsupported platform"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct UpsclrPluginInfo
 * @brief Structure containing information about the plugin itself
 */
typedef struct UpsclrPluginInfo {
    const char8_t* name;        /**< Name of the plugin */
    const char8_t* version;     /**< Version of the plugin */
    const char8_t* description; /**< Description of the plugin */
} UpsclrPluginInfo;

/**
 * @struct UpsclrEngineInfo
 * @brief Structure containing information about an upscale engine
 */
typedef struct UpsclrEngineInfo {
    const char8_t* name;               /**< Name of the engine */
    const char8_t* description;        /**< Description of the engine */
    const char8_t* version;            /**< Version of the engine */
    const char8_t* config_json_schema; /**< JSON schema for engine configuration */
} UpsclrEngineInfo;

/**
 * @struct UpsclrEngineInstance
 * @brief Opaque pointer type for engine instances
 *
 * The actual definition is provided in the implementation file.
 */
typedef struct UpsclrEngineInstance UpsclrEngineInstance;

/**
 * @struct UpsclrEngineConfigValidationResult
 * @brief Structure containing validation results for engine configuration
 */
typedef struct UpsclrEngineConfigValidationResult {
    bool is_valid;                          /**< Whether the configuration is valid */
    size_t error_count;                     /**< Number of errors */
    size_t warning_count;                   /**< Number of warnings */
    const char8_t* const* warning_messages; /**< Array of warning messages */
    const char8_t* const* error_messages;   /**< Array of error messages */
} UpsclrEngineConfigValidationResult;

/**
 * @enum UpsclrErrorCode
 * @brief Enumeration defining error codes for API functions
 */
typedef enum UpsclrErrorCode {
    UPSCLR_SUCCESS = 0,                          /**< Success */
    UPSCLR_ERROR_PLUGIN_NOT_INITIALIZED = 1,     /**< Plugin not initialized */
    UPSCLR_ERROR_PLUGIN_ALREADY_INITIALIZED = 2, /**< Plugin already initialized */
    UPSCLR_ERROR_PLUGIN_ALREADY_DESTROYED = 3,   /**< Plugin already destroyed */
    UPSCLR_ERROR_INVALID_ARGUMENT = 4,           /**< Invalid argument */
    UPSCLR_ERROR_ENGINE_NOT_FOUND = 5,           /**< Engine not found */
    UPSCLR_ERROR_PRELOAD_FAILED = 6,             /**< Preload operation failed */
    UPSCLR_ERROR_UPSCALE_FAILED = 7,             /**< Upscale operation failed */
    UPSCLR_ERROR_OTHER = 9999,                   /**< Other error */
} UpsclrErrorCode;

/**
 * @enum UpsclrColorFormat
 * @brief Enumeration defining color formats for images
 */
typedef enum UpsclrColorFormat {
    UPSCLR_COLOR_FORMAT_RGB = 0, /**< RGB format (Red, Green, Blue) */
    UPSCLR_COLOR_FORMAT_BGR = 1, /**< BGR format (Blue, Green, Red) */
} UpsclrColorFormat;

/**
 * @brief Initialize the plugin
 *
 * @return Error code indicating the result
 */
UPSCLR_API UpsclrErrorCode upsclr_plugin_initialize();

/**
 * @brief Shutdown the plugin
 *
 * @return Error code indicating the result
 */
UPSCLR_API UpsclrErrorCode upsclr_plugin_shutdown();

/**
 * @brief Get information about the plugin
 *
 * @return Pointer to a static structure containing plugin information. NULL on failure.
 */
UPSCLR_API const UpsclrPluginInfo* upsclr_plugin_get_info();

/**
 * @brief Get the number of engines provided by the plugin
 *
 * @return Number of available engines
 */
UPSCLR_API size_t upsclr_plugin_count_engines();

/**
 * @brief Get information about the engine at the specified index
 *
 * @param engine_index Engine index (0-based)
 * @return Pointer to a static structure containing engine information. NULL on failure.
 */
UPSCLR_API const UpsclrEngineInfo* upsclr_plugin_get_engine_info(size_t engine_index);

/**
 * @brief Validate a JSON configuration string for an engine
 *
 * @param engine_index Engine index (0-based)
 * @param config_json Configuration string in JSON format
 * @param config_json_length Length of the configuration string in bytes
 * @return Pointer to a structure containing validation results. Must be freed with `upsclr_free_validation_result` after use.
 */
UPSCLR_API const UpsclrEngineConfigValidationResult* upsclr_plugin_validate_engine_config(size_t engine_index, const char8_t* config_json, size_t config_json_length);

/**
 * @brief Free memory allocated for validation results
 *
 * @param result Validation result obtained from upsclr_validate_engine_config
 */
UPSCLR_API void upsclr_plugin_free_validation_result(const UpsclrEngineConfigValidationResult* result);

/**
 * @brief Create an engine instance
 *
 * @param engine_index Engine index (0-based)
 * @param config_json Configuration string in JSON format
 * @param config_json_length Length of the configuration string in bytes
 * @return Pointer to the created engine instance. NULL on failure. Must be freed with `upsclr_plugin_destroy_engine_instance` after use.
 */
UPSCLR_API UpsclrEngineInstance* upsclr_plugin_create_engine_instance(size_t engine_index, const char8_t* config_json, size_t config_json_length);

/**
 * @brief Destroy an engine instance
 *
 * @param instance Engine instance to destroy
 */
UPSCLR_API void upsclr_plugin_destroy_engine_instance(UpsclrEngineInstance* instance);

/**
 * @brief Preload resources for upscaling
 *
 * @param instance Engine instance
 * @param scale Scale factor
 * @return Error code indicating the result
 */
UPSCLR_API UpsclrErrorCode upsclr_plugin_preload_upscale(UpsclrEngineInstance* instance, int32_t scale);

/**
 * @brief Perform image upscaling
 *
 * The plugin validates the following conditions:
 *
 * - `in_size == in_width * in_height * in_channels`
 *
 * - `out_size == in_width * in_height * in_channels * scale * scale`
 *
 * @param instance Engine instance
 * @param scale Scale factor
 * @param in_data Input image data
 * @param in_size Size of input data in bytes
 * @param in_width Width of input image in pixels
 * @param in_height Height of input image in pixels
 * @param in_channels Number of channels in input image
 * @param in_color_format Color format of input image
 * @param out_data Buffer for output image data
 * @param out_size Size of output buffer in bytes
 * @param out_color_format Color format of output image
 * @return Error code indicating the result
 */
UPSCLR_API UpsclrErrorCode upsclr_plugin_upscale(UpsclrEngineInstance* instance, int32_t scale, const unsigned char* in_data, size_t in_size, uint32_t in_width, uint32_t in_height, uint32_t in_channels, UpsclrColorFormat in_color_format, unsigned char* out_data, size_t out_size, UpsclrColorFormat out_color_format);

#ifdef __cplusplus
}
#endif

#endif
