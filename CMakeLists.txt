cmake_minimum_required(VERSION 4.0)

# Set C++ standard to C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>")

# Set warning level
cmake_policy(SET CMP0092 NEW)
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/W4>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>")

project(realesrgan-ncnn-vulkan CXX)

option(USE_STATIC_MOLTENVK "link moltenvk static library" OFF)

find_package(Threads)
find_package(OpenMP)

find_package(Vulkan REQUIRED)
if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    message(STATUS "Found glslangValidator: ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
else()
    message(FATAL_ERROR "Vulkan GLSL compiler not found.")
endif()

macro(compile_shader SHADER_SRC)
    set(SHADER_SRC_FULLPATH ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SHADER_SRC})

    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_SPV_HEX_FILE}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
        ARGS -V -s -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_FILE})

    # fp16 storage
    set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")

    set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -V -s -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_fp16s_SPV_HEX_FILE})

    # int8 storage
    set(SHADER_int8s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_int8s")

    set(SHADER_int8s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_int8s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_int8s_SPV_HEX_FILE}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -DNCNN_int8_storage=1 -V -s -x -o ${SHADER_int8s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_int8s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_int8s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_int8s_SPV_HEX_FILE})
endmacro()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# enable global link time optimization
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_supported_output)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO is not supported: ${ipo_supported_output}")
endif()

# look for vulkan compute shader and compile
set(SHADER_SPV_HEX_FILES)

compile_shader(realcugan_4x_postproc_tta.comp.glsl)
compile_shader(realcugan_4x_postproc.comp.glsl)
compile_shader(realcugan_postproc_tta.comp.glsl)
compile_shader(realcugan_postproc.comp.glsl)
compile_shader(realcugan_preproc_tta.comp.glsl)
compile_shader(realcugan_preproc.comp.glsl)

compile_shader(realesrgan_preproc.comp.glsl)
compile_shader(realesrgan_postproc.comp.glsl)
compile_shader(realesrgan_preproc_tta.comp.glsl)
compile_shader(realesrgan_postproc_tta.comp.glsl)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

# Add the library target for the plugin
add_library(upsclr-plugin-ncnn-vulkan SHARED
    src/engines/base.cpp
    src/engines/engine_factory.cpp
    src/engines/realesrgan/realesrgan.cpp
    src/engines/realcugan/realcugan.cpp
    src/plugin/upsclr_plugin.cpp
)

# Add the executable target
add_executable(upsclr-ncnn-vulkan
    src/bin/main.cpp
)

# Find packages
find_package(glaze CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(ncnn CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(WebP CONFIG REQUIRED)
find_package(Stb REQUIRED)

# -- Library target --
target_compile_definitions(upsclr-plugin-ncnn-vulkan PRIVATE UPSCLR_EXPORTS)

target_link_libraries(upsclr-plugin-ncnn-vulkan PRIVATE ncnn)
target_link_libraries(upsclr-plugin-ncnn-vulkan PRIVATE Vulkan::Vulkan)
target_link_libraries(upsclr-plugin-ncnn-vulkan PRIVATE glaze::glaze)
target_link_libraries(upsclr-plugin-ncnn-vulkan PRIVATE spdlog::spdlog_header_only)

target_link_libraries(upsclr-ncnn-vulkan PRIVATE cxxopts::cxxopts)
target_link_libraries(upsclr-ncnn-vulkan PRIVATE glaze::glaze)

if(USE_STATIC_MOLTENVK)
    find_library(CoreFoundation NAMES CoreFoundation)
    find_library(Foundation NAMES Foundation)
    find_library(Metal NAMES Metal)
    find_library(QuartzCore NAMES QuartzCore)
    find_library(CoreGraphics NAMES CoreGraphics)
    find_library(Cocoa NAMES Cocoa)
    find_library(IOKit NAMES IOKit)
    find_library(IOSurface NAMES IOSurface)

    target_link_libraries(upsclr-plugin-ncnn-vulkan PRIVATE
        ${Metal}
        ${QuartzCore}
        ${CoreGraphics}
        ${Cocoa}
        ${IOKit}
        ${IOSurface}
        ${Foundation}
        ${CoreFoundation}
    )
endif()

if(OPENMP_FOUND)
    target_link_libraries(upsclr-plugin-ncnn-vulkan PRIVATE OpenMP::OpenMP_CXX)
endif()

# Build the SPIR-V shaders
add_dependencies(upsclr-ncnn-vulkan generate-spirv)

# -- Executable target --
target_include_directories(upsclr-ncnn-vulkan PRIVATE ${Stb_INCLUDE_DIR})

target_link_libraries(upsclr-ncnn-vulkan PRIVATE WebP::webp WebP::webpdecoder WebP::webpdemux)
target_link_libraries(upsclr-ncnn-vulkan PRIVATE spdlog::spdlog_header_only)
target_link_libraries(upsclr-ncnn-vulkan PRIVATE upsclr-plugin-ncnn-vulkan)

target_compile_definitions(upsclr-plugin-ncnn-vulkan PRIVATE SPDLOG_USE_STD_FORMAT)
target_compile_definitions(upsclr-ncnn-vulkan PRIVATE SPDLOG_USE_STD_FORMAT)

if(OPENMP_FOUND)
    target_link_libraries(upsclr-plugin-ncnn-vulkan PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(upsclr-ncnn-vulkan PRIVATE OpenMP::OpenMP_CXX)
endif()

# UTF-8 support for Windows
if(WIN32)
    target_compile_definitions(upsclr-plugin-ncnn-vulkan PRIVATE NOMINMAX SPDLOG_WCHAR_TO_UTF8_SUPPORT)
    target_compile_definitions(upsclr-ncnn-vulkan PRIVATE NOMINMAX SPDLOG_WCHAR_TO_UTF8_SUPPORT)

    target_sources(upsclr-ncnn-vulkan PRIVATE src/bin/win_utf8.cpp)
endif()

# Install the library
install(TARGETS upsclr-plugin-ncnn-vulkan
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install the headers
install(FILES
    src/plugin/upsclr_plugin.h
    DESTINATION include/upsclr-plugin-ncnn-vulkan
)
