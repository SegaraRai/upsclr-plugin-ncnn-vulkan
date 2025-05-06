cmake_minimum_required(VERSION 4.0)

# Set C++ standard to C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
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
add_library(upsclr-ncnn-vulkan-lib SHARED
    src/engines/base.cpp
    src/engines/engine_factory.cpp
    src/engines/realesrgan/realesrgan.cpp
    src/engines/realcugan/realcugan.cpp
    #src/upscaler_plugin_api.cpp
    #src/upscaler_registry.cpp
    #src/realesrgan_upscaler.cpp
)

# Set export macro for the library
target_compile_definitions(upsclr-ncnn-vulkan-lib PRIVATE UPSCALER_EXPORTS)

# Add the executable target
add_executable(upsclr-ncnn-vulkan
    src/main.cpp
    src/engines/base.cpp
    src/engines/engine_factory.cpp
    src/engines/realesrgan/realesrgan.cpp
    src/engines/realcugan/realcugan.cpp
)

if(WIN32)
    target_compile_definitions(upsclr-ncnn-vulkan-lib PRIVATE NOMINMAX SPDLOG_WCHAR_TO_UTF8_SUPPORT)
    target_compile_definitions(upsclr-ncnn-vulkan PRIVATE NOMINMAX SPDLOG_WCHAR_TO_UTF8_SUPPORT)

    target_compile_options(upsclr-ncnn-vulkan-lib PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    target_compile_options(upsclr-ncnn-vulkan PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

    target_sources(upsclr-ncnn-vulkan PRIVATE src/win_utf8.cpp)
endif()

find_package(Vulkan REQUIRED)
target_link_libraries(upsclr-ncnn-vulkan PRIVATE Vulkan::Vulkan)
target_link_libraries(upsclr-ncnn-vulkan-lib PRIVATE Vulkan::Vulkan)

find_package(ncnn CONFIG REQUIRED)
target_link_libraries(upsclr-ncnn-vulkan PRIVATE ncnn)
target_link_libraries(upsclr-ncnn-vulkan-lib PRIVATE ncnn)

find_package(Stb REQUIRED)
target_include_directories(upsclr-ncnn-vulkan PRIVATE ${Stb_INCLUDE_DIR})

find_package(WebP CONFIG REQUIRED)
target_link_libraries(upsclr-ncnn-vulkan PRIVATE WebP::webp WebP::webpdecoder WebP::webpdemux)

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(upsclr-ncnn-vulkan PRIVATE spdlog::spdlog_header_only)
target_link_libraries(upsclr-ncnn-vulkan-lib PRIVATE spdlog::spdlog_header_only)

add_dependencies(upsclr-ncnn-vulkan generate-spirv)
add_dependencies(upsclr-ncnn-vulkan-lib generate-spirv)

if(USE_STATIC_MOLTENVK)
    find_library(CoreFoundation NAMES CoreFoundation)
    find_library(Foundation NAMES Foundation)
    find_library(Metal NAMES Metal)
    find_library(QuartzCore NAMES QuartzCore)
    find_library(CoreGraphics NAMES CoreGraphics)
    find_library(Cocoa NAMES Cocoa)
    find_library(IOKit NAMES IOKit)
    find_library(IOSurface NAMES IOSurface)

    target_link_libraries(upsclr-ncnn-vulkan-lib PRIVATE
        ${Metal}
        ${QuartzCore}
        ${CoreGraphics}
        ${Cocoa}
        ${IOKit}
        ${IOSurface}
        ${Foundation}
        ${CoreFoundation}
    )

    target_link_libraries(upsclr-ncnn-vulkan PRIVATE
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
    target_link_libraries(upsclr-ncnn-vulkan PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(upsclr-ncnn-vulkan-lib PRIVATE OpenMP::OpenMP_CXX)
endif()


# Install the library
install(TARGETS upsclr-ncnn-vulkan-lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install the headers
install(FILES
    src/upscaler_plugin_api.h
    DESTINATION include/upsclr-ncnn-vulkan-lib
)

# Add examples directory
#add_subdirectory(examples)
