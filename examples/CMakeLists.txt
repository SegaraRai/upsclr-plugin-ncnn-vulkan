# Examples CMakeLists.txt

# Add the plugin demo example
add_executable(plugin_demo plugin_demo.cpp)

# Link against the realesrgan-lib
target_link_libraries(plugin_demo PRIVATE realesrgan-lib)

# Install the example
install(TARGETS plugin_demo
    RUNTIME DESTINATION bin
)
