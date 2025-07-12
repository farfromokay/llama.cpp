# This file defines the build targets for custom tools for the Crux application.

add_executable(gguf-metadata-get gguf-metadata-get.cpp)

target_link_libraries(gguf-metadata-get PRIVATE llama)

add_executable(gguf-tensor-info gguf-tensor-info.cpp)
target_link_libraries(gguf-tensor-info PRIVATE llama)
