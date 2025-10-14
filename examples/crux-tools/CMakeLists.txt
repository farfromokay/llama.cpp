# This file defines the build targets for custom tools for the Crux application.

add_executable(gguf-metadata-dump gguf-metadata-dump.cpp)

target_link_libraries(gguf-metadata-dump PRIVATE llama)

add_executable(gguf-tensor-dump gguf-tensor-dump.cpp)
target_link_libraries(gguf-tensor-dump PRIVATE llama)
