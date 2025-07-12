#include "gguf.h" // Use the GGUF API directly for file inspection
#include "ggml.h" // Required for ggml_context and ggml_tensor
#include <iostream>
#include <string>
#include <vector>

// A simple function to escape strings for JSON output.
std::string json_escape(const std::string& s) {
    std::string escaped;
    escaped.reserve(s.length());
    for (char c : s) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b";  break;
            case '\f': escaped += "\\f";  break;
            case '\n': escaped += "\\n";  break;
            case '\r': escaped += "\\r";  break;
            case '\t': escaped += "\\t";  break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    // Handle non-printable control characters
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    escaped += buf;
                } else {
                    escaped += c;
                }
                break;
        }
    }
    return escaped;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        // Output errors as JSON to stderr for consistent parsing by the caller
        std::cerr << "{\"error\": \"Usage: " << argv[0] << " <model_path>\"}" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];

    struct ggml_context * ctx_ggml = nullptr;

    // --- Load GGUF file structure without loading tensor data ---
    // We use `no_alloc = true` to tell gguf to only parse the file
    // header and tensor info, without allocating memory for the tensor data.
    // It also creates a ggml_context required to get tensor metadata.
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_ggml,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), params);

    if (ctx_gguf == nullptr) {
        std::cerr << "{\"error\": \"Failed to load GGUF file structure from " << json_escape(model_path) << "\"}" << std::endl;
        return 1;
    }

    // --- Extract and Print Tensor Info as JSON to stdout ---
    std::cout << "{";

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        // The correct pattern is to get the tensor name from the GGUF context,
        // then use that name to get the tensor metadata from the GGML context.
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx_ggml, name);

        // On the latest versions, the ggml_tensor struct members are directly accessible.
        const enum ggml_type type = tensor->type;
        const int64_t * ne = tensor->ne;

        // In recent llama.cpp versions, the n_dims member was removed from ggml_tensor.
        // We calculate it by finding the first dimension with a size of 1.
        int n_dims = 0;
        for (n_dims = 0; n_dims < GGML_MAX_DIMS; n_dims++) {
            if (ne[n_dims] == 1) {
                break;
            }
        }

        // Print the key (tensor name)
        std::cout << "\"" << json_escape(name) << "\": {";

        // Print tensor type using the correct function for ggml_type
        std::cout << "\"type\": \"" << ggml_type_name(type) << "\",";

        // Print tensor dimensions
        std::cout << "\"dims\": [";
        // GGUF dimensions are stored in reverse order. Print them in standard (non-reversed) order.
        for (int d = 0; d < n_dims; ++d) {
            // Print dimensions in standard (non-reversed) order for clarity.
            std::cout << ne[n_dims - 1 - d];
            if (d < n_dims - 1) {
                std::cout << ",";
            }
        }
        std::cout << "]}";

        if (i < n_tensors - 1) {
            std::cout << ",";
        }
    }

    std::cout << "}" << std::endl;

    // --- Cleanup ---
    gguf_free(ctx_gguf);
    ggml_free(ctx_ggml);

    return 0;
}