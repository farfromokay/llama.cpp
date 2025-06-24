#include "llama.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

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

    // Initialize the llama.cpp backend.
    // false = do not use NUMA optimizations.
    llama_backend_init();

    // --- Load Model Metadata Only ---
    // We use `vocab_only = true` which is the most efficient way to tell
    // llama.cpp to only parse the file header and vocabulary, without
    // loading any tensor data into memory.
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == nullptr) {
        std::cerr << "{\"error\": \"Failed to load model metadata from " << json_escape(model_path) << "\"}" << std::endl;
        llama_backend_free();
        return 1;
    }

    // --- Extract and Print Metadata as JSON to stdout ---
    std::cout << "{";

    const int metadata_count = llama_model_meta_count(model);
    for (int i = 0; i < metadata_count; ++i) {
        char key_buf[256];
        llama_model_meta_key_by_index(model, i, key_buf, sizeof(key_buf));

        char val_buf[2048]; // Use a larger buffer for potentially long values
        llama_model_meta_val_str_by_index(model, i, val_buf, sizeof(val_buf));
        
        std::string key_str(key_buf);
        std::string val_str(val_buf);
        
        // Print the key
        std::cout << "\"" << json_escape(key_str) << "\":";

        // The version of the llama.cpp library being used does not support getting
        // metadata value types. Therefore, we must treat all values as strings and
        // wrap them in quotes for valid JSON. The client application will be
        // responsible for parsing numeric values from these strings.
        std::cout << "\"" << json_escape(val_str) << "\"";

        if (i < metadata_count - 1) {
            std::cout << ",";
        }
    }

    std::cout << "}" << std::endl;

    // --- Cleanup ---
    llama_model_free(model);
    llama_backend_free();

    return 0;
}