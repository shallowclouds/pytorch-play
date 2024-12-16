// Use minimal includes to avoid glog dependency
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec256/vec256.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <iostream>
#include <chrono>

// Explicitly use ATen CPU implementations
using namespace at;

// Disable glog
#undef GOOGLE_GLOG_DLL_DECL
#define GOOGLE_GLOG_DLL_DECL

int main() {
    try {
        // Initialize CPU backend only
        at::init_num_threads();
        at::set_num_threads(4);

        // Create CPU tensors
        TensorOptions options = TensorOptions()
            .dtype(kFloat)
            .device(kCPU)
            .requires_grad(false);

        // Create tensors that will use MKL for operations
        auto a = randn({1000, 1000}, options);
        auto b = randn({1000, 1000}, options);

        std::cout << "Created input tensors of size: "
                  << a.sizes() << " and " << b.sizes() << std::endl;

        // Perform matrix multiplication (uses MKL if enabled)
        auto start = std::chrono::high_resolution_clock::now();
        auto c = mm(a, b);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        std::cout << "Matrix multiplication took " << diff.count() << " seconds" << std::endl;

        // Additional operations to verify MKL
        auto d = randn({1000, 1000}, options);

        // Perform multiple operations that benefit from MKL
        start = std::chrono::high_resolution_clock::now();
        auto e = addmm(ones({1000, 1000}, options), a, b);
        auto f = mm(c, d);
        end = std::chrono::high_resolution_clock::now();

        diff = end - start;
        std::cout << "Additional operations took " << diff.count() << " seconds" << std::endl;

        std::cout << "Output tensor sizes:" << std::endl;
        std::cout << "c: " << c.sizes() << std::endl;
        std::cout << "e: " << e.sizes() << std::endl;
        std::cout << "f: " << f.sizes() << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
