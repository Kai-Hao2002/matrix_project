#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <functional>
#include <fstream>  
#include "strassen.hpp"
#include "gemm_openblas.hpp"

using Mat = std::vector<std::vector<double>>;

// Custom naive GEMM, parallelized using OpenMP (multi-core version)
void gemm_multi(const Mat &A, const Mat &B, Mat &C) {
    int n = (int)A.size();
    C.assign(n, std::vector<double>(n, 0));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < n; ++k)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

// Single core naive GEMM
void gemm_single(const Mat &A, const Mat &B, Mat &C) {
    int n = (int)A.size();
    C.assign(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

template<typename Func>
void benchmark(const std::string &name, Func f, const Mat &A, const Mat &B, Mat &C) {
    auto start = std::chrono::high_resolution_clock::now();
    f(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    int n = (int)A.size();
    double gflops = 2.0 * n * n * n / (elapsed * 1e9);

    std::cout << name << ": " << elapsed << "s, " << gflops << " GFLOPS" << std::endl;
}
// template<typename Func>
// double benchmark(const std::string& name, Func f, const Mat& A, const Mat& B, Mat& C) {
//     auto start = std::chrono::high_resolution_clock::now();
//     f(A, B, C);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     double time = elapsed.count();

//     std::cout << name << ": " << time << "s\n";
//     return time;  
// }

Mat random_matrix(int n) {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Mat M(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = dist(gen);
    return M;
}

int main() {
    int n = 256;
    Mat A = random_matrix(n);
    Mat B = random_matrix(n);
    Mat C;

    std::cout << "Matrix size: " << n << " x " << n << std::endl;

    
    set_openblas_single_thread();

    // single
    std::cout << "Single cores" << std::endl;

    benchmark("GEMM Single Core", gemm_single, A, B, C);
    benchmark("Strassen-2 Single Core", [](const Mat &A, const Mat &B, Mat &C) {
        strassen(A, B, C, 256, false);
    }, A, B, C);
    benchmark("Strassen-2x2 Single Core", [](const Mat &A, const Mat &B, Mat &C) {
        strassen_2x2(A, B, C, 256, false);
    }, A, B, C);
    benchmark("Multifactor-3x2x2 Single Core", [](const Mat &A, const Mat &B, Mat &C) {
        strassen_3x2x2(A, B, C, 256, false);
    }, A, B, C);
    benchmark("GEMM OpenBLAS Single Core", gemm_openblas, A, B, C);

    std::cout << "Multi cores"  << std::endl;

    // multicores
    benchmark("GEMM 16 Cores", gemm_multi, A, B, C);
    benchmark("Strassen-2 16 Cores", [](const Mat &A, const Mat &B, Mat &C) {
        strassen(A, B, C, 256, true);
    }, A, B, C);
    benchmark("Strassen-2x2 16 Cores", [](const Mat &A, const Mat &B, Mat &C) {
        strassen_2x2(A, B, C, 256, true);
    }, A, B, C);
    benchmark("Multifactor-3x2x2 16 Cores", [](const Mat &A, const Mat &B, Mat &C) {
        strassen_3x2x2(A, B, C, 256, true);
    }, A, B, C);

    
    benchmark("GEMM OpenBLAS", gemm_openblas, A, B, C);

    return 0;
}

// struct Result {
//     int n;
//     std::string method;
//     double time_sec;
//     double gflops;
// };

// int main() {
//     std::vector<int> sizes = {256, 512, 1024, 2048};
//     std::vector<Result> results;

//     for(auto n : sizes) {
//         Mat A = random_matrix(n);
//         Mat B = random_matrix(n);
//         Mat C;

//         std::cout << "Matrix size: " << n << " x " << n << std::endl;

//         auto record = [&](const std::string& name, auto func) {
//             double time = benchmark(name, func, A, B, C);
//             double gflops = 2.0 * n * n * n / (time * 1e9);
//             results.push_back({n, name, time, gflops});
//         };

//         record("GEMM Single Core", gemm_single);
//         record("Strassen-2 Single Core", [](const Mat &A, const Mat &B, Mat &C) {
//             strassen(A, B, C, 256, false);
//         });
//         record("Strassen-2x2 Single Core", [](const Mat &A, const Mat &B, Mat &C) {
//             strassen_2x2(A, B, C, 256, false);
//         });
//         record("Multifactor-3x2x2 Single Core", [](const Mat &A, const Mat &B, Mat &C) {
//             strassen_3x2x2(A, B, C, 256, false);
//         });
//         set_openblas_single_thread();
//         record("GEMM OpenBLAS Single Core", gemm_openblas);

//         record("GEMM 16 Cores", gemm_multi);
//         record("Strassen-2 16 Cores", [](const Mat &A, const Mat &B, Mat &C) {
//             strassen(A, B, C, 256, true);
//         });
//         record("Strassen-2x2 16 Cores", [](const Mat &A, const Mat &B, Mat &C) {
//             strassen_2x2(A, B, C, 256, true);
//         });
//         record("Multifactor-3x2x2 16 Cores", [](const Mat &A, const Mat &B, Mat &C) {
//             strassen_3x2x2(A, B, C, 256, true);
//         });

//         record("GEMM OpenBLAS", gemm_openblas);
//     }

//     std::ofstream ofs("benchmark_results.csv");
//     ofs << "n,method,time_sec,gflops\n";
//     for(auto &r : results) {
//         ofs << r.n << "," << r.method << "," << r.time_sec << "," << r.gflops << "\n";
//     }
//     ofs.close();

//     std::cout << "Benchmark results saved to benchmark_results.csv\n";

//     return 0;
// }