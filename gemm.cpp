#include "gemm.hpp"
#include <omp.h>

void gemm_single(const std::vector<std::vector<double>> &A,
                 const std::vector<std::vector<double>> &B,
                 std::vector<std::vector<double>> &C, bool parallel) {
    int n = A.size();
    C.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < n; ++k)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

void gemm_multi(const std::vector<std::vector<double>> &A,
                const std::vector<std::vector<double>> &B,
                std::vector<std::vector<double>> &C, bool parallel) {
    int n = A.size();
    C.assign(n, std::vector<double>(n, 0.0));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < n; ++k)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}
