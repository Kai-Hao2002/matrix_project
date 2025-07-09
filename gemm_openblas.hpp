#pragma once
#include <vector>

extern "C" {
#include <cblas.h>
void openblas_set_num_threads(int num_threads);
}

using Mat = std::vector<std::vector<double>>;

void set_openblas_single_thread() {
    openblas_set_num_threads(1);
}

using Mat = std::vector<std::vector<double>>;

void gemm_openblas(const Mat &A, const Mat &B, Mat &C) {
    int n = (int)A.size();
    int lda = n, ldb = n, ldc = n;

    // Initialize C to 0
    C.assign(n, std::vector<double>(n, 0.0));

    // Convert A and B into continuous one-dimensional arrays (Row-major)
    std::vector<double> flatA(n * n), flatB(n * n), flatC(n * n, 0.0);

    for (int i = 0; i < n; ++i) {
        std::copy(A[i].begin(), A[i].end(), flatA.begin() + i * n);
        std::copy(B[i].begin(), B[i].end(), flatB.begin() + i * n);
    }

    // Call OpenBLAS cblas_dgemm to do matrix multiplication
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0, flatA.data(), lda,
                flatB.data(), ldb,
                0.0, flatC.data(), ldc);

    // Write the result back to C
    for (int i = 0; i < n; ++i) {
        std::copy(flatC.begin() + i * n, flatC.begin() + (i + 1) * n, C[i].begin());
    }
}
