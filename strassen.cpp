#include "strassen.hpp"
#include <omp.h>

void naive_mult(const std::vector<std::vector<double>> &A,
                const std::vector<std::vector<double>> &B,
                std::vector<std::vector<double>> &C) {
    int n = A.size();
    C.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

void add(const std::vector<std::vector<double>> &A,
         const std::vector<std::vector<double>> &B,
         std::vector<std::vector<double>> &C) {
    int n = A.size();
    C.assign(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j];
}

void sub(const std::vector<std::vector<double>> &A,
         const std::vector<std::vector<double>> &B,
         std::vector<std::vector<double>> &C) {
    int n = A.size();
    C.assign(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] - B[i][j];
}
#include <iostream>
#include <vector>
static int recursion_level = 0;
void strassen(const std::vector<std::vector<double>> &A,
              const std::vector<std::vector<double>> &B,
              std::vector<std::vector<double>> &C,
              int leaf_size,
              bool parallel) {
    int n = A.size();
    // for (int i = 0; i < recursion_level; ++i) std::cout << "  ";
    // std::cout << "Strassen level " << recursion_level << ", n = " << n << std::endl;

    recursion_level++;
    if (n <= leaf_size) {
        naive_mult(A, B, C);
        recursion_level--;
        return;
    }

    int mid = n / 2;

    // Initialize the size of M1 ~ M7
    std::vector<std::vector<double>> M1(mid, std::vector<double>(mid));
    std::vector<std::vector<double>> M2(mid, std::vector<double>(mid));
    std::vector<std::vector<double>> M3(mid, std::vector<double>(mid));
    std::vector<std::vector<double>> M4(mid, std::vector<double>(mid));
    std::vector<std::vector<double>> M5(mid, std::vector<double>(mid));
    std::vector<std::vector<double>> M6(mid, std::vector<double>(mid));
    std::vector<std::vector<double>> M7(mid, std::vector<double>(mid));

    std::vector<std::vector<double>> A11(mid), A12(mid), A21(mid), A22(mid);
    std::vector<std::vector<double>> B11(mid), B12(mid), B21(mid), B22(mid);

    for (int i = 0; i < mid; ++i) {
        A11[i] = std::vector<double>(A[i].begin(), A[i].begin() + mid);
        A12[i] = std::vector<double>(A[i].begin() + mid, A[i].end());
        A21[i] = std::vector<double>(A[i + mid].begin(), A[i + mid].begin() + mid);
        A22[i] = std::vector<double>(A[i + mid].begin() + mid, A[i + mid].end());

        B11[i] = std::vector<double>(B[i].begin(), B[i].begin() + mid);
        B12[i] = std::vector<double>(B[i].begin() + mid, B[i].end());
        B21[i] = std::vector<double>(B[i + mid].begin(), B[i + mid].begin() + mid);
        B22[i] = std::vector<double>(B[i + mid].begin() + mid, B[i + mid].end());
    }

    auto recurse = [&](auto &A1, auto &A2, auto &B1, auto &B2, auto &M) {
        std::vector<std::vector<double>> AResult, BResult;
        add(A1, A2, AResult);
        add(B1, B2, BResult);
        strassen(AResult, BResult, M, leaf_size, parallel);
    };

    if (parallel) {
        #pragma omp parallel sections num_threads(7)
        {
            #pragma omp section
            recurse(A11, A22, B11, B22, M1);
            #pragma omp section
            recurse(A21, A22, B11, B11, M2);
            #pragma omp section
            recurse(A11, A11, B12, B22, M3);
            #pragma omp section
            recurse(A22, A22, B21, B11, M4);
            #pragma omp section
            recurse(A11, A12, B22, B22, M5);
            #pragma omp section
            recurse(A21, A11, B11, B12, M6);
            #pragma omp section
            recurse(A12, A22, B21, B22, M7);
        }
    } else {
        recurse(A11, A22, B11, B22, M1);
        recurse(A21, A22, B11, B11, M2);
        recurse(A11, A11, B12, B22, M3);
        recurse(A22, A22, B21, B11, M4);
        recurse(A11, A12, B22, B22, M5);
        recurse(A21, A11, B11, B12, M6);
        recurse(A12, A22, B21, B22, M7);
    }

    C.assign(n, std::vector<double>(n));
    for (int i = 0; i < mid; ++i)
        for (int j = 0; j < mid; ++j) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + mid] = M3[i][j] + M5[i][j];
            C[i + mid][j] = M2[i][j] + M4[i][j];
            C[i + mid][j + mid] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    recursion_level--;
}

void strassen_2x2(const std::vector<std::vector<double>> &A,
                  const std::vector<std::vector<double>> &B,
                  std::vector<std::vector<double>> &C,
                  int leaf_size,
                  bool parallel) {
    strassen(A, B, C, leaf_size, parallel);
}

void strassen_3x2x2(const std::vector<std::vector<double>> &A,
                    const std::vector<std::vector<double>> &B,
                    std::vector<std::vector<double>> &C,
                    int leaf_size,
                    bool parallel) {
    strassen(A, B, C, leaf_size, parallel);
}