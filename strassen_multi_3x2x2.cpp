// strassen_multi_3x2x2.cpp
#include "strassen_multi_3x2x2.hpp"
#include "strassen.hpp"  
#include "gemm.hpp"
#include <vector>
#include <omp.h>

using Mat = std::vector<std::vector<double>>;

// Create a zero matrix
Mat zero_matrix(int n) {
    return Mat(n, std::vector<double>(n, 0.0));
}

// Split the large matrix A into sub-matrices starting from (row_start, col_start), size size×size
Mat submatrix(const Mat &A, int row_start, int col_start, int size) {
    Mat sub(size, std::vector<double>(size));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            sub[i][j] = A[row_start + i][col_start + j];
    return sub;
}

// Put the submatrix back into the large matrix C, starting from (row_start, col_start)
void add_submatrix(Mat &C, const Mat &sub, int row_start, int col_start) {
    int size = sub.size();
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            C[row_start + i][col_start + j] += sub[i][j];
}

void strassen_multi_3x2x2(const Mat &A, const Mat &B, Mat &C, int leaf_size, bool parallel) {
    int n = A.size();
    if (n <= leaf_size) {
        gemm_single(A, B, C);  // base case use naive GEMM
        return;
    }

    // Split size
    int size_3 = n / 3;  // Divide into 3×3 blocks
    int size_2 = size_3 / 2;  // Then split the sub-blocks into 2×2

    // Create a temporary sub-matrix result
    std::vector<Mat> results(9, zero_matrix(size_3));  // 3x3 total 9 blocks result

    // Index combination under 3x3 partitioning
    // Traverse 3 blocks of rows and columns for i,j
    // Each sub-matrix size size_3 x size_3
    #pragma omp parallel for collapse(2) if(parallel)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Mat subC = zero_matrix(size_3);

            // Calculate the block C[i,j] = sum_k A[i,k] * B[k,j], k ranges from 0 to 2
            for (int k = 0; k < 3; ++k) {
                Mat subA = submatrix(A, i * size_3, k * size_3, size_3);
                Mat subB = submatrix(B, k * size_3, j * size_3, size_3);
                Mat tempC = zero_matrix(size_3);

                // Here we call the 2x2 Strassen algorithm (implemented in strassen_2x2)
                strassen_2x2(subA, subB, tempC, leaf_size, parallel);

                // Accumulate the results
                for (int x = 0; x < size_3; ++x)
                    for (int y = 0; y < size_3; ++y)
                        subC[x][y] += tempC[x][y];
            }

            // store results
            results[i * 3 + j] = subC;
        }
    }

    //Merge the 9 sub-matrices back to C
    //Clear C
    C.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            add_submatrix(C, results[i * 3 + j], i * size_3, j * size_3);
        }
}
