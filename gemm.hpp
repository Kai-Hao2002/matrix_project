#pragma once
#include <vector>

void gemm_single(const std::vector<std::vector<double>> &A,
                 const std::vector<std::vector<double>> &B,
                 std::vector<std::vector<double>> &C, bool parallel = false);

void gemm_multi(const std::vector<std::vector<double>> &A,
                const std::vector<std::vector<double>> &B,
                std::vector<std::vector<double>> &C, bool parallel = true);
