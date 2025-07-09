#pragma once
#include <vector>

void strassen_multi_3x2x2(const std::vector<std::vector<double>> &A,
                          const std::vector<std::vector<double>> &B,
                          std::vector<std::vector<double>> &C,
                          int leaf_size,
                          bool parallel);
