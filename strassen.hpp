#pragma once
#include <vector>

void strassen(const std::vector<std::vector<double>> &A,
              const std::vector<std::vector<double>> &B,
              std::vector<std::vector<double>> &C,
              int leaf_size,
              bool parallel);

void strassen_2x2(const std::vector<std::vector<double>> &A,
                  const std::vector<std::vector<double>> &B,
                  std::vector<std::vector<double>> &C,
                  int leaf_size,
                  bool parallel);

void strassen_3x2x2(const std::vector<std::vector<double>> &A,
                    const std::vector<std::vector<double>> &B,
                    std::vector<std::vector<double>> &C,
                    int leaf_size,
                    bool parallel);
