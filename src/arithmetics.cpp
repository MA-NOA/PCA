#include "../include/arithmetics.hpp"
#include <cmath>
#include <stdexcept>

double division(double numerator, double denominator) {
  if (denominator != 0) {
    return numerator / denominator;
  }

  throw std::invalid_argument("Attempting to divide by zero");
}

double square_root(double radicand) {
  if (radicand < 0) {
    throw std::domain_error(
        "Attempting to take square root of negative number");
  }
  return std::sqrt(radicand);
}
