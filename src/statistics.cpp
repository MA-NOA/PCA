#include "../include/statistics.hpp"
#include "../include/arithmetics.hpp"

int calculate_population_number(int initial_value, int final_value) {

  int population_number;

  if (final_value >= initial_value) {
    population_number = final_value - initial_value + 1;
  } else {
    population_number = 0;
  }

  return population_number;
}

double brute_force_mean(double data[], int initial, int final) {

  double sum_x = 0;
  int n = calculate_population_number(initial, final);

  for (int i = initial; i <= final; i++) {
    sum_x += data[i];
  }

  double mean = arithmetics::division(sum_x, n);

  return mean;
}

double brute_force_variance(double data[], double mean, int initial,
                            int final) {

  double sum_x = 0;
  int n = calculate_population_number(initial, final);

  for (int i = initial; i <= final; i++) {
    sum_x += (data[i] - mean) * (data[i] - mean);
  }

  double variance = arithmetics::division(sum_x, n);

  return variance;
}

double brute_force_covariance(double data_x[], double data_y[], double mean_x,
                              double mean_y, int initial, int final) {
  double sum_xy = 0;
  int n = calculate_population_number(initial, final);
  for (int i = initial; i <= final; i++) {
    sum_xy += (data_x[i] - mean_x) * (data_y[i] - mean_y);
  }
  double covariance = arithmetics::division(sum_xy, n);
  return covariance;
}
