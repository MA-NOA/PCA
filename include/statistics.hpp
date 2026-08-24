#ifndef STATISTICS
#define STATISTICS

int calculate_population_number(int initial_value, int final_value);

static inline double brute_force_mean(double data[], int initial, int final);

double brute_force_variance(double data[], double mean, int initial, int final);

double brute_force_covariance(double data_x[], double data_y[], double mean_x,
                              double mean_y, int initial, int final);

#endif
