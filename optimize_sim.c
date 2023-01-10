#include <immintrin.h>

// Assuming sim_array and randoms are both arrays of floats
// and num_days is an integer

void optimize_sim_array(float* sim_array, float* randoms, int num_days) {
  for (int y = 0; y < num_days; y++) {
    __m256 sim_vec = _mm256_setzero_ps();
    // Get pointer to start of row
    float* sim_row = sim_array + y * num_days;
    for (int x = 0; x < num_days; x += 8) {
      __m256 rand_vec = _mm256_load_ps(randoms + y * num_days + x);
      __m256 result_vec;
      if (x == 0) {
        result_vec = _mm256_add_ps(sim_vec, rand_vec);
      } else {
        result_vec = _mm256_mul_ps(sim_vec, _mm256_add_ps(_mm256_set1_ps(1), rand_vec));
      }
      // Use pointer to access elements of sim_array
      _mm256_store_ps(sim_row + x, result_vec);
      sim_vec = result_vec;
    }
  }
}
