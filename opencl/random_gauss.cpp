#include "random_gauss.h"
#include <chrono>

namespace gauss {

    RandomGauss::RandomGauss(double mean, double stddev)
        : distribution(mean, stddev)
    {
        std::random_device rd;
        generator.seed(rd());
    }

    double RandomGauss::next()
    {
        return distribution(generator);
    }


    long long get_current_time_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
}