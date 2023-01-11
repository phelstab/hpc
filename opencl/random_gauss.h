#ifndef RANDOM_GAUSS_H
#define RANDOM_GAUSS_H

#include <random>
#include <chrono>
namespace gauss {

    class RandomGauss {
        public:
            RandomGauss(double mean, double stddev);
            double next();
        public:
            std::mt19937 generator;
            std::normal_distribution<double> distribution;
        public:
            static long long get_current_time_ns();
        };
}

#endif