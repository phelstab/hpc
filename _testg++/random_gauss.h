#ifndef RANDOM_GAUSS_H
#define RANDOM_GAUSS_H

#include <random>

namespace gauss {
    class RandomGauss {
        public:
            RandomGauss(double mean, double stddev);
            double next();

        public:
            std::mt19937 generator;
            std::normal_distribution<double> distribution;
        };
}

#endif