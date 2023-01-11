#include "random_gauss.h"

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
}