
"""
    HPC part, monte carlo simulation
"""

import random
import math

"""Draw random samples from mean loc and standard deviation scale"""
def rndm_normal(mu, sigma):
    return random.gauss(mu, sigma)

"""
    Draw random samples from mean loc and standard deviation scale. With Box Muller transform
    https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    Inspiration -> https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
"""
def random_normal(mu, sigma):
    U1 = random.uniform(-1, 1)
    U2 = random.uniform(-1, 1)
    W = U1 ** 2 + U2 ** 2
    while W >= 1 or W == 0:
        U1 = random.uniform(-1, 1)
        U2 = random.uniform(-1, 1)
        W = U1 ** 2 + U2 ** 2
    mult = math.sqrt((-2 * math.log(W)) / W)
    X1 = U1 * mult
    X2 = U2 * mult
    return mu + sigma * X1

# HPC possible, split the list into chunks and send to different nodes
def get_sum(series):
    _sum = 0
    for i in range(len(series)):
        _sum += series[i]
    return _sum

#https://en.wikipedia.org/wiki/Standard_deviation
# HPC possible, split the list into chunks and send to different nodes
def get_standard_deviation(series):
    # Calculate the mean of the series
    _sum = get_sum(series)
    _mean = _sum / len(series)
    # Differences between each element and the mean
    differences = [(x - _mean) ** 2 for x in series]
    # Variance by dividing the sum of the differences by the length of the series
    _sumdiff = get_sum(differences)
    variance = _sumdiff / len(series)
    # Standard deviation, square root of the variance
    std_dev = variance ** 0.5
    return std_dev

#Alt to: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html
# HPC possible, split the list into chunks and send to different nodes
def get_pct_change(series):
    pct_changes = []
    for i in range(1, len(series)):
        # Percentage change between the current element and the previous element
        pct_change = (series[i] - series[i-1]) / series[i-1]
        pct_changes.append(pct_change)  
    return pct_changes


"""
    Main monte-carlo part
"""

def run(close_list, num_sim, num_days, last_price):
    returns = get_pct_change(close_list)
    vola = get_standard_deviation(returns)
    sim_array = []
    # HPC possible, split the list into chunks and send to different nodes
    for x in range(num_sim):
        price_series = []
        count = 0        
        price_series.append(last_price * (1 + rndm_normal(0, vola)))
        while count < num_days - 1:
            price_series.append(price_series[count] * (1 + rndm_normal(0, vola)))
            count += 1
        sim_array.append(price_series)

    return sim_array
