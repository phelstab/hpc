import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time

num_sim = 1000
num_days = 252*21
sim_array = []

"""
    Non-HPC part, get data and input
"""
aapl = yf.Ticker("AAPL")
df = aapl.history(period="max")
# filter df from start date 2020-01-01
df = df[df.index >= '2000-01-01']

#df to dict
df_dict = df.to_dict()

# all items from df_dict["Close"] to list
close_list = list(df_dict["Close"].values())

# last traded price
last_price = close_list[-1]


"""
    HPC part, monte carlo simulation
"""

# HPC possible, split the list into chunks and send to different nodes
def get_sum(series):
    _sum = 0
    flops = 0
    for i in range(len(series)):
        _sum += series[i]
        flops += 1
    return _sum, flops

#https://en.wikipedia.org/wiki/Standard_deviation
# HPC possible, split the list into chunks and send to different nodes
def get_standard_deviation(series):
    flops = 0
    # Calculate the mean of the series
    _sum, _f = get_sum(series)
    flops += _f
    _mean = _sum / len(series)
    flops += 1
    # Differences between each element and the mean
    differences = [(x - _mean) ** 2 for x in series]
    flops += len(series)
    # Variance by dividing the sum of the differences by the length of the series
    _sumdiff, _f = get_sum(differences)
    variance = _sumdiff / len(series)
    flops += len(series)
    # Standard deviation, square root of the variance
    std_dev = variance ** 0.5
    flops += 1
    return std_dev, flops

#Alt to: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html
# HPC possible, split the list into chunks and send to different nodes
def get_pct_change(series):
    flops = 0
    pct_changes = []
    for i in range(1, len(series)):
        # Percentage change between the current element and the previous element
        pct_change = (series[i] - series[i-1]) / series[i-1]
        flops += 2
        pct_changes.append(pct_change)  
    return pct_changes, flops


"""
    Main monte-carlo part
"""
start_time = time.perf_counter()
count_flops = 0

returns, f = get_pct_change(close_list)
count_flops += f
vola, f = get_standard_deviation(returns)
count_flops += f

# HPC possible, split the list into chunks and send to different nodes
for x in range(num_sim):
    price_series = []
    count = 0
    
    price = last_price * (1 + np.random.normal(0, vola))  
    count_flops += 1  
    price_series.append(price)

    while count < num_days - 1:
        price = price_series[count] * (1 + np.random.normal(0, vola))
        count_flops += 2
        price_series.append(price)
        count += 1
    sim_array.append(price_series)


# Print results
end_time = time.perf_counter()
print("Time taken for sim: ", end_time - start_time)
print("Total FLOP's (rough estimate): ", count_flops)

"""
    Non-HPC part, output
"""

# sim to df_sim, first dimension is columns, second dimension is rows
df_sim = pd.DataFrame(sim_array).T

plt.figure(figsize=(16, 8))
plt.plot(df_sim)
plt.axhline(last_price)
plt.show()