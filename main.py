import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time
import cython

import omp_test as omp_test
import mc_seq as python_seq
import mc_cython_seq as cython_seq
import mc_cython_omp as cython_omp
import mc_cython_simd as cython_simd

start_all = time.perf_counter()

num_sim = 1000
num_days = 252*21
sim_array = []

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
    Run and cout time
"""

start = time.perf_counter()
#test_openmp = omp_test.c_array_f_multi()

#result = python_seq.run(close_list, num_sim, num_days, last_price)
#result = cython_seq.run(close_list, num_sim, num_days, last_price)
#result = cython_omp.run(close_list, num_sim, num_days, last_price)
result = cython_simd.run(close_list, num_sim, num_days, last_price)

total_time = time.perf_counter() - start
print("Time taken for compute: ", total_time)
df_sim_data = pd.DataFrame(result).T
print(df_sim_data)
plt.figure(figsize=(16, 8))
plt.plot(df_sim_data)
plt.axhline(last_price)

total_time_all = time.perf_counter() - start_all
print("Total time: ", total_time_all)
plt.show()


