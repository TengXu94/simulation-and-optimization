"""
Just testing the timing when using different
data structures
"""

import numpy as np
import time
from models import SingleCustomer
from random import randrange, sample
from constants import SINGLE_CUSTOMER_DTYPE

#np.array([(1, 10, 1.123, 2, 2.121, 'dfe', True)], dtype=SINGLE_CUSTOMER_DTYPE)

def generate_customers(sample_size):

    total = np.arange(0, sample_size, 1)
    customers = []
    for i in total:
        customers.append(SingleCustomer(1, np.random.rand(), 2,3,4))
    return customers

def generate_ndarray_customers(sample_size):
    total = np.arange(0, sample_size, 1)
    customers = []
    for i in total:
        customers.append((1, np.random.rand(), randrange(10),'server',4))
    return np.array(customers, dtype=[('id_', int), ('time', np.float64), ('movie_choice', int), ('server_address', 'S12'), ('waiting_time', np.float64)])

customers = generate_customers(10**3)
start = time.monotonic()

sorted(customers, key=lambda x: (x.time, x.status))
end = time.monotonic()
print(f'{end-start}s sorted with 1k')

customers = generate_customers(10**3)
start = time.monotonic()

customers.sort(key=lambda x: (x.time, x.status))
end = time.monotonic()
print(f'{end-start}s sort with 1k')


customers_dtype = generate_ndarray_customers(10**3)
start = time.monotonic()

np.sort(customers_dtype, order=['time','movie_choice'])
end = time.monotonic()
print(f'{end-start}s DTYPE with 1k')






customers = generate_customers(10**6)
start = time.monotonic()

sorted(customers, key=lambda x: (x.time, x.status))
end = time.monotonic()
print(f'{end-start}s sorted with 10M')


customers_dtype = generate_ndarray_customers(10**6)
start = time.monotonic()

np.sort(customers_dtype, order=['time','movie_choice'])
end = time.monotonic()
print(f'{end-start}s DTYPE with 10M')
