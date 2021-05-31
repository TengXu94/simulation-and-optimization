import numpy as np
import time
from models import SingleCustomer
from random import randrange

def generate_customers():
    
    total = np.arange(0, 10**6, 1)
    customers = []
    for i in total:
        customers.append(SingleCustomer(1, np.random.rand(), 2,3,4))
    return customers

def generate_ndarray_customers():
    total = np.arange(0, 10**6, 1)
    customers = []
    for i in total:
        customers.append((1, np.random.rand(), randrange(10),'server',4))
    return np.array(customers, dtype=[('id_', int), ('time', np.float64), ('movie_choice', int), ('server_address', 'S12'), ('waiting_time', np.float64)])

customers = generate_customers()
start = time.monotonic()

sorted(customers, key=lambda x: (x.time, x.status))
end = time.monotonic()
print(f'{end-start}s')


customers_dtype = generate_ndarray_customers()
start = time.monotonic()

np.sort(customers_dtype, order=['time','movie_choice'])
end = time.monotonic()
print(f'{end-start}s DTYPE')
