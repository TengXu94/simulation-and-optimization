import time 
from SingleCustomerCython import SingleCustomerCython
from models import SingleCustomer
import numpy as np


total = np.arange(1,10**6, 1)
customers = []
for i in total:
    customers.append(SingleCustomer(1,np.random.rand(), 2,3,4))
start = time.monotonic()
sorted(customers, key=lambda x: (x.time, x.status))

end = time.monotonic()

print(f'{end-start}s for python')


total = np.arange(1,10**6, 1)
customers = []
for i in total:
    customers.append(SingleCustomerCython(1,np.random.rand(), 2,3,4))
start = time.monotonic()
sorted(customers, key=lambda x: (x.time, x.status))

end = time.monotonic()

print(f'{end-start}s for cython')


total = np.arange(1,10**6, 1)
customers = []
for i in total:
    customers.append(SingleCustomerCython(1,np.random.rand(), 2,3,4))
start = time.monotonic()
np.argsort(customers, order=['time', 'status'])

end = time.monotonic()

print(f'{end-start}s for np argsort')

