from models import SingleCustomer
import numpy as np
import time
# we sort on time and then on status
customers = []
for i in np.arange(0, 10**6):
	customers.append(SingleCustomer(1,np.random.rand(),3,4,5))


start = time.monotonic()
customers = sorted(customers, key = lambda customer: (customer.time, customer.status))
end = time.monotonic()

print(f'{end-start} seconds')

customers = []
for i in np.arange(0, 10**6):
	customers.append(SingleCustomer(1,np.random.rand(),3,4,5))
start = time.monotonic()

customers = [list(x.__dict__.items()) for x in customers]
dtype = [('id_', int), ('status', int), ('time', float), ('movie_choice', int), ('waiting_time', float), \
('server_address', int), ('is_waiting', bool)]
customers = np.array(customers, dtype=dtype)
np.sort(customers)


end = time.monotonic()

print(f'{end-start} seconds')
