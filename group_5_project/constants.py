import numpy as np

# servers max capacity
MAX_CAPACITY=3500

# movie allocations as described in the simulation
# part
#XXX Find a better name
FIRST_ALLOCATION=[3,4,7,8]
SECOND_ALLOCATION=[2,3,9]

SINGLE_CUSTOMER_DTYPE=[
    ('id_', int),
    ('status', np.float64),
    ('time', np.float64),
    ('movie_choice', int),
    ('waiting_time', np.float64),
    ('server_address', 'S5'),
    ('is_waiting', bool)
]