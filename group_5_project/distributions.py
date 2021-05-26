import numpy as np

def exponential_rng(lam=1.0, u=None):  
    """ 
    Description:
        Generates exponential random number
    Args:
        float - lam, the rate parameter, the inverse expectation of the distribution
    Return:
        float - Exponential random number with given rate
    """
    if not u:
        return -np.log(np.random.rand()) / lam
    else:
        return -np.log(u) / lam

def homogeneous_poisson_process(lam, T):
    """ 
    Description:
        Simulate arrivals via homogeneous Poisson process
    Args:
        float - lam, the rate parameter, the inverse expectation of the distribution
        int - simulation period
    Return:
        list (float) - Arrival times
    """
    arrival_times=[]
    curr_arrival_time=0
    while True:
        curr_arrival_time += exponential_rng(lam)
        if curr_arrival_time > T:
            break
        else:
            arrival_times.append(curr_arrival_time)
    return arrival_times


def homogeneous_poisson_process_variance_reduction(lam, T, u):
    """ 
    Description:
        Simulate arrivals via homogeneous Poisson process
    Args:
        float - lam, the rate parameter, the inverse expectation of the distribution
        int - simulation period
    Return:
        list (float) - Arrival times
    """
    arrival_times=[]
    curr_arrival_time=0
    counter = -1
    while True:
        counter += 1
        curr_arrival_time += exponential_rng(lam, u[counter])
        if curr_arrival_time > T:
            break
        else:
            arrival_times.append(curr_arrival_time)
    return arrival_times
