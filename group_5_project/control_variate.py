import numpy as np
import matplotlib.pyplot as plt

# project packages
from models import SingleCustomer, LoadBalancer, Server, CustomerGroup, Status, Groups, ServerIDs, Statistics
from distributions import exponential_rng, homogeneous_poisson_process,homogeneous_poisson_process_variance_reduction

## Simulation
from simulation import handle_requests

## bootstrap
from bootstrap import bootstrap, moving_mean_var, test_bootstrap

## Scenario
from Scenario import Scenario

## utils -> get statistics
from utils import get_queue_statistics, get_statistics

## constants
from constants import MAX_CAPACITY, FIRST_ALLOCATION, SECOND_ALLOCATION

## Parameters for the simulations run: independent run, antithetic run
from SimulationParameters import SimulationParameters

## plot functions
from plot_functions import plot_variable_reduction_results

# ## Variance Reduction

# ## 0. Independent Runs

def independent_runs(allocation: list):
    """
    Description:
        Parameters estimation, benchmark against antithetic runs and control variable
    Args:
        allocation (list) - The list of initial movies allocated to the ASNs
        file_name (str) - custom plot file name
    """
    t = 0
    scenario = Scenario(allocation, allocation)
    simulation_parameters = SimulationParameters()

    var = 0
    mean  = 0
    mean_waiting_time_all = []
    mean_waiting_time_mean_all = []
    mean_waiting_time_var_all = []
    while True: 
        t += 1

        # Run simulation
        results, _ = handle_requests(scenario)

        # Collect statistics
        statistics = get_statistics(results)


        mean_waiting_time = statistics['overall'][Statistics.mean.value]
        mean, var = moving_mean_var(
            mean_waiting_time,\
            mean,\
            var,\
            t
        )
  
        mean_waiting_time_all.append(mean_waiting_time)
        mean_waiting_time_mean_all .append(mean)
        mean_waiting_time_var_all.append(var)

        # Check if necessary precision reached
        if t >= simulation_parameters.run and np.sqrt(var / t) < simulation_parameters.precision:
            break
    
    return np.sqrt(mean_waiting_time_var_all)


# ## 1. Antithetic Runs
def antithetic_runs(allocation: list, scenario=None):
    t = 0
        # this is specified only if called from the optimization side
    if not scenario:
        scenario = Scenario(allocation, allocation)
    simulation_parameters = SimulationParameters()

    var = 0
    mean  = 0
    mean_waiting_time_all = []
    mean_waiting_time_mean_all = []
    mean_waiting_time_var_all = []
    
        
    #For optimization
    max_waiting_time = []
    average_waiting_time = []
    q75_waiting_time = []
    
    for i in np.arange(0, simulation_parameters.run/2): 
        t += 1

        # Run simulation (independent)
        u = np.random.rand(12, 15000)
        results, _ = handle_requests(scenario, u)


        # Run simulation (antithetic)
        u = 1 - u
        results_antithetic, _ = handle_requests(scenario, u)
        
        # Collect statistics
        statistics = get_statistics(results)
        statistics_antithetic = get_statistics(results_antithetic)
        ##################
        # For optimization
        ##################
        max_waiting_time.append(statistics['overall'][Statistics.max_.value])
        average_waiting_time = [statistics['overall'][Statistics.mean.value]]
        q75_waiting_time.append(statistics['overall'][Statistics.q75.value])
        mean_waiting_time = statistics['overall'][Statistics.mean.value]
        mean_waiting_time_antithetic = statistics_antithetic['overall'][Statistics.mean.value]
        
        mean_waiting_time = (mean_waiting_time + mean_waiting_time_antithetic) / 2
        mean, var = moving_mean_var(
            mean_waiting_time,\
            mean,\
            var,\
            t
        )

        mean_waiting_time_all.append(mean_waiting_time)
        mean_waiting_time_mean_all .append(mean)
        mean_waiting_time_var_all.append(var)
    # optimization
    statistics = dict()
    statistics['overall'] = dict()
    statistics['overall'][Statistics.mean.value] = np.mean(average_waiting_time)
    statistics['overall'][Statistics.max_.value] = np.mean(max_waiting_time)
    statistics['overall'][Statistics.q75.value] = np.mean(q75_waiting_time)
    
    return np.sqrt(mean_waiting_time_var_all), statistics


# ## 2. Controlled Mean

def controlled_mean(x, y, mu):
    """ Calculates the controlled mean.
    
    Keywords:
        x (array): Data.
        y (array): Control data.
        mu (float): Scalar expectation of the control data.
    
    Returns:
        avg (float): Controlled mean of the data.
        var (float): Variance of the controlled mean.
        z (array): Optimal linear combination of the data and the control data. 
    """

    cov = np.cov(x, y)
    cov_xy = cov[1, 0]
    var_y = cov[1, 1]
    
    c = -cov_xy / var_y
    z = x + c * (y - mu)
    
    avg = z.mean()
    var = z.var()
    
    return avg, var, z

def control_variate_runs(allocation: list, scenario=None):
    t = 0
    
    # this is specified only if called from the optimization side
    if not scenario:
        scenario = Scenario(allocation, allocation)
    simulation_parameters = SimulationParameters()
    
    var = 0
    mean  = 0
    mean_waiting_time_all = []
    mean_waiting_time_mean_all = []
    mean_waiting_time_var_all = []
    
    
    total_msn_customers_all = []
    mean_waiting_time_control_all = []
    
    
    #For optimization
    max_waiting_time = []
    average_waiting_time = []
    q75_waiting_time = []
    
    #Main loop
    for j in np.arange(simulation_parameters.run):
        t += 1

        # Run simulation
        results, queue = handle_requests(scenario)
        # the maximum queue length for each group
        queue = get_queue_statistics(queue)
        # Collect statistics
        statistics = get_statistics(results)
        
        
        ##################
        # For optimization
        ##################
        max_waiting_time.append(statistics['overall'][Statistics.max_.value])
        average_waiting_time = [statistics['overall'][Statistics.mean.value]]
        q75_waiting_time.append(statistics['overall'][Statistics.q75.value])
        
        mean_waiting_time = statistics['overall'][Statistics.mean.value]
        mean, var = moving_mean_var(
            mean_waiting_time,\
            mean,\
            var,\
            t
        )
  
        mean_waiting_time_all.append(mean_waiting_time)
        mean_waiting_time_mean_all .append(mean)
        mean_waiting_time_var_all.append(var)
        
        total_msn_customers = statistics['overall']['total_msn_customers']

        if j:
            total_msn_customers_all.append(total_msn_customers)
            _, var_control, _ = controlled_mean(
                np.array(mean_waiting_time_all),
                np.array(total_msn_customers_all),
                0.5
            )
            mean_waiting_time_control_all.append(var_control)
            # Control the average waiting time
            print(f'='*47)
            print(f'Variance Control: {np.sqrt(var_control)}')
            print(f'='*47)
        else:
            total_msn_customers_all = [total_msn_customers]
            mean_waiting_time_control_all = [var]

    print(f'Correlation Matrix')
    print(f'{np.corrcoef(total_msn_customers_all, mean_waiting_time_all)}')
    
    # optimization
    statistics = dict()
    statistics['overall'] = dict()
    statistics['overall'][Statistics.mean.value] = np.mean(average_waiting_time)
    statistics['overall'][Statistics.max_.value] = np.mean(max_waiting_time)
    statistics['overall'][Statistics.q75.value] = np.mean(q75_waiting_time)
    
    return np.sqrt(mean_waiting_time_control_all), statistics



if __name__ == '__main__':
    control_variate, _ = control_variate_runs(SECOND_ALLOCATION)
    independent = independent_runs(SECOND_ALLOCATION)
    antithetic, _ = antithetic_runs(SECOND_ALLOCATION)
    plot_variable_reduction_results(
        independent=independent,
        antithetic=antithetic,
        control_variate=control_variate,
        allocation='second_allocation',
        control='total_msn_customers'
    )

    
