import numpy as np

from Scenario import Scenario
from SimulationParameters import SimulationParameters
from models import Statistics
from plot_functions import plot_empirical_mean_waiting_time
from simulation import handle_requests
from utils import get_statistics

def bootstrap(data, f_statistic, draws):
    """ Calculates the bootstrap mse of a statistic of choice
    
    Keywords:
        data (array): data array.
        f_statistic: function handle calculating the statistic of interest.
        draws (int): number of bootstrap draws.
    
    Returns:
        mse (float): mean square error of the statistic of interest.
    """
    theta = f_statistic(data)
    se = np.zeros((draws, ))
    for r in np.arange(draws):
        data_draw = np.random.choice(data, size=data.shape[0], replace=True)
        theta_emp = f_statistic(data_draw)
        se[r] = (theta_emp -  theta)**2
    mse = se.mean()
    return mse


def moving_mean_var(new_data, old_mean, old_var, t):
    """ Calculates moving sample mean and variance at time t.
    
    Keywords:
        new_data (float): new data point arriving at time t.
        old_mean (float): previous sample mean.
        old_var (float): previous sample variance.
        t (int): time index
    
    Returns:
        new_mean (float): updated sample mean.
        new_var (float): updated sample variance.
    """
    if t == 1:
        new_mean = new_data
        new_var = 0
    else:
        new_mean = old_mean + (new_data - old_mean) / t
        new_var = (1 - 1 / (t - 1)) * old_var + t * (new_mean - old_mean)**2
    return new_mean, new_var

def test_bootstrap(allocation: list, file_name: str):
    """
    Description:
        Parameters estimation using bootstrap
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


    f_mean = lambda data: np.mean(data)

    bootstrap_mean = bootstrap(np.array(mean_waiting_time_all), f_mean, t)
    empirical_mean = np.mean(mean_waiting_time_all)
    print('{:<20} {:<20}'.format(f'Bootstrap MSE: {bootstrap_mean}', f'Empirical: {empirical_mean}'))
    plot_empirical_mean_waiting_time(
        mean_waiting_time_all,
        np.mean(mean_waiting_time_all),
        np.quantile(mean_waiting_time_all, q=0.95),
        np.max(mean_waiting_time_all),
        f'overall_average_waiting_time'
    )

