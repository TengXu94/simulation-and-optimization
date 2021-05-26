import numpy as np
import matplotlib.pyplot as plt

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

def plot_empirical_mean_waiting_time(mean_queue_all, emp_mean, emp_p95, emp_max, filename):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    num_bins = 25
    n, bins, patches = ax.hist(mean_queue_all, num_bins, density=0, label='Draws')
    ax.axvline(emp_mean, label='Mean', color='r')
    ax.axvline(emp_p95, label='95th percentile', color='r', linestyle='--')
    ax.axvline(emp_max, label='Worst Case', color='purple', linestyle='-.')
    ax.set(title='Online Average Waiting Time Simulation',
           xlabel= 'Waiting Time (seconds)',
           ylabel='Frequency')
    plt.legend()
    fig.savefig(f'plots/bootstrap/bootstrap_group={filename}.pdf', dpi=300)
    plt.show()

