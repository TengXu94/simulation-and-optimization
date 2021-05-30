import numpy as np
import matplotlib.pyplot as plt

def plot_variate_reduction_results_for_groups(
    independent: np.array, 
    antithetic: np.array, 
    control_variate: np.array,
    file_name: str,
    group: str,
):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    ax.plot(independent[group], label='Std. dev. - independent runs')
    ax.plot(np.arange(0,len(independent[group]),step=2), antithetic[group], label='Std. dev.- antithetic runs')
    ax.plot(control_variate[group], label='Std. dev. - control variate runs')
    ax.set(
           xlabel='Epoch',
           ylabel='Average Waiting Time')
    ax.legend()
    fig.savefig(f'plots/extra/variance_reduction_{file_name}.png')
    plt.show()

def plot_variable_reduction_results(
    independent: np.array, 
    antithetic: np.array, 
    control_variate: np.array
):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    ax.plot(independent, label='Std. dev. - independent runs')
    ax.plot(np.arange(0,len(independent),step=2), antithetic, label='Std. dev.- antithetic runs')
    ax.plot(control_variate, label='Std. dev. - control variate runs')
    ax.set(
        xlabel='Epoch',
        ylabel='Average Waiting Time'
    )
    ax.legend()
    fig.savefig('plots/variance_reduction/second_allocation.png', dpi=300)

def plot_queues(queues, title: str, file_name: str):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    
    for key, time_queue in queues.items():
        times = [x[0] for x in time_queue]
        queue = [x[1] for x in time_queue]
        ax.plot(times, queue, label=key)
    ax.vlines([1200,2400], 0, 1, transform=ax.get_xaxis_transform(),linestyles='dashed',colors='k')
    ax.set(title=title,
           xlabel='Time',
           ylabel='Queue length')
    plt.legend()
    fig.savefig(f'plots/queue_time_series_{file_name}.png', dpi=300)

def plot_empirical_waiting_time(mean_queue_all, emp_mean, emp_p95, emp_max, filename):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    num_bins = 25
    n, bins, patches = ax.hist(mean_queue_all, num_bins, density=0, label='Draws')
    ax.axvline(emp_mean, label='Mean', color='r')
    ax.axvline(emp_p95, label='95th percentile', color='r', linestyle='--')
    ax.axvline(emp_max, label='Worst Case', color='purple', linestyle='-.')
    ax.set(
           xlabel= 'Waiting Time (seconds)',
           ylabel='Frequency'
        )
    plt.legend()
    fig.savefig(f'plots/bootstrap/{filename}.png', dpi=300)