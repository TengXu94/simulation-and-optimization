import matplotlib.pyplot as plt


def plot_queues(queues, title: str, file_name: str):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    
    for key, time_queue in queues.items():
        times = [x[0] for x in time_queue]
        queue = [x[1] for x in time_queue]
        ax.plot(times, queue, label=key)
    ax.set(title=title,
           xlabel='Time',
           ylabel='Queue length')
    fig.savefig(f'plots/queue_time_series_{file_name}.png', dpi=300)
    plt.legend()
    plt.show()

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
    fig.savefig(f'plots/bootstrap/{filename}.png', dpi=300)
    plt.show()