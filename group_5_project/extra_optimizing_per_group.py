from models import ServerIDs
import numpy as np

from bootstrap import bootstrap, moving_mean_var
from Group_5_Simulation import controlled_mean
from SimulationParameters import SimulationParameters
from Scenario import Scenario
from models import ServerIDs
from simulation import handle_requests
from utils import get_statistics
from plot_functions import plot_variate_reduction_results_for_groups
from constants import FIRST_ALLOCATION, SECOND_ALLOCATION

def get_all_groups_variances(final_statistics: dict, groups: list, variance_key: str):
    """
    """
    
    group_variances = dict()
    for group in groups:
        group_variances[group] = np.sqrt(final_statistics[group][variance_key])
    return group_variances


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
	converged = False
	while not converged: 
	    t += 1

	    # Run simulation
	    results, _ = handle_requests(scenario)

	    # Collect statistics
	    statistics = get_statistics(results)

	    for group in simulation_parameters.groups:

	        par_queue = statistics[group][simulation_parameters.parameter_queue]
	        mean, var = moving_mean_var(
	            par_queue,\
	            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean],\
	            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var],\
	            t
	        )
	        simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean] = mean
	        simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var] = var
	        
	        simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_all].append(par_queue)
	        simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean_all].append(mean)
	        simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var_all].append(var)

	    # Check if necessary precision reached
	    converged = True
	    for group in simulation_parameters.groups:
	        var = simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var_all][-1]
	        converged = converged and (t >= simulation_parameters.run and np.sqrt(var / t) < simulation_parameters.precision)
	    print(f'{t}, {converged}, {var}')

	f_mean = lambda data: np.mean(data)
	for group in simulation_parameters.groups:

	    queue_all = simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_all]
	    bootstrap_mean = bootstrap(np.array(queue_all), f_mean, t)
	    empirical_mean = np.mean(queue_all)
	    print('{:<20} {:<20} {:<20}'.format(f'Group: {group}', f'Bootstrap MSE: {bootstrap_mean}', f'Empirical: {empirical_mean}'))
	    plot_empirical_mean_waiting_time(
	        simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_all],
	        np.mean(simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_all]),
	        np.quantile(simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_all], q=0.95),
	        np.max(simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_all]),
	        f'{file_name}_{group}'
	    )


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
    converged = False
    while not converged: 
        t += 1

        # Run simulation
        results, _ = handle_requests(scenario)

        # Collect statistics
        statistics = get_statistics(results)

        for group in simulation_parameters.groups:

            par_queue = statistics[group][simulation_parameters.parameter_queue]
            mean, var = moving_mean_var(
                par_queue,\
                simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean],\
                simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var],\
                t
            )
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean] = mean
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var] = var
            
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_all].append(par_queue)
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean_all].append(mean)
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var_all].append(var)

        # Check if necessary precision reached
        converged = True
        for group in simulation_parameters.groups:
            var = simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var_all][-1]
            converged = converged and (t >= simulation_parameters.run and np.sqrt(var / t) < simulation_parameters.precision)
        print(f'{t}, {converged}, {var}')
    
    output = get_all_groups_variances(
        simulation_parameters.final_statistics,
        simulation_parameters.groups,
        simulation_parameters.parameter_queue_var_all
    )
    return output


def antithetic_runs(allocation: list):
    t = 0

    simulation_parameters = SimulationParameters()
    scenario = Scenario(allocation, allocation)
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
        for group in simulation_parameters.groups:

            par_queue = (statistics[group][simulation_parameters.parameter_queue] + statistics_antithetic[group][simulation_parameters.parameter_queue]) / 2
            mean, var = moving_mean_var(
                par_queue,\
                simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean],\
                simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var],\
                t
            )
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean] = mean
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var] = var

            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_all].append(par_queue)
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean_all].append(mean)
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var_all].append(var)

    output = get_all_groups_variances(
        simulation_parameters.final_statistics,
        simulation_parameters.groups,
        simulation_parameters.parameter_queue_var_all
    )
    return output


def control_variate_runs(allocation: list):
    t = 0
    scenario = Scenario(allocation, allocation)
    simulation_parameters = SimulationParameters()

    #Main loop
    for j in np.arange(simulation_parameters.run):
        t += 1

        # Run simulation
        results, _ = handle_requests(scenario)

        # Collect statistics
        statistics = get_statistics(results)
        for group in simulation_parameters.groups:
            par_queue = statistics[group][simulation_parameters.parameter_queue]
            mean, var = moving_mean_var(
                par_queue,\
                simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean],\
                simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var],\
                t
            )
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean] = mean
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var] = var
            
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_all].append(par_queue)
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean_all].append(mean)
            simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_var_all].append(var)
            
            
            # Control the average waiting time
            if j:
                simulation_parameters.final_statistics[group]['max_waiting_time_all'].append(statistics['overall']['max'])
                _, var_control, _ = controlled_mean(
                    np.array(simulation_parameters.final_statistics[group][simulation_parameters.parameter_queue_mean_all]),
                    np.array(simulation_parameters.final_statistics[group]['max_waiting_time_all']),
                    0.5
                )
                simulation_parameters.final_statistics[group]['mean_waiting_time_control'].append(var_control)
            else:
                simulation_parameters.final_statistics[group]['max_waiting_time_all'] = [statistics['overall']['max']]
                simulation_parameters.final_statistics[group]['mean_waiting_time_control'] = [var]


    output = get_all_groups_variances(
        simulation_parameters.final_statistics,
        simulation_parameters.groups,
        'mean_waiting_time_control'
    )
    return output


if __name__ == '__main__':

    # independent = independent_runs(FIRST_ALLOCATION)
    # antithetic = antithetic_runs(FIRST_ALLOCATION)
    # controlled_variates = control_variate_runs(FIRST_ALLOCATION)

    # plot_variate_reduction_results_for_groups(
    #     independent,
    #     antithetic,
    #     controlled_variates,
    #     'first_allocation',
    #     ServerIDs.msn.value
    # )

    independent = independent_runs(SECOND_ALLOCATION)
    antithetic= antithetic_runs(SECOND_ALLOCATION)
    controlled_variates = control_variate_runs(SECOND_ALLOCATION)

    plot_variate_reduction_results_for_groups(
        independent,
        antithetic,
        controlled_variates,
        'second_allocation',
        ServerIDs.msn.value
    )
    
