import numpy as np

from models import ServerIDs

def get_queue_statistics(queue: list)->dict:
    """
    Description:
        Calculate queue statistics for each server group
    Args:
        list (class) - List of tuples (time, queue_length)
    Return:
        dict - Statistics of the queues
    """
    output = dict()
    groups = [ServerIDs.msn.value, ServerIDs.asn_2.value, ServerIDs.asn_1.value]
    overall_queue = np.array([])
    for group in groups:
        queue_length = np.array([x[1] for x in queue[group]])
        
        output[group] = dict()
        output[group] = {
            'std': queue_length.std(),
            'max': queue_length.max(),
            'min': queue_length.min(),
            'var': np.var(queue_length)
        }
        overall_queue = np.concatenate([overall_queue, queue_length])
        
    output['overall'] = {
        'std': overall_queue.std(),
        'max': overall_queue.max(),
        'min': overall_queue.min(),
        'var': np.var(overall_queue),
    }
    
    return output



def get_statistics(results: list):
    """
    Description:
        Calculate performance indicators / statistics of process
    Args:
        list (class) - List of served customers
    Return:
        dict - Statistics of process
    """ 
    waiting_times = dict()
    waiting_times[ServerIDs.msn.value] = []
    waiting_times[ServerIDs.asn_1.value] = []
    waiting_times[ServerIDs.asn_2.value] = []
    waiting_times[1] = []
    waiting_times[2] = []
    waiting_times[3] = []
    
    overall = []
    movie_choices_var = []
    total_msn_customers = 0
    for customer in results:
        overall.append(customer.waiting_time)
        movie_choices_var.append(customer.movie_choice)

        if customer.server_address == ServerIDs.msn.value:
            total_msn_customers += 1

        waiting_time = customer.waiting_time
        waiting_times[customer.server_address].append(waiting_time)    
        waiting_times[customer.id_].append(waiting_time)
    
    statistics = dict()
    for key_ in waiting_times.keys():
        waiting_list = np.array(waiting_times[key_])
        statistics[key_] = {
            'mean': waiting_list.mean(),
            'std': waiting_list.std(),
            'max': waiting_list.max(),
            'min': waiting_list.min(),
            'median': np.median(waiting_list),
            'q25': np.quantile(waiting_list, 0.25),
            'q75': np.quantile(waiting_list, 0.75),
            'var': np.var(waiting_list)
        }
    
    overall = np.array(overall)
    statistics['overall'] = dict()
    statistics['overall']['max'] = overall.max()
    statistics['overall']['q75'] = np.quantile(overall, 0.75)
    statistics['overall']['average'] = overall.mean()
    statistics['overall']['total_customers'] = len(overall) #0.19 correlation with average waiting time
    statistics['overall']['movie_choices_var'] = np.var(movie_choices_var) # 0.07
    statistics['overall']['total_msn_customers'] = total_msn_customers #0.55
    return statistics
