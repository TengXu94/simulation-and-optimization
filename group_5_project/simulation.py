import numpy as np
import time
from Scenario import Scenario
from models import SingleCustomer, LoadBalancer, Server, CustomerGroup, Status, Groups, ServerIDs
from distributions import exponential_rng, homogeneous_poisson_process, homogeneous_poisson_process_variance_reduction


def adjust_time_and_pick_a_movie(arrival_times: list, delta_T: int, G: CustomerGroup, load_balancer: LoadBalancer)-> list:
    """
    Description:
        Create single customers and assign attributes (e.g. movie, server address)
    Args:
        list (float) - arrival times
        int - delta_T, time offset for arrival times
        class - Customer group
        class - Load balancer
    Return:
        list (class) - List of customers
    """
    output = []
    for arrival_time in arrival_times:
        movie = np.random.choice(np.arange(10, step=1), size=None, replace=True, p=G.weights)
        servers_with_movie = check_movie_availability_on_server(movie,load_balancer)
        best_servers = G.best_server_options()
        serverAddress = assign_server(servers_with_movie,best_servers)
        waiting_time = G.distances[serverAddress]
        output.append(SingleCustomer(G.ID, arrival_time + delta_T+waiting_time,\
                                     movie,serverAddress,waiting_time))
    return output

def assign_server(servers_with_movie,best_servers):
    """
    Description:
        Find server assignment for single customer
    Args:
        list (str) - server names that movie is stored on
        list (str) - sorted list of optimal servers
    Return:
        list (class) - List of customers
    """
    for best in best_servers:
        if best in servers_with_movie:
            if best!=np.inf:
                return best
            else:
                raise Exception('Movie not available for customer!')

def check_movie_availability_on_server(movie: int,load_balancer: LoadBalancer):
    """
    Description:
        Find all servers that the selected movie is stored on
    Args:
        int - selected movie by customer
        class - Load balancer
    Return:
        list (str) - server names that movie is stored on
    """        
    servers_with_movie=[]
    if movie in load_balancer.msn.movies_stored:
        servers_with_movie.append(ServerIDs.msn.value)
    if movie in load_balancer.asn_1.movies_stored:
        servers_with_movie.append(ServerIDs.asn_1.value)
    if movie in load_balancer.asn_2.movies_stored:
        servers_with_movie.append(ServerIDs.asn_2.value)
    if servers_with_movie==[]:
        raise Exception('Movie not stored on any server!')
    return servers_with_movie


def update_and_sort_customers_queue(customers: list, current_time: float, times: list):
    """
    Description:
        Updates the time for each new customer(arrival)
    Args:
        list - customers list
        float - current_time
    Return:
        the list of sorted customers (handled customers first)
    """
    
    customers = sorted(customers, key = lambda customer: (customer.time, customer.status))
    
    # last (time, queue)
    current_queue = times[-1][1]

    for customer in customers:
        
        # current time is greater than customer time and
        # customer is not handled
        # then
        # we increment the current_queue and we change the customer status
        if customer.time <= current_time and customer.status != Status.handled.value:
            
            customer.waiting_time += (current_time - customer.time)
            customer.time = current_time
            if not customer.is_waiting:
                current_queue += 1
                customer.is_waiting = True
        elif len(customers) == 1 and customer.time > current_time:
            times.append([current_time, current_queue])
            return customers, times
        else:
            assert (customer.time == current_time) and (customer.status == Status.handled.value)
            times.append([current_time, current_queue])
            return sorted(customers, key = lambda customer: (customer.time, customer.status)), times

def ordered_insert(customers: list, customer: SingleCustomer):
    """
    """
    if len(customers) == 0:
        return [customer]
    # customers should be ordered in time, status
    for index, customer_ in enumerate(customers):
        # if the current customer in the list has time greater than the one considered
        # or if the current customer has the same time but lower priority
        # then insert the customer before this one
        if (customer_.time > customer.time) \
            or (customer.time == customer_.time and customer_.status.value > customer.status.value):
            customers.insert(index,customer)
            return customers


def generate_customers(G1: CustomerGroup, G2: CustomerGroup, G3: CustomerGroup, LB: LoadBalancer, u=np.array([])):
    """
    Description:
        Create happy hour demand from three customer groups
    Args:
        class - Customer group 1
        class - Customer group 2
        class - Customer group 3
        class - Load balancer
    Return:
        list (class) - List of customers
    """ 
    
    # Simulation time in seconds
    T = 20 * 60
    
    requests = []
    
    # Generate Arrivals per group via homogeneous Poisson process based on group specific activity pattern
    for activity_number in range(0,3):
        index = activity_number * 3
        
        # either 0, 20, 40 to make 1 hour of requests
        delta_T = activity_number * 20 * 60
        
        # Time of the events
        if not np.any(u):
            arrival_times_1 = homogeneous_poisson_process(G1.activity_pattern[activity_number], T)
            arrival_times_2 = homogeneous_poisson_process(G2.activity_pattern[activity_number], T)
            arrival_times_3 = homogeneous_poisson_process(G3.activity_pattern[activity_number], T)
        else:
            arrival_times_1 = homogeneous_poisson_process_variance_reduction(G1.activity_pattern[activity_number], T, u[0 + index,:])
            arrival_times_2 = homogeneous_poisson_process_variance_reduction(G2.activity_pattern[activity_number], T, u[1 + index,:])
            arrival_times_3 = homogeneous_poisson_process_variance_reduction(G3.activity_pattern[activity_number], T, u[2 + index,:])
            
        customers_1 = adjust_time_and_pick_a_movie(arrival_times_1, delta_T, G1, LB)
        customers_2 = adjust_time_and_pick_a_movie(arrival_times_2, delta_T, G2, LB)
        customers_3 = adjust_time_and_pick_a_movie(arrival_times_3, delta_T, G3, LB)
    
        merged_customers = customers_1 + customers_2 + customers_3
        merged_customers.sort(key=lambda customers:customers.time)
        
        requests = requests + merged_customers
    return requests

def handle_requests(scenario: Scenario, u=np.array([])):
    """
    Description:
        Simulate customer arrival and request handling process
    Return:
        list (class) - List of served customers
    """ 
    
    # assuming that the ASNs have the same movies stored - Otherwise adjust movies_stored_asn
    load_balancer = LoadBalancer(
        scenario.movies_stored_msn, 
        scenario.movies_stored_asn_1, 
        scenario.movies_stored_asn_2,
        scenario.capacities, 
        scenario.serve_times, 
        scenario.movie_sizes
    )
    
    G1 = CustomerGroup(
        Groups.Group_1.value,
        scenario.popularities[0],
        scenario.activity_patterns[0],
        scenario.distances_g1
    )
    G2 = CustomerGroup(
        Groups.Group_2.value,
        scenario.popularities[1],
        scenario.activity_patterns[1],
        scenario.distances_g2
    )
    G3 = CustomerGroup(
        Groups.Group_3.value,
        scenario.popularities[2],
        scenario.activity_patterns[2],
        scenario.distances_g3
    )
    
    # Generate customers according to customer groups
     
    customers = generate_customers(G1,G2,G3,load_balancer,u) # time to generate customers ~0.4s

    customers_msn = []
    customers_asn_1 = []
    customers_asn_2 = []
    
    ############################
    # BEGIN: Splitting customers per server
    ###########################
    for customer in customers:
        if customer.server_address==ServerIDs.msn.value:
            customers_msn.append(customer)
        elif customer.server_address==ServerIDs.asn_1.value:
            customers_asn_1.append(customer)
        elif customer.server_address==ServerIDs.asn_2.value:
            customers_asn_2.append(customer)
    
    # Sorting to ensure order dependent on time
    customers_msn.sort(key=lambda customers:customers.time)
    customers_asn_1.sort(key=lambda customers:customers.time)
    customers_asn_2.sort(key=lambda customers:customers.time)

    # Get customers per server
    print('{:<30}'.format(f'MSN Customers Length: {len(customers_msn)}'))
    print('{:<30}'.format(f'ASN2 Customers Length: {len(customers_asn_2)}'))
    print('{:<30}'.format(f'ASN1 Customers Length: {len(customers_asn_1)}'))
    #############################
    # END: it takes less than 0.01s
    #############################
    
    if np.any(u):
        results_msn = process_customers(customers_msn, load_balancer, ServerIDs.msn.value, u[9,:])
        results_asn_1 = process_customers(customers_asn_1, load_balancer, ServerIDs.asn_1.value, u[10, :])
        results_asn_2 = process_customers(customers_asn_2, load_balancer, ServerIDs.asn_2.value, u[11,:])
    else:
        start = time.monotonic()
        results_msn = process_customers(customers_msn, load_balancer, ServerIDs.msn.value)
        results_asn_1 = process_customers(customers_asn_1, load_balancer, ServerIDs.asn_1.value)
        results_asn_2 = process_customers(customers_asn_2, load_balancer,ServerIDs.asn_2.value)
        end = time.monotonic()
        print(f'{end-start}seconds to process all customers in three server')

    customers = []
    queues = dict()
    for result in [results_msn, results_asn_1, results_asn_2]:
        group_id = result['server_id']
        queues[group_id] = result['times']
        customers += result['customers_served']
        
    
    # Generate all stats that you want
    return customers, queues
    
        
def process_customers(customers, load_balancer, server_id, u=None):
    """
    Description:
        Process all customers of specified server and calcuate processing times accordingly
    Args:
        list (class) - List of unserved customers
        class - Load balancer
    Return:
        a dict containing:
            list (class) - List of served customers
            dict (<int, int>) - Dict of <time, queue length>
            int - the server id
    """ 
    
    # First customer
    current_time = customers[0].time
    # List of served customers
    customers_served = []
    
    # either the server is busy or not
    server_busy = False
    server_busy_count = 0
    
    counter = -1
    
    queues = 0 
    times = []

    while len(customers):
        c = customers[0]
        # A new request arrives to the server
        # This request has to be "handled"
        # The handle time follows an exponential distribution
        # with the mean of 0.5 second
        if c.status == Status.arrived.value and not server_busy:
            
            
            server_busy_count = 0
            
            c.status = Status.handled.value
            if np.any(u):
                counter += 1
                time_to_handle = exponential_rng(lam=2, u=u[counter])
            else:
                time_to_handle = exponential_rng(lam=2)
            
            if c.time > current_time:
                current_time = c.time

            c.waiting_time += time_to_handle
            current_time += time_to_handle
            c.time += time_to_handle
                
            server_busy = True
            
            if not c.is_waiting:
                queues += 1
                c.is_waiting = True
            times.append([current_time, queues])
            
            # remove the updated customer from the list
            customers.pop(0)
            
            # insert it again in a ordered fashion
            customers = ordered_insert(customers, c)
            #customers = sorted(customers, key = lambda customer: (customer.time, customer.status))

        # The client request was already handled
        # Now we have to serve the movie
        # The serve time is defined in Table 5 + some noise
        # The noise is uniformly distributed between [0.3, 0.7]
        elif c.status == Status.handled.value and server_busy:

            
            movie = c.movie_choice
            group_id = c.id_
            server_id = c.server_address
            time_to_serve = load_balancer.get_serve_time(movie, group_id, server_id)
            time_to_serve += np.random.uniform(0.3, 0.7)
            server_busy_count = 0
            server_busy = False
            
            c.status = Status.served.value
            c.waiting_time += time_to_serve
            
            customers_served.append(c)
            customers.pop(0)
                        
            queues -= 1
            times.append([current_time, queues])
            
            # we don't need sorting here, we are killing the very first element in the list
            #customers = sorted(customers, key = lambda customer: (customer.time, customer.status))
 
        elif c.status == Status.arrived.value and server_busy:

            # A new request arrives but the server is busy
            # update the waiting time and time
            customers, times = update_and_sort_customers_queue(customers, current_time, times)
            queues = times[-1][1]
        if not customers:
            break
    output = dict()
    output['customers_served'] = customers_served
    output['times'] = times
    output['server_id'] = server_id
    return output
