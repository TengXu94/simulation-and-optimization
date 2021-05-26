
import numpy as np
from enum import Enum

class ServerIDs(Enum):
    """
    Description: 
        Class storing server names
    """
    msn = 'MSN'
    asn_1 = 'ASN1'
    asn_2 = 'ASN2'

class Groups(Enum):
    """
    Description: 
        Class storing group ids
    """
    Group_1 = 1
    Group_2 = 2
    Group_3 = 3

class Statistics(Enum):
    """
    Description: 
        Class storing statistcs ids
    """
    mean = 'average'
    q75 = 'q75'
    max_ = 'max'

    
    
class Status(Enum):
    """
    Description: 
        Class storing status - First Come First Serve (FCFS) principle
    """
    handled = -1
    served = 0
    arrived = np.inf
    
class CustomerGroup:
    """
    Description: 
        Class storing all customer group specific information
    """
    def __init__(self,ID,popularity,activity_pattern,distances):
        self.ID = ID
        self.popularity = popularity
        self.weights=popularity/np.sum(popularity)
        self.activity_pattern = activity_pattern
        self.distances = distances
    
    def best_server_options(self):
        """
        Description: 
            Finding server options
        Return:
            list (str) - sorted list of server options from best to worst
        """
        return sorted(self.distances, key=self.distances.get)
     
class Server:
    """ 
    Description: 
        Class representing the servers {MSN, ASN1, ASN2}
    """
    def __init__(self, id_, movies_stored, capacity, movie_sizes):
        self.id = id_
        self.movies_stored = movies_stored
        self.capacity = capacity
        self.movie_sizes = movie_sizes
        
         # Parameter for checking that the capacity limited of the server is not succeeded
        self.capacity_check = capacity
        self.check_capacity_limit()
        
    def check_capacity_limit(self):
        """
        Description: 
            Check capacity limited is not succeeded
        """
        for movie in self.movies_stored:
            self.capacity_check -= self.movie_sizes[movie]
            if self.capacity_check < 0:
                raise Exception('Server Capacity Exceeded')
        
class LoadBalancer:
    """ 
    Description: 
        Superclass storing the servers and all associated information (e.g. such as serving time)
    """
    def __init__(self, movies_stored_msn, movies_stored_asn_1, movies_stored_asn_2,\
                 capacities, serve_times, movie_sizes):
        self.msn = Server(ServerIDs.msn.value, movies_stored_msn, capacities[0], movie_sizes)
        self.asn_1 = Server(ServerIDs.asn_1.value, movies_stored_asn_1, capacities[1], movie_sizes)
        self.asn_2 = Server(ServerIDs.asn_2.value, movies_stored_asn_2, capacities[2], movie_sizes)
        self.serve_times = serve_times
        self.movie_sizes = movie_sizes
        
    def get_serve_time(self, movie: int, group_id: int, server_id: str):
        """
        Description: 
            Return service time based on input arguments
        Args: 
            int - chosen movie
            int - group_id of customer group \in {1,2,3}
            string - server_id of server \in {'MSN', 'ASN1', 'ASN2'}
        Return:
            Int - serve time
        """
        movie_size = self.movie_sizes[movie]
        if movie_size < 900:
            return self.serve_times[900][server_id][group_id]
            
        elif movie_size >= 900 and movie_size < 1100:
            return self.serve_times[1100][server_id][group_id]
            
        elif movie_size >= 1100:
            return self.serve_times[1500][server_id][group_id]
        else:
            raise Exception(f'Movie Size: {movie_size}')
        
class SingleCustomer:
    """ 
    Description: 
        Class representing single customer
    """
    def __init__(self,id_,time,movie_choice,server_address,waiting_time):
        self.id_ = id_
        self.status = Status.arrived.value
        self.time = time
        self.movie_choice = movie_choice
        self.waiting_time = waiting_time
        self.server_address = server_address
        self.is_waiting = False