import numpy as np
from models import ServerIDs
from constants import MAX_CAPACITY
class Scenario:
    """
    This class represents the simulation Scenario,
    that is, it contains all variables modeling 
    our setting.
    In our case, the server capacities,
    server storage, server serving time, etc.
    """
    
    def __init__(self, asn_allocations_1: list, asn_allocations_2: list):
        
        #[MSN, ASN_1, ASN_2]
        self.capacities = [np.inf, MAX_CAPACITY,MAX_CAPACITY]

        self.movie_sizes = {
            0: 850,
            1: 950,
            2: 1000,
            3: 1200,
            4: 800,
            5: 900,
            6: 1000,
            7: 750,
            8: 700,
            9: 1100
        }

        self.movies_stored_msn = list(range(0, 10))
        self.movies_stored_asn_1 = asn_allocations_1
        self.movies_stored_asn_2 = asn_allocations_2

        # --- Table 2 ---
        # Preferences
        self.popularities = [[2,4,9,8,1,3,5,7,10,6],[6,1,3,4,7,9,2,5,8,10],[4,7,3,6,1,10,2,9,8,5]]

        # --- Table 3 ---
        # Lambdas
        self.activity_patterns = [[0.8,1.2,0.5],[0.9,1.3,0.3],[0.7,1.5,0.4]]

        # --- Table 4 ---
        self.distances_g1 = {ServerIDs.msn.value: 0.5, ServerIDs.asn_1.value: 0.2, ServerIDs.asn_2.value: np.inf}
        self.distances_g2 = {ServerIDs.msn.value: 0.5, ServerIDs.asn_1.value: 0.3, ServerIDs.asn_2.value: 0.4}
        self.distances_g3 = {ServerIDs.msn.value: 0.5, ServerIDs.asn_1.value: np.inf, ServerIDs.asn_2.value: 0.2}

        # --- Table 5 ---
        # [700- 900)
        self.msn_serve_time_900 = {1: 9, 2: 8, 3: 10}
        self.asn_1_serve_time_900 = {1: 3, 2: 4, 3: np.inf}
        self.asn_2_serve_time_900 = {1: np.inf, 2: 5, 3: 4}

        self.serve_times_900 = {
            ServerIDs.msn.value: self.msn_serve_time_900,
            ServerIDs.asn_1.value: self.asn_1_serve_time_900,
            ServerIDs.asn_2.value: self.asn_2_serve_time_900
        }

        # [900- 1100)
        self.msn_serve_time_1100 = {1: 12, 2: 11, 3: 13}
        self.asn_1_serve_time_1100 = {1: 4, 2: 5, 3: np.inf}
        self.asn_2_serve_time_1100 = {1: np.inf, 2: 6, 3: 5}

        self.serve_times_1100 = {
            ServerIDs.msn.value: self.msn_serve_time_1100,
            ServerIDs.asn_1.value: self.asn_1_serve_time_1100,
            ServerIDs.asn_2.value: self.asn_2_serve_time_1100
        }

        # [1100- 1500)
        self.msn_serve_time_1500 = {1: 15, 2: 14, 3: 16}
        self.asn_1_serve_time_1500 = {1: 5, 2: 6, 3: np.inf}
        self.asn_2_serve_time_1500 = {1: np.inf, 2: 7, 3: 6}

        self.serve_times_1500 = {
            ServerIDs.msn.value: self.msn_serve_time_1500,
            ServerIDs.asn_1.value: self.asn_1_serve_time_1500,
            ServerIDs.asn_2.value: self.asn_2_serve_time_1500
        }

        self.serve_times = {
            900: self.serve_times_900,
            1100: self.serve_times_1100,
            1500: self.serve_times_1500
        }