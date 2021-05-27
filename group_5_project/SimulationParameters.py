from models import ServerIDs, Groups

class SimulationParameters:
	"""
	This class contains the main parameters when estimating parameters,
	that is:
		- boostraping
		- variance reduction
	"""

	def __init__(self):
		self.precision = 2
	 
		self.groups = [
	        ServerIDs.msn.value,
	        ServerIDs.asn_1.value,
	        ServerIDs.asn_2.value,
	        Groups.Group_1.value,
	        Groups.Group_2.value,
	        Groups.Group_3.value
	    ]
		self.parameter_queue = 'mean'
		self.parameter_queue_mean = f'{self.parameter_queue}_queue_mean'
		self.parameter_queue_var = f'{self.parameter_queue}_queue_var'
		self.parameter_queue_mean_all = f'{self.parameter_queue}_queue_mean_all'
		self.parameter_queue_var_all = f'{self.parameter_queue}_queue_var_all'
		self.parameter_queue_all = f'{self.parameter_queue}_all'

		self.final_statistics = self.initialize_final_statistics()
		self.run = 1

	def initialize_final_statistics(self)-> dict:
		"""
		Description
		--------------
		The function initializes an empty dictionary
		where statistical variables are stored when
		using:
			- boostraping
			- a variance reduction technique


		Input:
		---------------
		Void

		Output
		---------------
		final_statistics, a dictionary for each group
		"""
		final_statistics = dict()
		for group in self.groups:
		    final_statistics[group] = dict()

		for group in self.groups:
		    final_statistics[group][self.parameter_queue_mean] = 0
		    final_statistics[group][self.parameter_queue_var] = 0
		    final_statistics[group][self.parameter_queue_mean_all] = []
		    final_statistics[group][self.parameter_queue_var_all] = []
		    final_statistics[group][self.parameter_queue_all] = []
		return final_statistics

	def reset_final_statistics(self):
		"""
		Description
		----------------
		The function reset the final_statistics

		Input:
		----------------
		Void

		Output:
		----------------
		Void
		"""
		self.final_statistics = self.initialize_final_statistics()