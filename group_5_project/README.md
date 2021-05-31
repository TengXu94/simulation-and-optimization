# Description

This folder contains the main notebook with Group 5 project solution.

# Project Code

The main Project Code, where all the simulation and optimization experiments are done, is Group_5_Simulation.ipynb.

# Results

There are two ways to replicate all figures shown in the slide presentation:

- run ./reproduce_all_figures.sh
- go through the jupyter notebook Group_5_project.ipynb

# Packages

- boostrap, helpful functions used in the Boostrap task.
- constants, we store all constant variables here.
- control_variate, all functions that are related to control variate topic are stored here.
- distributions, all functions that are related to distributions are stored here.
- extra_optimizing_per_group, some extra work that could have been interesting to study further. Here we have implemented the control_variate functions able to deal with single servers instead of the overall system.
- models, this file contains multiple classes which help us to represent the simulation environment.
- plot_functions, all plotting functions are stored here.
- simulation, all functions to create our simulation environment are stored here.
- Scenario, our main class which represents and stores all environment variables.
- SimulationParameters, all parameters used and stored when performing bootstraping and variance-reduction techniques are stored here.
- utils, handy self-crafted general-purpose functions that can be used as black-box are stored here.

# Plots

Every figure produced is saved in the 'plots' subfolder.


# Excel Files

Optimization results are saved in the 'excel_files subfolder.