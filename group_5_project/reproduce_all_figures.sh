#!/bin/bash

set -e;
set -x;

# Create Folders
mkdir -p plots;
mkdir -p plots/bootstrap;
mkdir -p plots/extra;
mkdir -p plots/optimization;
mkdir -p plots/variance_reduction;

############
# Simulation Figures
############

# Slide 10 Figures
python3 simulation.py;

# Slide 11 Figures
python3 bootstrap.py;

# Slide 12 Figures
python3 extra_optimizing_per_group.py

# Slide 13 Figures
python3 control_variate.py

#############
# Optimization Figures
#############
python3 optimization.py