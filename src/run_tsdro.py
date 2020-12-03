"""
Example script to solve two-stage distributionally robust model of
a disaster management problem using a Wasserstein ambiguity set.
"""

import numpy as np
from numpy.random import RandomState
import scenario as sc
from location_routing import LocationRouting
import simulation as sim
from tsdro_decomposition import TSDRO
from kadaptability import KAdaptability
from global_opt import GlobalOpt
from time import time
import visualization as vis
import os
import matplotlib.pyplot as plt
from gurobipy import *

# Initialize data
import data_process as dat

prng = RandomState(0)

scenarios, true_probs_unscaled = sc.create_scenarios()
total_prob = sum(true_probs_unscaled.values())
true_probs = {s: p / total_prob for s, p in true_probs_unscaled.items()}
scenario_names = scenarios.keys()
print("Total number of scenarios:", len(scenarios))

num_samples = 10
sample_names, empirical_probs = sim.generate_samples(
    scenario_names, true_probs, num_samples, prng=prng)

num_samples = len(sample_names)

# Starting set of scenarios
initial_scenario_names = sample_names

samples = {}
for sample_name in sample_names:
    samples[sample_name] = scenarios[sample_name]


opening_costs = {}
allocation_costs = {}
capacities = {}
nfacility_budget = np.inf  # maximum number of facilities that can be opened
capacity = 1e6             # maximum number of resources that can be stored
item_sizes = {"bundle": 1}
facility_cost = 300000
bundle_cost = 6000
for f in dat.facilities:
    opening_costs[f] = 0 if nfacility_budget < np.inf else facility_cost
    allocation_costs[f, "bundle"] = bundle_cost
    capacities[f] = capacity 
lr_instance = LocationRouting(opening_costs, allocation_costs, capacities,
                              item_sizes, dat.demand_nodes,
                              dat.edge_weight_mult)

# Only include given samples as starting set of scenarios in the master
initial_scenario_names = sample_names
wass_rad = 1e-3  # Wasserstein ball radius

tsdro = TSDRO(lr_instance, scenarios=scenarios,
                  samples=samples, wass_rad=wass_rad,
                  probs=empirical_probs,
                  initial_scenario_names=initial_scenario_names)
globalOpt = GlobalOpt(tsdro)
globalOpt.master.setParam("OutputFlag", 0)
print("Starting solve...")
t_solve_start = time()
globalOpt.solve()
t_solve_end = time() - t_solve_start
print("DRO solved in", t_solve_end, "seconds")
lr_instance.print_stage1_solution(globalOpt.master, globalOpt.stage1_vars)
