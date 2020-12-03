"""
In this script, a set of scenarios are sampled and the two-stage
disaster management problem is solved using Sample Average Approximation. 
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

# Solve Sample Average Approximation
SAAModel, stage1_vars_saa, stage2_vars_saa = lr_instance.construct_saa(
  samples, empirical_probs)
SAAModel.setParam("OutputFlag", 0)
SAAModel.optimize()
print("SAA Optimal Value:", SAAModel.ObjVal)
print("SAA Solution:")
lr_instance.print_stage1_solution(SAAModel, stage1_vars_saa)