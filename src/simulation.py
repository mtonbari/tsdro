"""
Functions used to generate samples and simulate solutions.
"""


from gurobipy import *
from numpy.random import RandomState
import numpy as np
import sys


def generate_samples(scenario_names, probs, num_samples, prng=None):
    scenario_names = list(scenario_names)
    if prng is None:
        prng = RandomState()
    if isinstance(probs, dict):
        probs = list(probs.values())
    rand_indices = prng.choice(range(len(scenario_names)), size=int(num_samples),
                              p=probs, replace=True)
    sample_names = []
    for rand_ind in rand_indices:
        sample_names.append(scenario_names[rand_ind])
    unique_sample_names = list(set(sample_names))
    empirical_probs = {}
    for s in unique_sample_names:
        empirical_probs[s] = sample_names.count(s) / len(sample_names)
    return unique_sample_names, empirical_probs


def simulate_stage2(lr_instance, stage1_vars_vals, scenarios, weights=None,
                    stage2_info=None):
    """
    Solve second stage problem for given scenarios to simulate stage 1 solution.
    """
    weighted_costs = []
    stage2_costs = {}
    for i, (scenario_name, scenario) in enumerate(scenarios.items()):
        if stage2_info is None and i == 0:
            stage2_info = lr_instance.construct_stage2(scenario)
        # update and solve stage 2
        stage2_info, stage2_objval = lr_instance.update_stage2(
            stage1_vars_vals, scenario, stage2_info, True)
        if stage2_objval is None:
            sys.exit("Stage 2 is infeasible or unbounded for scenario",
                     scenario_name)
        stage2_costs[scenario_name] = stage2_objval
        if weights is not None:
            weighted_costs.append(weights[scenario_name] * stage2_objval)
    return np.sum(weighted_costs), stage2_costs
