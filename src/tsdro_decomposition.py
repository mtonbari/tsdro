import numpy as np
import numpy.linalg as npl
from gurobipy import *
import sys
import scenario as sc


class TSDRO():

    def __init__(self, lr_instance, scenarios=None, samples=None, probs=None,
                 initial_scenario_names=None, wass_rad=None):
        """
        Parameters
        ----------
        lr_instance: LocationRouting() object
        scenarios:  dict
            Maps scenario names (string) to Scenario dataclasses.
        samples: same struture as scenarios.
        probs: dictionary with scenario names as keys and their associated
               probabilties as values. (sample probabilities)
        initial_scenario_names: list of scenario names in initial master
                              (int or String)
        wass_rad: Float64 scalar
        """

        total_probs = sum(probs[sample_name] for sample_name in samples.keys())
        assert np.abs(1 - total_probs) < 1e-6

        self.lr_instance = lr_instance
        self.scenarios = scenarios
        self.samples = samples

        if initial_scenario_names is None:
            self.initial_scenario_names = scenarios.keys()
            self.remaining_scenario_names = None
        else:
            self.initial_scenario_names = initial_scenario_names

            # for global_opt
            # self.remaining_scenario_names = []
            # for scenario_name in scenarios.keys():
            #     if scenario_name not in initial_scenario_names:
            #         self.remaining_scenario_names.append(scenario_name)

            # for global_opt_test
            self.remaining_scenario_names = {}
            for sample_name in samples.keys():
                self.remaining_scenario_names[sample_name] = [
                    scenario_name
                    for scenario_name in scenarios.keys()
                    if scenario_name not in initial_scenario_names]

        self.probs = probs
        self.numScenarios = len(scenarios)
        self.numSamples = len(samples)
        self.wass_rad = wass_rad
        return

    def solve_inner_dro(self, stage2_objvals):
        """
        Solve inner maximization over Wasserstein ball given first stage
        decisions.
        """
        dro_inner_model = Model()
        dro_inner_model.params.Method = 0  # method set to primal simplex
        q = dro_inner_model.addVars(stage2_objvals.keys(), self.samples.keys(),
                                    lb=0)
        lhs = LinExpr()
        for sample_name, sample in self.samples.items():
            curr_tot_prob = quicksum(q[s, sample_name]
                                     for s in stage2_objvals.keys())
            dro_inner_model.addLConstr(curr_tot_prob == 1)
            for scenario_name in stage2_objvals.keys():
                scenario = self.scenarios[scenario_name]
                sampleDev = sc.get_scenario_distance(scenario, sample,
                                                     self.lr_instance)
                lhs.addTerms(sampleDev * self.probs[sample_name],
                             q[scenario_name, sample_name])
        dro_inner_model.addLConstr(lhs, "<", self.wass_rad)

        objExpr = quicksum(
            q[scenario_name, sample_name] * stage2_objvals[scenario_name]
            for scenario_name in stage2_objvals.keys()
            for sample_name in self.samples.keys())
        dro_inner_model.setObjective(objExpr, GRB.MAXIMIZE)
        dro_inner_model.setParam("OutputFlag", 0)
        return dro_inner_model, q
