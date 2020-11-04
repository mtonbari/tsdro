"""
Row generation algorithm implementation to solve general two-stage
distributionally robust models with a Wasserstein ambiguity set.
"""
from gurobipy import *
import sys
import scenario as sc
from warnings import warn
import numpy as np


class GlobalOpt():

    def __init__(self, tsdro, method="", separation_method="optimization"):
        """
        Parameters
        ----------
        tsdro : TSDRO object
        method : {"RO", ""}
        separation_method : {"enumeration", "optimization"}
        """
        self.tsdro = tsdro
        self.method = method

        self.master = None
        self.stage1_vars = None
        self.stage2_vars = {}
        self.objexpr_master = None

        self.stage2_objvals = {}

        # initialize subproblem
        if tsdro.remaining_scenario_names:
            tmp = tsdro.lr_instance.construct_stage2()
            self.subproblem_model = tmp[0]
            self.subproblem_vars = tmp[1]
            self.subproblem_constrs = tmp[2]
            self.master_scenarios = tsdro.initial_scenario_names
        
        if separation_method == "optimization":
            scenario_sep_output = sc.initialize_scenario_model()
            self.scenario_model = scenario_sep_output[0]
            self.scenario_vars = scenario_sep_output[1]

        self.construct_master(self.tsdro.initial_scenario_names)
        return

    def construct_master(self, master_scenarios):
        """
        Construct master model.

        Included in the model are first stage decisions and constraints, and
        second stage decisions and constraints for each scenario in
        master_scenarios

        Parameters
        ----------
        master_scenarios : list of str
            List of scenario names to be initially included in the master
            model. 
        """
        lr_instance = self.tsdro.lr_instance

        self.master, self.stage1_vars = lr_instance.construct_stage1()
        if self.method == "RO":
            self.wass_mult = 0
        else:
            self.wass_mult = self.master.addVar(name="wass_multiplier", lb=0)
        self.epi_vars = self.master.addVars(self.tsdro.samples.keys(), lb=0,
                                            name="epi_vars")
        objexpr_stage1 = lr_instance.get_objective_stage1(self.stage1_vars)
        objexpr_stage2 = quicksum(
            self.tsdro.probs[sample_name] * self.epi_vars[sample_name]
            for sample_name in self.tsdro.samples.keys())

        self.objexpr_master = (objexpr_stage1
                               + self.tsdro.wass_rad * self.wass_mult
                               + objexpr_stage2)
        self.master.setObjective(self.objexpr_master, GRB.MINIMIZE)

        for scenario_name in master_scenarios:
            scenario = self.tsdro.scenarios[scenario_name]
            curr_vars, _ = lr_instance.add_stage2(
                self.master, self.stage1_vars, scenario, scenario_name)
            for sample_name, sample in self.tsdro.samples.items():
                objexpr_stage2 = lr_instance.get_objective_stage2(curr_vars,
                                                                  scenario)
                if scenario_name == sample_name or self.method == "RO":
                    scenario_distance = 0
                else:
                    scenario_distance = sc.get_scenario_distance(
                        scenario, sample, lr_instance)
                rhs = objexpr_stage2 - self.wass_mult * scenario_distance
                self.master.addLConstr(self.epi_vars[sample_name], ">", rhs,
                                       name=("epi_constr_" + str(scenario_name)
                                             + "_" + str(sample_name)))
            self.stage2_vars[scenario_name] = curr_vars
        return

    def get_stage2_costs(self, stage1_vars, scenario_names):
        """
        Solve and return the optimal value of the second stage problem
        for each scenario in scenario_names, given fixed stage 1 solutions.

        stage1_vars : Stage1Vars dataclass
            Fixed stage 1 solution.
        """
        lr_instance = self.tsdro.lr_instance
        stage2_objvals = {}
        for scenario_name in scenario_names:
            scenario = self.tsdro.scenarios[scenario_name]
            stage1_vars_vals = lr_instance.get_stage1_vals(self.master,
                                                           stage1_vars)
            self.stage2_info, stage2_objval = lr_instance.update_stage2(
                self.subproblem_model, self.subproblem_vars,
                self.subproblem_constrs, stage1_vars_vals, scenario,
                scenario.demands, True)
            if stage2_objval is None:
                sys.exit("Stage 2 is infeasible or unbounded for scenario",
                         scenario_name)
            stage2_objvals[scenario_name] = stage2_objval
        if len(stage2_objvals) == 1:
            scenario_name = scenario_names.pop()
            return stage2_objvals[scenario_name]
        else:
            return stage2_objvals

    def enumeration_separation(self, stage1_vars, sample_name, limit=1,
                               enumerate_all=True):
        """
        Enumerate remaining scenarios and update master if violated constraint if found.

        Parameters
        ----------
        stage1_vars : location_routing.Stage1Vars dataclass
        sample_name : str
        limit : int
            Maximum number of scenarios that can be added to the master
        enumerate_all : bool
            If True, enumerate all scenarios and only add the scenario
            with the greatest violation.
        """
        sample = self.tsdro.samples[sample_name]
        scenarios = self.tsdro.scenarios
        lr_instance = self.tsdro.lr_instance
        curr_remaining = self.tsdro.remaining_scenario_names[sample_name]
        curr_remaining_copy = curr_remaining.copy()

        found_hyp = False
        count = 0
        max_violation_scenario_name = None
        max_violation = -np.inf
        for scenario_name in curr_remaining:
            scenario = scenarios[scenario_name]
            if scenario_name not in self.stage2_objvals:
                self.stage2_objvals[scenario_name] = self.get_stage2_costs(
                    stage1_vars, [scenario_name])

            if scenario_name == sample_name or self.method == "RO":
                scenario_distance = 0
                rhs_val = self.stage2_objvals[scenario_name]
            else:
                scenario_distance = sc.get_scenario_distance(scenario, sample,
                                                             lr_instance)
                rhs_val = (self.stage2_objvals[scenario_name]
                           - self.wass_mult.X * scenario_distance)
            if self.epi_vars[sample_name].X < rhs_val + 1e-7:
                found_hyp = True
                if not enumerate_all:
                    count += 1
                    if scenario_name not in self.master_scenarios:
                        tmp = lr_instance.add_stage2(self.master, self.stage1_vars,
                                                     scenario, scenario_name)
                        self.stage2_vars[scenario_name] = tmp[0]
                        self.master_scenarios.append(scenario_name)
                    objexpr_stage2 = lr_instance.get_objective_stage2(
                        self.stage2_vars[scenario_name], scenario)
                    rhs = objexpr_stage2 - self.wass_mult * scenario_distance
                    self.master.addLConstr(self.epi_vars[sample_name], ">", rhs)
                    curr_remaining_copy.remove(scenario_name)
                else:
                    if rhs_val - self.epi_vars[sample_name].X > max_violation:
                        max_violation = rhs_val - self.epi_vars[sample_name].X
                        max_violation_scenario_name = scenario_name
            if count == limit:
                break
        if enumerate_all and found_hyp:
            scenario = scenarios[max_violation_scenario_name]
            if scenario_name == sample_name or self.method == "RO":
                scenario_distance = 0
            else:
                scenario_distance = sc.get_scenario_distance(scenario, sample,
                                                             lr_instance)
            if scenario_name not in self.master_scenarios:
                tmp = lr_instance.add_stage2(self.master, self.stage1_vars,
                                             scenario,
                                             max_violation_scenario_name)
                self.stage2_vars[max_violation_scenario_name] = tmp[0]
                self.master_scenarios.append(max_violation_scenario_name)
            objexpr_stage2 = lr_instance.get_objective_stage2(
                self.stage2_vars[max_violation_scenario_name], scenario)
            rhs = objexpr_stage2 - self.wass_mult * scenario_distance
            self.master.addLConstr(self.epi_vars[sample_name], ">", rhs)
            curr_remaining_copy.remove(max_violation_scenario_name)
        self.tsdro.remaining_scenario_names[sample_name] = curr_remaining_copy

        if count > 0:
            print(sample_name, "added", count, "scenarios.")
        return found_hyp

    def worst_case_prob_separation(self, stage1_vars):
        scenarios = self.tsdro.scenarios
        samples = self.tsdro.samples
        stage2_objvals = self.get_stage2_costs(stage1_vars, scenarios)
        dro_inner_model, q = self.tsdro.solveInnerDRO(stage2_objvals)
        dro_inner_model.params.OutputFlag = 0
        dro_inner_model.optimize()
        pos_prob_scenarios = []
        for scenario_name in scenarios.keys():
            for sample_name in samples.keys():
                if q[scenario_name, sample_name].X > 0:
                    pos_prob_scenarios.append(scenario_name)
        pos_prob_scenarios = set(pos_prob_scenarios)
        if pos_prob_scenarios == set(self.master_scenarios):
            found_hyp = False
        else:
            # scens = pos_prob_scenarios.union(self.master_scenarios)
            scens = pos_prob_scenarios
            # print(scens - pos_prob_scenarios.intersection(self.master_scenarios))
            # print(len(pos_prob_scenarios), len(scens))
            self.construct_master(scens)
            self.master_scenarios = scens
            found_hyp = True
        return found_hyp

    def optimization_separation(self, stage1_vars, sample_name):
        curr_scenario_model = self.scenario_model.copy()
        curr_scenario_model.optimize()
        if curr_scenario_model.Status != 2:
            warn("Separation not solved to optimality, Status:"
                 + str(curr_scenario_model.Status))
        parsed_solution = sc.parse_scenario_solution(self.scenario_vars)
        scenario = sc.create_single_scenario(*parsed_solution)
        scenario_name = sc.get_scenario_name(scenario)
        # get 2nd stage solution then get objexpr...
        stage2_objval = self.get_stage2_costs(stage1_vars, scenario_name)
        hypograph_val = self.scenario_vars.hypograph_var.X
        if stage2_objval < hypograph_val - 1e-6:
            # get stage2objexpr
            sc.update_scenario_model_row(curr_scenario_model, 
                                         self.scenario_vars,
                                         stage2_objval)
            found_hyp = True
        else:
            found_hyp = False
        return found_hyp

    def solve(self):
        """
        Scenario generation algorithm.
        """
        samples = self.tsdro.samples
        while True:
            self.master.optimize()
            if self.master.Status != 2:
                warn("Master not solved to optimality. Status:"
                     + str(self.master.Status))
                self.master.write("master.lp")
            else:
                print("Master objective:", self.master.ObjVal)
            if self.tsdro.remaining_scenario_names is None:
                break
            found_hyp = {}
            # tmp = [s for sample_name in samples.keys() for s in self.tsdro.remaining_scenario_names[sample_name]]
            # stage2_objvals = self.get_stage2_costs(self.stage1_vars, set(tmp))
            self.stage2_objvals = {}
            for sample_name, sample in samples.items():
                found_hyp[sample_name] = self.enumeration_separation(
                    self.stage1_vars, sample_name)
            # terminate if no separating hyperplanes found

            if not any(found_hyp.values()):
                break
        return
