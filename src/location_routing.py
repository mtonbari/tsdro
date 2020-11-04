from gurobipy import *
import numpy as np
from dataclasses import dataclass
from warnings import warn


@dataclass
class Stage1Vars:
    x: tupledict
    s: tupledict
    excess_order: tupledict = None


@dataclass
class Stage2Vars:
    y: tupledict
    t: tupledict
    demand_viol: tupledict = None


@dataclass
class Stage2Constrs:
    recourse: dict
    scenario: dict
    transport_ub: dict


@dataclass
class Options:
    """
    If charge_stage1 is True, charge_allocation is ignored.
    """
    # First stage options
    charge_excess_orders_only: bool = False
    nfacility_budget: float = np.inf
    stage1_budget: float = 1e9
    charge_stage1: bool = True
    charge_allocation: bool = True
    # Second stage options
    charge_per_unit: bool = True
    is_demand_uncertain: bool = True
    use_direct_links: bool = True


class LocationRouting():
    @staticmethod
    def set_params_extensive_model(model):
        model.setParam("OutputFlag", 0)
        return

    @staticmethod
    def set_params_stage1(model):
        model.setParam("OutputFlag", 0)
        return

    @staticmethod
    def set_params_stage2(model):
        model.setParam("OutputFlag", 0)
        return

    # Dictionary: opening_costs, allocation_costs, capacities, demands, item_sizes
    def __init__(self, opening_costs, allocation_costs, capacities, item_sizes,
                 demand_nodes, link_charge_scale, demands=None, options=None):
        self.opening_costs = opening_costs
        self.allocation_costs = allocation_costs
        self.capacities = capacities
        self.item_sizes = item_sizes
        self.options = Options() if options is None else options
        if demands is None:
            assert not self.options.charge_excess_orders_only
            assert self.options.is_demand_uncertain
            self.demand_penalty = 10 * max(self.allocation_costs.values())
            # self.demand_penalty = 1e12
        else:
            assert not self.options.is_demand_uncertain
            self.demands = demands

        self.facilities = list(opening_costs.keys())
        self.demand_nodes = demand_nodes
        self.items = list(item_sizes.keys())

        self.link_charge_scale = link_charge_scale

        self.max_population = 3e6
        return

    def add_stage1(self, model):
        x = model.addVars(self.facilities, vtype=GRB.BINARY, name="x")
        s = model.addVars(self.facilities, self.items, vtype=GRB.CONTINUOUS,
                          lb=0, name="s")
        if self.options.charge_excess_orders_only:
            excess_order = model.addVars(self.items, lb=0)

        # facility budget
        if self.options.nfacility_budget < np.inf:
            model.addLConstr(quicksum(x) <= self.options.nfacility_budget)

        # capacity constraints
        for f in self.facilities:
            expr = LinExpr(sum(itemSize * s[f, item]
                               for item, itemSize in self.item_sizes.items()))
            model.addLConstr(expr, "<", self.capacities[f] * x[f],
                             name="stage1_capacity_" + str(f))

        if not self.options.is_demand_uncertain:
            # total demand satisfaction
            for item in self.items:
                total_supply = LinExpr(sum(s[f, item] for f in self.facilities))
                total_demand = sum(self.demands[d, item]
                                   for d in self.demand_nodes)
                model.addLConstr(total_supply, ">", total_demand,
                                 name="stage1_demand_" + str(item))
                if self.options.charge_excess_orders_only:
                    model.addConstr(
                        excess_order[item] == total_supply - total_demand)

        if self.options.charge_excess_orders_only:
            stage1_vars = Stage1Vars(x, s, excess_order)
        else:
            stage1_vars = Stage1Vars(x, s)

        # budget
        if self.options.stage1_budget < np.inf:
            stage1_obj = self.get_objective_stage1(stage1_vars, override=True)
            model.addLConstr(stage1_obj <= self.options.stage1_budget)
        return stage1_vars

    def add_stage2(self, model, stage1_vars=None, scenario=None, stage_name="",
                   demand_in_obj=False, t_ub=None):
        t = model.addVars(self.facilities, self.demand_nodes,
                          self.items, vtype=GRB.CONTINUOUS,
                          lb=0, name="t" + str(stage_name))
        y = model.addVars(self.facilities, self.demand_nodes,
                          vtype=GRB.BINARY, name="y" + str(stage_name))
        z = model.addVars(self.facilities, self.items, vtype=GRB.BINARY)

        if self.options.is_demand_uncertain and not demand_in_obj:
            if scenario is not None:
                demands = scenario.demands
            else:
                demands = {(dn, item): 0
                           for dn in self.demand_nodes for item in self.items}
        elif not self.options.is_demand_uncertain and not demand_in_obj:
            demands = self.demands

        if stage1_vars is not None:
            s = stage1_vars.s
        else:
            s = {(f, item): 0 for f in self.facilities for item in self.items}
        if t_ub is None:
            t_ub = {}
            for f in self.facilities:
                for d in self.demand_nodes:
                    for item in self.items:
                        t_ub[f, d, item] = self.capacities[f]

        # facility supply constraint
        # z[f, item] = 1 if demand at facility f is greater than supply
        recourse_constrs = {}
        for f in self.facilities:
            for item in self.items:
                outgoing_resources = LinExpr(
                    sum(t[f, dn, item] for dn in self.demand_nodes))
                recourse_constrs[f, item, "unaffected"] = model.addLConstr(
                    outgoing_resources, "<", s[f, item],
                    name=("stage2_s" + str(stage_name) + "_budget_" + str(f)
                          + "_" + str(item)))
                if f in self.demand_nodes:
                    recourse_constrs[f, item, "affected"] = model.addLConstr(
                        outgoing_resources - self.max_population * z[f, item],
                        "<",
                        s[f, item] - demands[f, item]
                    )
                    recourse_constrs[f, item, "zero_supply"] = model.addLConstr(
                        z[f, item],
                        ">=",
                        (demands[f, item] - s[f, item]) / self.max_population
                        )

        # demand satisfaction
        if self.options.is_demand_uncertain and not demand_in_obj:
            demand_viol = model.addVars(self.demand_nodes, self.items, lb=0,
                                        name="dv" + str(stage_name))
        else:
            # assign 0 to all elements of demand_viol
            tempKeys = [(d, i) for d in self.demand_nodes for i in self.items]
            demand_viol = dict.fromkeys(tempKeys, 0)
        scenario_constrs = {}
        if not demand_in_obj:
            for d in self.demand_nodes:
                for item in self.items:
                    supply = LinExpr(quicksum(t[f, d, item]
                                              for f in self.facilities))
                    lhs = supply + demand_viol[d, item]
                    scenario_constrs[d, item] = model.addLConstr(
                        lhs, ">", demands[d, item],
                        name=("stage2_s" + str(stage_name) + "_demand_"
                              + str(d) + "_" + str(item)))

        # big-M constraints
        transport_ub_constrs = {}
        for f in self.facilities:
            for dn in self.demand_nodes:
                for item in self.items:
                    transport_ub_constrs[f, d, item] = model.addLConstr(
                        t[f, dn, item], "<", t_ub[f, d, item] * y[f, d],
                        name=("stage2_s" + str(stage_name)
                              + "_bigM_" + str(f) + "_" + str(d)))
                    model.addLConstr(t[f, dn, item], "<",
                                     t_ub[f, dn, item] * (1 - z[f, item]))
        stage2_vars = Stage2Vars(y, t, demand_viol)
        stage2_constrs = Stage2Constrs(recourse_constrs, scenario_constrs,
                                       transport_ub_constrs)
        return stage2_vars, stage2_constrs

    def update_stage2_recourse(self, recourse_constrs, stage1_vars, scenario):
        demands = scenario.demands
        for f in self.facilities:
            for item in self.items:
                recourse_constrs[f, item, "unaffected"].setAttr(
                    "rhs", stage1_vars.s[f, item])
                if f in self.demand_nodes:
                    recourse_constrs[f, item, "affected"].setAttr(
                        "rhs", stage1_vars.s[f, item] - demands[f, item]
                    )
                    recourse_constrs[f, item, "zero_supply"].setAttr(
                        "rhs",
                        (demands[f, item] - stage1_vars.s[f, item]) / self.max_population
                    )
        return recourse_constrs

    def update_stage2_scenario(self, scenario_constrs, scenario):
        demands = scenario.demands
        for d in self.demand_nodes:
            for item in self.items:
                scenario_constrs[d, item].setAttr("rhs", demands[d, item])
        return scenario_constrs

    def update_stage2_transport_ub(self, transport_ub_constrs, ub,
                                   key_indices=[1, 2]):
        """Update transport upper-bounds (big M constraints)

        Parameters
        ----------
        transport_ub_constrs : dict
            Map 3-tuple (facility, demand node, item) to a gurobipy
            constraint.
        ub : dict
            Dictionary containing new upper bounds. Keys can be any tuple
            that is a subset of (facility, demand node, item), respecting
            that order, and is queried according to key_indices.
        key_indices : list of ints
            List must be ascending and can by any of the following:
            [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2].
            Provides indices of the 3-tuple (facility, demand node, item)
            to query ub. Example: if key_indices = [1, 2], then the RHS of
            constraint transport_ub_constrs[f, dn, item] is set to ub[dn, item],
            where f, dn and item are elements of facilities, demand_nodes
            and items, respectively.
        """
        for f in self.facilities:
            for dn in self.demand_nodes:
                for item in self.items:
                    key_tmp = (f, dn, item)
                    key = tuple([key_tmp[i] for i in key_indices])
                    curr_ub = ub[key]
                    transport_ub_constrs[f, dn, item].setAttr("rhs", curr_ub)
        return transport_ub_constrs

    def update_stage2(self, stage2_model, stage2_vars, stage2_constrs,
                      stage1_vars_vals, scenario, transport_ub,
                      solve_updated_model=True):
        """
        Update stage 1 solution and uncertain parameters of the second stage.

        First stage variables can only appear on the right-hand side. Uncertain
        parameters can appear on the right-hand side and in the objective.

        Parameters
        ----------
        stage2_model : Gurobi model object
        stage2_vars : Stage2Vars
        stage2_constrs : Stage2Constrs
        stage1_vars_vals : Stage1Vars
            Each attribute must contain values in a gurobipy.tupledict
        scenario : Scenario
        transport_ub : New upperbounds on 
        solve_updated_model : bool, optional
            Flag for whether or not the updated model should be immediately
            solved.

        Returns
        -------
        stage2_constrs : Stage2Constrs
        stage2_objval : float
            Returned if solve_updated_model is True. If model is infeasible,
            None is returned.
        """
        recourse_constrs = stage2_constrs.recourse
        scenario_constrs = stage2_constrs.scenario
        transport_ub_constrs = stage2_constrs.transport_ub

        # update recourse and scenario constraints
        recourse_constrs = self.update_stage2_recourse(recourse_constrs,
                                                       stage1_vars_vals,
                                                       scenario)
        scenario_constrs = self.update_stage2_scenario(scenario_constrs,
                                                       scenario)

        # This is only needed if t_ub depend on the random demands
        # transport_ub_constrs = self.update_stage2_transport_ub(
        #     transport_ub_constrs, transport_ub)

        # update scenario costs
        objexpr_stage2 = self.get_objective_stage2(stage2_vars, scenario)
        stage2_model.setObjective(objexpr_stage2, GRB.MINIMIZE)

        stage2_constrs = Stage2Constrs(recourse_constrs, scenario_constrs,
                                       transport_ub_constrs)
        if solve_updated_model:
            stage2_model.optimize()
            if stage2_model.status != 2:
                warn("Subproblem not solved to optimality")
            if stage2_model.status in {3, 4}:
                stage2_objval = None
            else:
                stage2_objval = stage2_model.ObjVal
            return stage2_constrs, stage2_objval
        else:
            return stage2_constrs

    def get_objective_stage1(self, stage1_vars, override=False):
        """
        Return stage 1 objective function.

        If stage1_vars is a Stage1Vars dataclass where each attribute is a
        gurobipy.tupledict containing variables, then return a Gurobi
        expression. If the attributes are gurobipy.tupledict containing
        the values, return the objective value evaluated at stage1_vars.
        If options.charge_stage1 is False, the objective Gurobi expression
        can still be returned by setting override to True
        (otherwise, 0 is returned).
        """
        if not self.options.charge_stage1 and not override:
            return 0
        else:
            x = stage1_vars.x
            s = stage1_vars.s                

            opening_costs = quicksum(self.opening_costs[f] * x[f]
                                     for f in self.facilities)
            if self.options.charge_excess_orders_only:
                excess_order = stage1_vars.excess_order
                f = self.facilities[0]
                resource_costs = quicksum(
                    self.allocation_costs[f, item] * excess_order[item]
                    for item in self.items)
            elif self.options.charge_allocation:
                resource_costs = sum(self.allocation_costs[f, item] * s[f, item]
                                     for item in self.items
                                     for f in self.facilities)
            else:
                resource_costs = 0
            return opening_costs + resource_costs

    def get_objective_stage2(self, stage2_vars, scenario, scenario_vars=None):
        y = stage2_vars.y
        t = stage2_vars.t
        demand_viol = stage2_vars.demand_viol
        if scenario_vars is None:
            per_unit_cost = scenario.fixed_charge_costs
        else:
            per_unit_cost = scenario_vars.cost_var
        fixed_charge_cost = quicksum(
            self.link_charge_scale * per_unit_cost[f, d] * y[f, d]
            for f in self.facilities
            for d in self.demand_nodes)

        if self.options.charge_per_unit:
            transport_cost = quicksum(per_unit_cost[f, d] * t[f, d, item]
                                      for f in self.facilities
                                      for d in self.demand_nodes
                                      for item in self.items)
        else:
            transport_cost = 0

        if self.options.is_demand_uncertain:
            viol_cost = (self.demand_penalty
                         * quicksum(demand_viol[d, item]
                                    for d in self.demand_nodes
                                    for item in self.items))
        else:
            viol_cost = 0
        objExpr = fixed_charge_cost + transport_cost + viol_cost
        return objExpr

    def construct_stage1(self):
        model_stage1 = Model("First Stage")
        self.set_params_stage1(model_stage1)
        stage1_vars = self.add_stage1(model_stage1)
        return model_stage1, stage1_vars

    def construct_stage2(self, stage1_vars=None, scenario=None):
        model_stage2 = Model("Second Stage")
        self.set_params_stage2(model_stage2)
        stage2_vars, stage2_constrs = self.add_stage2(model_stage2, stage1_vars,
                                                      scenario)
        return model_stage2, stage2_vars, stage2_constrs

    def construct_saa(self, samples, probs):
        sample_names = samples.keys()
        extensive_model = Model("Extensive Model")
        self.set_params_extensive_model(extensive_model)
        stage1_vars = self.add_stage1(extensive_model)
        objExpr = self.get_objective_stage1(stage1_vars)
        stage2_vars = {}
        for sampleName in sample_names:
            stage2_vars[sampleName], _ = self.add_stage2(extensive_model,
                                                            stage1_vars,
                                                            samples[sampleName],
                                                            sampleName)
            objexpr_stage2 = self.get_objective_stage2(stage2_vars[sampleName],
                                                       samples[sampleName])
            objExpr += probs[sampleName] * objexpr_stage2
        extensive_model.setObjective(objExpr, GRB.MINIMIZE)
        return extensive_model, stage1_vars, stage2_vars

    def get_stage1_vals(self, model, stage1_vars):
        xVal = model.getAttr("X", stage1_vars.x)
        sVal = model.getAttr("X", stage1_vars.s)
        return Stage1Vars(xVal, sVal)

    def get_stage2_vals(self, model, stage2_vars):
        y_val = model.getAttr("X", stage2_vars.y)
        t_val = model.getAttr("X", stage2_vars.t)
        if stage2_vars.demand_viol is not None:
            demand_viol_val = model.getAttr("X", stage2_vars.demand_viol)
        else:
            demand_viol_val = None
        return Stage2Vars(y_val, t_val, demand_viol_val)

    def print_stage1_solution(self, model, stage1_vars,
                              printVars=["x", "s", "excess_order"]):
        print("#################")
        if "x" in printVars:
            x = stage1_vars.x
            s = stage1_vars.s
            for f in self.facilities:
                if x[f].X > 0:
                    print("Facility", f, s[f, "bundle"].X)
            print("#################")
        if "excess_order" in printVars and self.options.charge_excess_orders_only:
            excess_order = stage1_vars.excess_order
            excessOrderVal = model.getAttr("X", excess_order)
            print("Excess Order:")
            print(excessOrderVal)
            print("#################")
        return

    def print_stage2_solution(self, model, stage2_vars):
        print("#################")
        y_val = model.getAttr("X", stage2_vars.y)
        t_val = model.getAttr("X", stage2_vars.t)
        print("Quantities transported")
        for f in self.facilities:
            for d in self.demand_nodes:
                if y_val[f, d] > 1e-6:
                    print((f, d), ":", t_val[f, d, "bundle"])
        demand_viol = stage2_vars.demand_viol
        demand_viol_val = model.getAttr("X", demand_viol)
        print("Demand Violation")
        for d in self.demand_nodes:
            if demand_viol_val[d, "bundle"] > 1e-6:
                print(d, demand_viol_val[d, "bundle"])
        print("#################")
        return
