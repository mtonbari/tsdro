"""
Implementation of the K-adaptability algorithm to solve
two-stage distributionally robust problem with binary variables in the
second stage.
"""
import numpy as np
from gurobipy import *
import scenario as sc
from dataclasses import dataclass, replace


@dataclass
class KAdaptVars:
    wassMult: Var
    beta: tupledict
    demandViol: tupledict
    z: tupledict


class KAdaptability():
    def __init__(self, tsdro, K):
        """
        Parameters
        ----------
        tsdro : TSDRO object
        K : int
            Number of second stage solutions that can be chosen from in the
            second stage.
        """
        self.tsdro = tsdro
        self.K = K
        self.stage1Vars = None
        self.stage2Vars = None
        self.maxDemand = None
        return

    def initializeMaster(self):
        """
        Create master model
        
        Initialize first stage variables and constraints, and K sets of second 
        stage variables and constraints.

        Returns
        -------
        master : Gurobi model
        stage1Vars : location_routing.Stage1Vars dataclass
        stage2Vars : dict
            Maps int (1 to K) to a location_routing.Stage2Vars dataclass
        """
        lrInstance = self.tsdro.lrInstance
        scenarios = self.tsdro.scenarios

        master, stage1Vars = lrInstance.constructStage1()

        self.maxDemand = {(d, item): -np.inf
                          for d in lrInstance.demandNodes
                          for item in lrInstance.items}
        for scenarioName in self.tsdro.initialScenarioNames:
            scenario = scenarios[scenarioName]
            for d in lrInstance.demandNodes:
                for item in lrInstance.items:
                    if self.maxDemand[d, item] < scenario.demands[d, item]:
                        self.maxDemand[d, item] = scenario.demands[d, item]
        tUB = {}
        for f in lrInstance.facilities:
            for d in lrInstance.demandNodes:
                for item in lrInstance.items:
                    tUB[f, d, item] = self.maxDemand[d, item]

        stage2Vars = {}
        for k in range(self.K):
            stage2Vars[k], _, _ = lrInstance.addStage2(
                master, stage1Vars, demandInObj=True, tUB=tUB, stageName=k)
        return master, stage1Vars, stage2Vars

    def getDisjunctionLHS(self, kAdaptVars, scenarioName, sampleName):
        scenario = self.tsdro.scenarios[scenarioName]
        sample = self.tsdro.samples[sampleName]
        beta = kAdaptVars.beta
        wassMult = kAdaptVars.wassMult
        scenario_distance = sc.get_scenario_distance(sample, scenario,
                                                     self.tsdro.lrInstance)
        lhs = LinExpr()
        lhs.addTerms(scenario_distance * self.tsdro.probs[sampleName], wassMult)
        lhs.addTerms(1, beta[sampleName])
        return lhs

    def getDisjunctionRHS(self, stage2Vars, kAdaptVars, scenarioName, k,
                          sampleName=None):
        lrInstance = self.tsdro.lrInstance
        scenario = self.tsdro.scenarios[scenarioName]
        demandViol = kAdaptVars.demandViol
        z = kAdaptVars.z
        costSum = sum(scenario.transportCosts.values())
        disjunctM = costSum + len(lrInstance.demandNodes) * lrInstance.demandPenalty * max(self.maxDemand.values())
        if sampleName is None:
            zCurr = z[k, scenarioName]
        else:
            zCurr = z[k, scenarioName, sampleName]
        currDemandViol = {(d, item):
                          demandViol[d, item, k, scenarioName]
                          for d in lrInstance.demandNodes
                          for item in lrInstance.items}
        currStage2Vars = replace(stage2Vars[k],
                                 demandViol=currDemandViol)
        objExpr = lrInstance.getObjectiveStage2(currStage2Vars,
                                                scenario)
        disjunction = disjunctM * (1 - zCurr)
        rhs = objExpr - disjunction
        return rhs

    def reformulateDemandViol(self, model, stage2Vars, demandViol, d,
                              scenarioName, k):
        lrInstance = self.tsdro.lrInstance
        scenario = self.tsdro.scenarios[scenarioName]
        t = self.stage2Vars[k].t
        currDemand = scenario.demands[d, "bundle"]

        supplyToNode = quicksum(t[f, d, "bundle"]
                                for f in lrInstance.facilities)
        model.addLConstr(
            demandViol[d, "bundle", k, scenarioName], ">",
            currDemand - supplyToNode,
            name="demandViol_" + "sc" + str(scenarioName) + "_k" + str(k) + "_d" + str(d))

    def addSubproblem(self, model, stage2Vars, fixedPolicies=False):
        lrInstance = self.tsdro.lrInstance
        samples = self.tsdro.samples
        initialScenarioNames = self.tsdro.initialScenarioNames

        wassMult = model.addVar(lb=0, name="wass_multiplier")
        beta = model.addVars(samples.keys(), lb=-np.inf, ub=np.inf, name="beta")
        if not fixedPolicies:
            z = model.addVars(self.K, initialScenarioNames,
                              vtype=GRB.BINARY, name="z")
            demandViol = model.addVars(lrInstance.demandNodes, lrInstance.items,
                                       self.K, initialScenarioNames, lb=0,
                                       name="dv")
        kAdaptVars = KAdaptVars(wassMult, beta, demandViol, z)
        for scenarioName in initialScenarioNames:
            model.addLConstr(quicksum(z[k, scenarioName]
                                      for k in range(self.K)), "==", 1,
                             name="disj_" + "sc" + str(scenarioName))  # + "_sa" + str(sampleName))
            for sampleName, sample in samples.items():
                lhs = self.getDisjunctionLHS(kAdaptVars, scenarioName,
                                             sampleName)
                for k in range(self.K):
                    pn = self.tsdro.probs[sampleName]
                    rhs = self.getDisjunctionRHS(stage2Vars, kAdaptVars,
                                                 scenarioName, k)
                    model.addLConstr(lhs, ">", pn * rhs,
                                     name="dro_" + "sc" + str(scenarioName) + "_sa" + str(sampleName) + "_k" + str(k))
                    model.addLConstr(lhs, ">", 0)
            for k in range(self.K):
                for d in lrInstance.demandNodes:
                    self.reformulateDemandViol(model, stage2Vars, demandViol,
                                               d, scenarioName, k)
        # model.addLConstr(quicksum(z), ">", len(samples) + 1)
        # for scenarioName in initialScenarioNames:
        #     lhs = quicksum(z[k, scenarioName] for k in range(self.K))
        #     model.addLConstr(lhs, "<", 1)
        return model, kAdaptVars

    def addConvexSubproblem(self, model, v):
        lrInstance = self.tsdro.lrInstance
        samples = self.tsdro.samples
        scenarios = self.tsdro.scenarios
        initialScenarioNames = self.tsdro.initialScenarioNames

        wassMult = model.addVar(lb=0, name="wass_multiplier")
        beta = model.addVars(samples.keys(), lb=-np.inf, ub=np.inf, name="beta")
        for scenarioName in initialScenarioNames:
            scenario = scenarios[scenarioName]
            for sampleName, sample in samples.items():
                scenario_distance = sc.get_scenario_distance(sample, scenario, lrInstance)
                lhs = LinExpr()
                lhs.addTerms(scenario_distance * self.tsdro.probs[sampleName], wassMult)
                lhs.addTerms(1, beta[sampleName])
                rhs = self.tsdro.probs[sampleName] * v[scenarios]
                constr_name = ("dro_"
                               + "sc" + str(scenarioName)
                               + "_sa" + str(sampleName)
                               + "_k" + str(k))
                model.addLConstr(lhs, ">", rhs, name= constr_name)
        return model, KAdaptVars(wassMult, beta, None, None)

    def initializeExtensive(self):
        """
        Build K-adaptability extensive model.

        Returns
        -------
        self.master : Gurobi model
        """
        self.master, self.stage1Vars, self.stage2Vars = self.initializeMaster()
        lrInstance = self.tsdro.lrInstance
        self.master, self.kAdaptVars = self.addSubproblem(self.master,
                                                          self.stage2Vars,
                                                          fixedPolicies=False)
        objExprStage1 = lrInstance.getObjectiveStage1(self.stage1Vars)
        objExprInner = (self.tsdro.wassrad * self.kAdaptVars.wassMult
                        + quicksum(self.kAdaptVars.beta))
        objExpr = objExprStage1 + objExprInner
        self.master.setObjective(objExpr, GRB.MINIMIZE)

        self.master.params.OutputFlag = 1
        return self.master

    def initializeExtensiveConvex(self):
        self.master, self.stage1Vars, self.stage2Vars = self.initializeMaster()
        lrInstance = self.tsdro.lrInstance
        scenarios = self.tsdro.scenarios

        v = self.master.addVars(scenarios.keys(), lb=0)
        self.master, self.kAdaptVars = self.addConvexSubproblem(self.master, v)
        objExprStage1 = lrInstance.getObjectiveStage1(self.stage1Vars)
        objExprInner = (self.tsdro.wassrad * self.kAdaptVars.wassMult
                        + quicksum(self.kAdaptVars.beta))
        objExpr = objExprStage1 + objExprInner
        self.master.setObjective(objExpr, GRB.MINIMIZE)

        self.master.params.OutputFlag = 1
        return self.master
