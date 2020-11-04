"""
This file contains the structure of a scenario, discretizations
of each component of a scenario, associated probabilities, and various
functions to create scenarios and compute the distance between two scenarios.
"""

from dataclasses import dataclass
import numpy.linalg as npl
import numpy as np
import helpers as hp
import math
from math import pi
from gurobipy import *

import data_process as dat


##########
# Scenario discretizations and probabilites
##########
intensities = [1, 2, 4, 6, 10]
# fractions_affected = [0.001, 0.005, 0.01, 0.1, 0.3]
fractions_scenarios = np.linspace(0.001, 0.3, 5).tolist()
fractions_scenarios = np.round(fractions_scenarios, 3).tolist()
cost_multiplier = [((i+1) / 25 + 1) for i in range(len(fractions_scenarios))]
radii = [i for i in range(0, 6)]
angles = [0, -pi / 4, -pi / 2, pi / 4, pi / 2]
data_normalized = hp.normalize(intensities, dat.coordinates, radii, angles,
                               fractions_scenarios)
intensities_normalized = data_normalized[0]
coordinates_normalized = data_normalized[1]
radii_normalized = data_normalized[2]
angles_normalized = data_normalized[3]
fractions_normalized = data_normalized[4]

# map normalized values to original values (needed in scenario model)
fractions_norm_to_original = {v: k for k, v in fractions_normalized.items()}
radii_norm_to_original = {v: k for k, v in radii_normalized.items()}
angles_norm_to_origial = {v: k for k, v in angles_normalized.items()}

# Define probabilities
fraction_weights = {}
num_categories = len(dat.category_weights)
spread = len(fractions_scenarios) / num_categories
remainder = len(fractions_scenarios) % num_categories
if remainder == 0:
    spread = [spread] * len(fractions_scenarios)
else:
    tmp = len(fractions_scenarios) - remainder
    spread = [math.floor(spread)] * tmp + [math.ceil(spread)] * remainder
start_ind = 0
for i, cw in enumerate(dat.category_weights):
    end_ind = int(start_ind + spread[i])
    for j in range(start_ind, end_ind):
        curr_fraction = fractions_scenarios[j]
        fraction_weights[curr_fraction] = cw
    start_ind = end_ind
total_weight = sum(fraction_weights.values())
fraction_probs = {f: w / total_weight for f, w in fraction_weights.items()}

radii_weights = dict(zip(radii, [3, 4, 5, 6, 5, 4, 1, 0.5, 0.1]))
total_weight = sum(radii_weights.values())
radius_probs = {r: w / total_weight for r, w in radii_weights.items()}

angle_weights = dict(zip(angles, [4, 5, 3, 2, 1]))
total_weight = sum(angle_weights.values())
angle_probs = {a: w / total_weight for a, w in angle_weights.items()}

landfall_probs = dat.landfall_probs


@dataclass
class Scenario:
    fixed_charge_costs: dict = None
    demands: dict = None
    landfall: int = None
    landfall_intensity: int = None
    radius: float = None
    affected_nodes: list = None
    angle: int = None               # angle in radians in [-\pi, \pi]
    fraction_affected: float = None


@dataclass
class ScenarioVars:
    landfall_bin: tupledict
    fraction_affected_bin: tupledict
    radius_bin: tupledict
    angle_bin: tupledict

    coordinates_var: tupledict
    fraction_affected_var: tupledict
    radius_var: tupledict
    angle_var: tupledict

    hypograph_var: tupledict

    # cost_var: tupledict


def _get_scenario_distance(scenario, sample, lr_instance, norm=2):
    """
    Simple l-2 norm distance between two scenarios represented by vectors
    of demands and link costs.
    """
    # cost_dev = [scenario.fixed_charge_costs[f, d]
    #             - sample.fixed_charge_costs[f, d]
    #             for f in dat.facilities
    #             for d in dat.demand_nodes]
    if lr_instance.is_demand_uncertain:
        demand_dev = [scenario.demands[d, item] - sample.demands[d, item]
                      for d in dat.demand_nodes for item in ["bundle"]]
        scenario_distance = npl.norm(demand_dev, norm)
        # scenario_distance = npl.norm(cost_dev + demandDev, norm)
    else:
        pass
        # scenario_distance = npl.norm(cost_dev, norm)
    return scenario_distance


def _get_scenario_hierarchy_distance(scenario, sample, lr_instance, norm=2):
    coordinates_scenario = np.array(coordinates_normalized[scenario.landfall])
    # intensity_scenario = intensities_normalized[scenario.landfall_intensity]
    radius_scenario = radii_normalized[scenario.radius]
    angle_scenario = angles_normalized[scenario.angle]
    fraction_scenario = fractions_normalized[scenario.fraction_affected]

    coordinates_sample = np.array(coordinates_normalized[sample.landfall])
    # intensity_sample = intensities_normalized[sample.landfall_intensity]
    radius_sample = radii_normalized[sample.radius]
    angle_sample = angles_normalized[sample.angle]
    fraction_sample = fractions_normalized[sample.fraction_affected]

    coords_dev = npl.norm(coordinates_scenario - coordinates_sample, norm)
    # intensity_dev = abs(intensity_scenario - intensity_sample)
    radius_dev = abs(radius_scenario - radius_sample)
    angle_dev = abs(angle_scenario - angle_sample)
    fraction_dev = abs(fraction_scenario - fraction_sample)

    sample_dev = (coords_dev**norm
                  + fraction_dev**norm
                  # + intensity_dev**norm
                  + radius_dev**norm
                  + angle_dev**norm)
    return sample_dev


def _get_scenario_ellipsoidal_distance(scenario, sample, D):
    Q = D.T.dot(D)
    s1 = np.zeros(len(dat.landfall_nodes))
    s2 = np.zeros(len(dat.landfall_nodes))
    landfall_ind1 = 0
    landfall_ind2 = 0
    for i, landfall in enumerate(dat.landfall_nodes):
        if scenario.landfall == landfall:
            landfall_ind1 = i
        if sample.landfall == landfall:
            landfall_ind2 = i
    s1[landfall_ind1] = scenario.landfall_intensity
    s2[landfall_ind2] = sample.landfall_intensity
    return math.sqrt((s2 - s1).T.dot(Q).dot(s2 - s1))


def get_scenario_name(scenario=None, **kwargs):
    if scenario is not None:
        intensity_ind = fractions_scenarios.index(scenario.fraction_affected)
        kwargs = {"lf": scenario.landfall, "i": intensity_ind,
                  "r": scenario.radius, "ang": scenario.angle} 
    scenario_name = ""
    for i, j in kwargs.items():
        if type(j) != list:
            scenario_name += str(i) + str(round(j, 3))
        else:
            j_rounded = [round(v, 3) for v in j]
            scenario_name += str(i) + "-".join(map(str, charge_allocation))
    return scenario_name


def create_cost_scenario(affected_nodes, intensity_inds, landfall):
    """Get costs associated with one scenario."""
    return dat.bipartite_costs.copy()


def create_demand_scenario(landfall, affected_nodes, fractions_range):
    # find how many nodes are affected at each intensity level
    # and store in intensity_spread
    spread = len(affected_nodes) / len(fractions_range)
    remainder = len(affected_nodes) % len(fractions_range)
    if remainder == 0:
        intensity_spread = [spread] * len(affected_nodes)
    else:
        tmp = len(fractions_range) - remainder
        intensity_spread = ([math.floor(spread)] * tmp
                            + [math.ceil(spread)] * remainder)
    distances = [dat.distances[landfall, n] for n in affected_nodes]
    sorted_inds = np.argsort(distances)
    fractions_affected = {}
    intensity_inds = {}
    affected_count = 0  # how many nodes have been affected at current intensity
    intensity_ind = len(fractions_range) - 1
    for n_ind in sorted_inds:
        n = affected_nodes[n_ind]
        intensity_inds[n] = intensity_ind
        fractions_affected[n] = fractions_range[intensity_ind]
        affected_count += 1
        # move to the next intensity level and reset affected_count
        if affected_count == intensity_spread[intensity_ind]:
            intensity_ind -= 1
            affected_count = 0

    demands = {}
    for dn in dat.demand_nodes:
        if dn in affected_nodes:
            demands[dn, "bundle"] = fractions_affected[dn] * dat.populations[dn]
        else:
            demands[dn, "bundle"] = 0
    return demands, fractions_affected, intensity_inds


def get_prob(landfall, fraction_affected, radius, angle):
    p = (landfall_probs[landfall]
         * fraction_probs[fraction_affected]
         * radius_probs[radius]
         * angle_probs[angle])
    return p


def create_single_scenario(landfall, fraction_affected,
                           fractions_range, radius, angle, prev_affected=[]):
    east_point = np.add(dat.coordinates[landfall], [5, 0])

    rotated_point = hp.rotate_vector(dat.coordinates[landfall], east_point,
                                     angle)
    neighbors = hp.get_neighbors(landfall, dat.coordinates,
                                 dat.distances, radius, rotated_point)
    affected_nodes = neighbors + [landfall]
    # if set(affected_nodes) in prev_affected:
    #     return None, affected_nodes

    demands, fractions_affected, intensity_inds = create_demand_scenario(
        landfall, affected_nodes, fractions_range)
    costs = create_cost_scenario(affected_nodes, intensity_inds, landfall)
    scenario = Scenario(costs, demands, landfall, None, radius,
                        affected_nodes, angle, fraction_affected)
    return scenario, affected_nodes


def create_scenarios():
    scenarios = {}
    true_probs = {}
    for landfall in dat.landfall_nodes:
        for intensity_ind, fraction_affected in enumerate(fractions_scenarios):
            # fractions_range = [dat.cat_to_fraction[c]
            #                    for c in dat.categories[:intensity_ind + 1]]
            fractions_range = [f
                               for f in fractions_scenarios[:intensity_ind + 1]]
            # prev_affected = []
            for radius in radii:
                for angle in angles:
                    scenario, _ = create_single_scenario(
                        landfall, fraction_affected, fractions_range, radius,
                        angle)
                    # if set(curr_affected_nodes) in prev_affected:
                    #     continue
                    # else:
                    #     prev_affected.append(set(curr_affected_nodes.copy()))
                    # scenario_name = get_scenario_name(
                    #     lf=landfall, i=intensity_ind, r=radius,
                    #     ang=math.degrees(angle))
                    scenario_name = get_scenario_name(scenario)
                    scenarios[scenario_name] = scenario
                    true_probs[scenario_name] = get_prob(
                        landfall, fraction_affected, radius, angle)
    return scenarios, true_probs


def initialize_scenario_model():
    """
    Scenario variables for coordinates, fraction_affected, radius and angle
    represent the normalized values, since they are only needed to compute
    the distance in the model.
    """
    scenario_model = Model()
    # scenario model here:
    hypograph_var = scenario_model.addVar(lb=0, ub=np.inf, name="hypograph_var")
    landfall_bin = scenario_model.addVars(dat.landfall_nodes, vtype=GRB.BINARY,
                                          name="lf")
    fraction_affected_bin = scenario_model.addVars(fractions_scenarios,
                                                   vtype=GRB.BINARY,
                                                   name="frac")
    radius_bin = scenario_model.addVars(radii, vtype=GRB.BINARY, name="r")
    angle_bin = scenario_model.addVars(angles, vtype=GRB.BINARY, name="a")

    # cost_var = scenario_model.addVars(dat.facilities, dat.demand_nodes,
    #                                   lb=0, ub=np.inf, name="cost")
    coordinates_var = scenario_model.addVars(["lon", "lat"], lb=0)
    fraction_affected_var = scenario_model.addVars(fractions_scenarios, lb=0,
                                                   ub=max(fractions_normalized))
    radius_var = scenario_model.addVars(radii, lb=0, ub=max(radii_normalized))
    angle_var = scenario_model.addVars(angles,
                                       lb=min(angles_normalized),
                                       ub=max(angles_normalized))

    # Can only pick one scenario
    scenario_model.addLConstr(quicksum(landfall_bin) == 1)
    scenario_model.addLConstr(quicksum(fraction_affected_bin) == 1)
    scenario_model.addLConstr(quicksum(radius_bin) == 1)
    scenario_model.addLConstr(quicksum(angle_bin) == 1)

    # Map binary variables to actual scenario values
    max_lon_norm = max(v[0] for v in coordinates_normalized.values())
    max_lat_norm = max(v[1] for v in coordinates_normalized.values())
    max_frac_norm = max(fractions_normalized.values())
    max_radius_norm = max(radii_normalized.values())
    max_angle_norm = max(angles_normalized.values())
    for landfall in dat.landfall_nodes:
        # longitude
        expr_lb = coordinates_normalized[landfall][0] * landfall_bin[landfall] 
        expr_ub = coordinates_normalized[landfall][0] + max_lon_norm * (1 - landfall_bin[landfall])
        scenario_model.addLConstr(coordinates_var["lon"] >= expr_lb)
        scenario_model.addLConstr(coordinates_var["lon"] <= expr_ub)
        # latitude
        expr_lb = coordinates_normalized[landfall][1] * landfall_bin[landfall] 
        expr_ub = coordinates_normalized[landfall][1] + max_lat_norm * (1 - landfall_bin[landfall])
        scenario_model.addLConstr(coordinates_var["lat"] >= expr_lb)
        scenario_model.addLConstr(coordinates_var["lat"] <= expr_ub)
    for frac, frac_norm in fractions_normalized.items():
        expr_lb = frac_norm * fraction_affected_bin[frac] 
        expr_ub = frac_norm + max_frac_norm * (1 - fraction_affected_bin[frac])
        scenario_model.addLConstr(fraction_affected_var[frac] >= expr_lb)
        scenario_model.addLConstr(fraction_affected_var[frac] <= expr_ub)
    for radius, radius_norm in radii_normalized.items():
        expr_lb = radius_norm * radius_bin[radius]
        expr_ub = radius_norm + max_radius_norm * (1 - radius_bin[radius])
        scenario_model.addLConstr(radius_var[radius] >= expr_lb)
        scenario_model.addLConstr(radius_var[radius] <= expr_ub)
    for angle, angle_norm in angles_normalized.items():
        expr_lb = angle_norm - max_angle_norm * (1 - angle_bin[angle])
        expr_ub = angle_norm + max_angle_norm * (1 - angle_bin[angle])
        scenario_model.addLConstr(angle_var[angle] >= expr_lb)
        scenario_model.addLConstr(angle_var[angle] <= expr_ub)

    scenario_vars = ScenarioVars(landfall_bin, fraction_affected_bin,
                                 radius_bin, angle_bin, coordinates_var,
                                 fraction_affected_var, radius_var, angle_var,
                                 hypograph_var)

    return scenario_model, scenario_vars


def retrieve_scenario_vars(scenario_model):
    pass


def parse_scenario_solution(scenario_vars):
    landfall = None
    for lf in dat.landfall_nodes:
        if scenario_vars.landfall_bin[lf].X == 1:
            landfall = lf
            break
    assert landfall is not None

    fraction_affected = fractions_norm_to_original[
        scenario_vars.fraction_affected_var]
    intensity_ind = fractions_scenarios.index(fraction_affected)
    fractions_range = [f for f in fractions_scenarios[:intensity_ind + 1]]
    radius = radii_norm_to_original[scenario_vars.radius_var]
    angle = angles_norm_to_origial[scenario_vars.angle_var]
    return landfall, fraction_affected, fractions_range, radius, angle


def update_scenario_model_objective(scenario_model, scenario_vars,
                                    sample, wass_mult):
    """
    Values in arguments should the original values and not normalized.
    """
    lon_scenario = scenario_vars.coordinates_var["lon"]
    lat_scenario = scenario_vars.coordinates_var["lat"]
    radius_scenario = radii_normalized[radius]
    angle_scenario = angles_normalized[angle]
    fraction_scenario = fractions_normalized[fraction_affected]

    coordinates_sample = np.array(coordinates_normalized[sample.landfall])
    radius_sample = radii_normalized[sample.radius]
    angle_sample = angles_normalized[sample.angle]
    fraction_sample = fractions_normalized[sample.fraction_affected]

    lon_dev = lon_scenario - coordinates_sample[0]
    lat_dev = lat_scenario - coordinates_sample[1]
    radius_dev = radius_scenario - radius_sample
    angle_dev = angle_scenario - angle_sample
    fraction_dev = fraction_scenario - fraction_sample

    sample_dev = (lon_dev * lon_dev
                  + lat_dev * lat_dev
                  + fraction_dev * fraction_dev
                  + radius_dev * radius_dev
                  + angle_dev * angle_dev)
    scenario_model.set_objective(
        scenario_vars.hypograph_var - wass_mult * sample_dev, GRB.MAXIMIZE)
    return


def update_scenario_model_row(scenario_model, scenario_vars, stage2_obj):
    """
    If there is no uncertainty in the cost, stage2_obj is a value. Otherwise,
    it is a Gurobi expression constructed using variables from scenario_vars.
    """
    scenario_model.addLConstr(scenario_vars.hypograph_var <= stage2_obj)
    return


get_scenario_distance = _get_scenario_hierarchy_distance
# get_scenario_distance = _get_scenario_distance


if __name__ == "__main__":
    output = initialize_scenario_model()
    output[0].write("scenario_model.lp")
    