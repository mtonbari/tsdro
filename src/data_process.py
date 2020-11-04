"""
This file parses and sets up the necessary data to create a LocationRouting
object. This data is from a case study on hurricanes affecting 
the Gulf of Mexico states in the U.S. and was obtained from a study
by Carmen G. Rawls and Mark A. Turnqist.

References:
Carmen G. Rawls and Mark A. Turnquist, Pre-positioning of emergency supplies
for disaster response, 2010.
"""

import numpy as np
import os
import sys
import helpers as hp
import itertools


hurricane_names = ["Alicia", "Camille", "Bonnie", "Floyd", "Andrew",
                  "Opal", "Isabel", "Lili", "Katrina", "Bertha", "Fran",
                  "Dennis", "Emily", "Georges", "Hugo"]
items = ["water", "food", "medical_kits"]
hurricane_to_category = dict(zip(hurricane_names,
                                 [2, 4, 1, 1, 3, 2, 1, 0, 4, 1, 2, 2, 2, 3, 3]))
populations = {0: 183392, 1: 325733, 2: 1.493e6, 3: 7573136,
               4: 2.326e6, 5: 197881, 6: 650618, 7: 164422,
               8: 47877, 9: 78001, 10: 221599, 11: 0, 12: 391006,
               13: 45968, 14: 189572, 15: 209880, 16: 692587,
               17: 498044, 18: 133451, 19: 872498, 20: 122607,
               21: 130113, 22: 145862, 23: 193551, 24: 0,
               25: 903889, 26: 285713, 27: 392890, 28: 470914,
               29: 24565}
highway_nodes = (11, 24)


def read_node_coordinates(filepath):
    """
    Read and return coordinates of nodes.
    
    Parameters
    ----------
    filepath : string
        Path to coordinates file.
    
    Returns
    -------
    coordinates : dict
        Maps nodes (int) to their coordinates (tuple).
    """
    coordinates = {}
    with open(filepath, mode="r") as f:
        for line in f:
            line = line.rstrip("\n")
            s = line.split(",")
            node = int(s[1])
            latitude = float(s[2])
            longitude = float(s[3])
            coordinates[node] = (longitude, latitude)
    return coordinates


def read_demands(demand_nodes, filepath):
    """
    Read and return historical demands
    
    Parameters
    ----------
    filepath : string
        Path to demands file.
    
    Returns
    -------
    demands : dict
        Maps hurricane names (string) to another dictionary, which maps
        a node and an item to the demand.
    """
    demands = {}
    node = None
    with open(filepath, mode="r") as f:
        for line in f:
            line = line.rstrip()
            s = line.split()
            if len(s) == 1:
                hurricane_name = s[0]
                demands[hurricane_name] = {}
            else:
                node = int(s[0]) - 1
                if node in demand_nodes:
                    for i, item in enumerate(items):
                        demands[hurricane_name][node, item] = float(
                            s[len(s) - 3 + i])
    return demands


def get_edges():
    """
    Return dictionary mapping edges to weights.

    Edges are stored as a tuple of two nodes.
    """
    arcs = {(0, 1): 2,
            (0, 2): 5,
            (1, 2): 2,
            (1, 4): 3.5,
            (2, 4): 3,
            (2, 3): 4.5,
            (3, 4): 4,
            (3, 8): 5.5,
            (3, 5): 6,
            (4, 9): 2.5,
            (5, 6): 2.5,
            (5, 8): 3,
            (6, 7): 3.5,
            (6, 15): 4,
            (6, 16): 3.5,
            (7, 8): 1.5,
            (7, 11): 2,
            (7, 15): 4,
            (7, 13): 2.5,
            (8, 10): 2,
            (8, 9): 3,
            (9, 10): 2,
            (10, 11): 1.5,
            (10, 12): 1,
            (11, 12): 1,
            (11, 13): 1,
            (13, 14): 1,
            (14, 15): 3.5,
            (14, 17): 5.5,
            (14, 23): 4,
            (15, 16): 3.5,
            (15, 17): 2.5,
            (16, 17): 4,
            (16, 19): 6,
            (17, 18): 3.5,
            (17, 19): 4,
            (17, 22): 4,
            (17, 23): 4,
            (17, 24): 5,
            (18, 19): 3,
            (18, 20): 3,
            (18, 21): 2,
            (19, 20): 3,
            (20, 21): 3,
            (21, 22): 2,
            (22, 25): 2,
            (23, 24): 2,
            (24, 25): 1,
            (24, 26): 2.5,
            (24, 27): 3,
            (26, 27): 1.5,
            (26, 28): 3,
            (27, 28): 3,
            (28, 29): 3}
    return arcs


def historical_fractions_affected(demands, combined=True, op=max):
    """
    Return fraction of affected population for each hurricane category and item.

    Parameters
    ----------
    combined: bool 
        True if all items are in one bundle.
    op: {"max", "sum"}
        function used to combine item demands if considering bundles instead.
    
    Returns
    -------
    fraction_affected : dict
        Maps a hurricane category and item to a list of historical demands
        across all affected nodes, normalized by the population at the node.
    """
    num_categories = 5
    if combined:
        assert op is not None
        fraction_affected = {c: [] for c in range(num_categories)
                             for item in items}
    else:
        fraction_affected = {(c, item): [] for c in range(num_categories)
                             for item in items}
    for hurricane_name in hurricane_names:
        curr_demands = demands[hurricane_name]
        currCat = hurricane_to_category[hurricane_name]
        unique_demand_nodes = set(k[0] for k in curr_demands.keys())
        for dn in unique_demand_nodes:
            if combined:
                val = op(curr_demands[dn, item] / populations[dn]
                         for item in items)
                fraction_affected[currCat].append(val)
            else:
                for item in items:
                    fraction_affected[currCat, item].append(
                        curr_demands[dn, item] / populations[dn])
    return fraction_affected


def representative_fractions(historical_fractions_affected):
    """
    From historical data, create bins to determine a value that is
    approximately the most common fraction of population affected by
    a hurricane, for each hurricane category.

    Parameters
    ----------
    historical_fractions_affected: dict
        Dictionary obtained from historical_fractions_affected.
    
    Returns
    -------
    cat_to_fraction : dict 
        Representative percentage for each category.
    """
    categories = list(historical_fractions_affected.keys())
    thresholds = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 4e-3, 7e-3, 1e-2,
                  4e-2, 7e-2, 1e-1, 4e-1, 7e-1]
    bins = {(cat, t): 0 for cat in range(5) for t in thresholds}
    for cat in categories:
        fractions_sorted = np.sort(historical_fractions_affected[cat])
        for f in fractions_sorted:
            for t in thresholds:
                if f <= t:
                    bins[cat, t] += 1
                    break
    cat_to_fraction = {}
    for cat in categories:
        currBins = [bins[cat, t] for t in thresholds]
        max_ind = np.argmax(currBins)
        cat_to_fraction[cat] = thresholds[max_ind]
    return cat_to_fraction


def get_bipartite_costs(facilities, demand_nodes, distances):
    """
    Return fixed-charge cost of transporting resources between a facility and demand node.
    """
    bipartite_costs = {}
    nonzero_distances = [v for v in distances.values() if v > 0]
    for f, dn in itertools.product(facilities, demand_nodes):
        bipartite_costs[f, dn] = distances[f, dn]
        if f == dn:
            bipartite_costs[f, dn] = min(nonzero_distances) / 2
    return bipartite_costs


data_path = "../data"
coordinates_filepath = os.path.join(data_path, "node_coordinates.csv")
demands_filepath = os.path.join(data_path, "demands.txt")

landfall_nodes = [0, 1, 4, 9, 10, 11, 12, 13, 14, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29]
categories = [0, 1, 2, 3, 4]
num_categories = len(categories)
edge_weight_mult = 10

demand_nodes = list(range(30))
for n in highway_nodes:
    demand_nodes.remove(n)
facilities = list(range(30))
coordinates = read_node_coordinates(coordinates_filepath)
demands = read_demands(demand_nodes, demands_filepath)
distances = hp.pairwise_geodesic(coordinates)
edges = get_edges()
bipartite_costs = get_bipartite_costs(facilities, demand_nodes, distances)

historical_fractions = historical_fractions_affected(demands)
# cat_to_fraction = representative_fractions(historical_fractions)
cat_to_fraction = {0: 0.001, 1: 0.005, 2: 0.01, 3: 0.1, 4: 0.3}
category_weights = [113, 74, 76, 18, 3]
landfall_probs = {lf: 1 / len(landfall_nodes) for lf in landfall_nodes}


if __name__ == "__main__":
    geodesic_distances = hp.pairwise_geodesic(coordinates)
    landfall = 28
    for n in range(30):
        print(n, geodesic_distances[landfall, n])
    print(cat_to_fraction)
