"""
This file contains functions to create random networks and data necessary
to create a LocationRouting object.
"""

from numpy.random import RandomState
import numpy.linalg as npl
import numpy as np
import itertools
import visualization as vis
import matplotlib.pyplot as plt
import pickle

prng = RandomState(8)


def create_nodes(num_nodes, gridx, gridy, min_distance_allowed=2):
    """Randomly create nodes within grid

    Parameters
    ----------
    num_nodes : int
        Number of nodes
    gridx : tuple or list
        Lower and upper limit along x-axis
    gridy: tuple or list
        Lower and upper limit along y-axis

    Returns
    -------
    coordinates : dict
        Maps nodes (represented by integers) to coordinates stored in a tuple.
    """
    # Create random coordinates
    coordinates = []
    for i in range(num_nodes):
        lon = prng.uniform(gridx[0], gridx[1])
        lat = prng.uniform(gridy[0], gridy[1])
        curr_coords = (lon, lat)
        if len(coordinates) >= 1:
            distances = []
            for c in coordinates:
                distances.append(npl.norm(np.subtract(curr_coords, c)))
            curr_min_distance = np.min(distances)
            while curr_min_distance < min_distance_allowed:
                lon = prng.uniform(gridx[0], gridx[1])
                lat = prng.uniform(gridy[0], gridy[1])
                curr_coords = (lon, lat)
                distances = []
                for c in coordinates:
                    if curr_coords != c:
                        distances.append(npl.norm(np.subtract(curr_coords, c)))
                curr_min_distance = np.min(distances)

        coordinates.append((lon, lat))

    # Name nodes in order of increasing longitude
    dtype = [("lon", float), ("lat", float)]
    coordinates_arr = np.array(coordinates, dtype=dtype)
    sorted_coordinates = np.sort(coordinates_arr, order="lon")
    coordinates = {}
    for i, curr_coords in enumerate(sorted_coordinates):
        coordinates[i] = tuple(curr_coords)
    distances = {}
    for n1, n2 in itertools.product(nodes, nodes):
        diff = np.subtract(coordinates[n1], coordinates[n2])
        distances[n1, n2] = npl.norm(diff)
    return coordinates, distances


def create_populations(nodes):
    """Randomly generate populations for each node.
    
    Parameters
    ----------
    nodes : list of ints

    Returns
    -------
    populations : dict
        Maps nodes to their population.
    """
    populations_arr = prng.randint(100000, 2e6, size=len(nodes))
    populations = dict(zip(nodes, populations_arr))
    return populations


def get_sorted_endpoints(s, nodes, distances):
    """ Return list of endpoints in order of increasing distance from s.
    
    Parameters
    ----------
    s : int
        Reference node from which we sort distances
    nodes : list of ints
    distances : dict
        Maps tuple (node1, node2) to the distance between them.
    
    Returns
    -------
    sorted_endpoints : list of ints
    """
    potential_nodes = [n for n in nodes if n != s]
    distances_from_s = [distances[s, n] for n in potential_nodes]
    sorted_inds = np.argsort(distances_from_s)
    sorted_endpoints = [potential_nodes[i] for i in sorted_inds]
    return sorted_endpoints


def get_endpoint(n, s, neighbor_nodes_s, distances):
    """Return endpoint to which n should be connected.

    Among nodes sharing an edge with s and s itself, find the closest node to n
    and return it as the endpoint.

    This function is helpful in avoiding creating edges that "pass through"
    a node in the middle, where it would be more realistic for the middle
    node to be one of the endpoints instead.
    """
    potential_endpoints = [s] + neighbor_nodes_s
    distances_from_n = {s: distances[s, n] for s in potential_endpoints}
    endpoint = min(distances_from_n, key=distances_from_n.get)
    return endpoint


def create_edges(nodes, distances):
    """
    Randomly create edges between nodes.

    Parameters
    ----------
    nodes : list of ints
    distances : dict
        Maps tuple (node1, node2) to the distance between them.
    
    Returns
    -------
    edges : dict
        Maps a created edge
    """
    edges = {}
    sources = [nodes[0]]
    nonconnected_nodes = nodes[1:]
    prev_sources = []
    neighbor_nodes = {n: [] for n in nodes}
    num_outgoing_edges = {n: 0 for n in nodes}
    while sources:
        s = sources.pop(0)
        prev_sources.append(s)
        sorted_endpoints = get_sorted_endpoints(s, nodes, distances)
        num_incident_edges = prng.randint(1, 4)
        count = 0
        for n in sorted_endpoints:
            endpoint = get_endpoint(n, s, neighbor_nodes[s], distances)
            e = tuple(np.sort([endpoint, n]))
            if e in edges:
                continue
            edges[e] = round(distances[e], 1)
            num_outgoing_edges[e[0]] += 1
            num_outgoing_edges[e[1]] += 1
            neighbor_nodes[endpoint].append(n)
            if n not in sources and n not in prev_sources:
                sources.append(n)
                nonconnected_nodes.remove(n)
            count += 1
            if count == num_incident_edges:
                break
    # test = [n for n, v in num_outgoing_edges.items() if v == 0]
    for s in nonconnected_nodes:
        sorted_endpoints = get_sorted_endpoints(s, nodes, distances)
        num_incident_edges = prng.randint(1, 4)
        count = 0
        for n in sorted_endpoints:
            e = tuple(np.sort([endpoint, n]))
            if e in edges:
                continue
            edges[e] = round(distances[e], 1)
            count += 1
            if count == num_incident_edges:
                break
    # Clean up
    edges_to_delete = [(19, 28), (5, 5), (0, 0), (15, 15), (22, 22), (28, 28),
                       (24, 24)]
    # Remove edges where endpoints are the same node
    for e in edges.keys():
        if e[0] == e[1]:
            edges_to_delete.append(e)
    
    for e in edges_to_delete:
        try:
            del edges[e]
        except KeyError:
            pass
    edges_to_add = [(7, 14)]
    for e in edges_to_add:
        edges[e] = round(distances[e], 1)
    return edges


def get_landfall_nodes(coordinates, gridx, gridy,
                       limit_from_south=None,
                       limit_from_east=None):
    """Given coordinates and a grid, get potential landfall nodes

    Parameters
    ----------
    coordinates : dict
        Maps an int to a tuple. The int represents the node and the tuple
        stores the coordinates of the node.
    gridx : tuple or list
        Lower and upper limit along x-axis
    gridy: tuple or list
        Lower and upper limit along y-axis

    Returns
    -------
    landfall_nodes : list of ints
    """
    if limit_from_east is None:
        limit_from_east = 0.25 * (gridx[1] - gridy[0])
    if limit_from_south is None:
        limit_from_south = 0.25 * (gridy[1] - gridy[0])

    distances_from_south = {}  # Distances from southern limit of grid
    distances_from_east = {}  # Distances from eastern limit of grid
    for n, c in coordinates.items():
        south_point = (c[0], 0)
        east_point = (gridx[1], c[1])
        distances_from_south[n] = npl.norm(np.subtract(c, south_point))
        distances_from_east[n] = npl.norm(np.subtract(c, east_point))
    closest_to_south = min(distances_from_south.values())
    closest_to_east = min(distances_from_east.values())

    landfall_nodes = set()
    for n, c in coordinates.items():
        if distances_from_south[n] - closest_to_south <= limit_from_south:
            landfall_nodes.add(n)
        if distances_from_east[n] - closest_to_east <= limit_from_east:
            landfall_nodes.add(n)
    return landfall_nodes, distances_from_east, distances_from_south


def get_landfall_probabilities(landfall_nodes, distances_from_east,
                               distances_from_south):
    """Returns probabilities of a hurricane hitting each landfall.
    
    The probabilities are computed based on how far the node is from the coast,
    with the probability being smaller the further the node is.

    Parameters
    ----------
    landfall_nodes : list of ints
        Subset of nodes at which a hurricane can happen
    distances_from_east, distances_from_south : dict
        Maps nodes to the approximate distance between the node and the eastern
        and southern coast, respectively.

    Returns
    -------
    landfall_probs : dict
        Maps landfall nodes (ints) to their probability of being struck by
        a hurricane.
    """
    weights = {}
    for lf in landfall_nodes:
        distance_to_border = (distances_from_east[lf]
                              + distances_from_south[lf]) / 2
        weights[lf] = 1 / distance_to_border
    total_weight = sum(weights.values())
    landfall_probs = {lf: p / total_weight for lf, p in weights.items()}
    return landfall_probs


def get_bipartite_costs(facilities, demand_nodes, distances):
    """
    Returns fixed-charge costs of transporting any amount of goods on an edge.
    
    This is used for the model where resources are transported using direct 
    links from facilities to demand nodes (by air, for example).
    """
    bipartite_costs = {}
    nonzero_distances = [v for v in distances.values() if v > 0]
    for f, dn in itertools.product(facilities, demand_nodes):
        bipartite_costs[f, dn] = distances[f, dn]
        if f == dn:
            bipartite_costs[f, dn] = min(nonzero_distances) / 2
    return bipartite_costs


if __name__ == "__main__":
    num_nodes = 30
    nodes = range(30)
    gridx = (0, 20)
    gridy = (0, 20)
    coordinates, distances = create_nodes(num_nodes, gridx, gridy)
    nodes = list(coordinates.keys())
    populations = create_populations(nodes)
    edges = create_edges(nodes, distances)

    landfall_nodes_info = get_landfall_nodes(coordinates, gridx, gridy)
    landfall_probs = get_landfall_probabilities(*landfall_nodes_info)

    ax = vis.initialize_ax(coordinates)
    ax = vis.annotate_nodes(ax, coordinates, landfall_probs, item=None)
    ax = vis.plot_edges(ax, coordinates, edges)
    plt.show()