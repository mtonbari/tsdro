"""
Helper functions used to parse data.
"""


import networkx as nx
import numpy as np
import itertools
from math import sin, cos, asin, radians, sqrt, atan2, degrees
import math
import sys


def gen_shortest_paths(edges, facilities, demandNodes):
    """
    Return shortest path lengths from facilities to demandNodes and Graph object
    """
    G = nx.Graph()
    for a, w in edges.items():
        if not G.has_edge(*a):
            G.add_edge(a[0], a[1], weight=w)
    sp_lengths = {}
    for f in facilities:
        for d in demandNodes:
            if f != d:
                sp_lengths[f, d] = nx.shortest_path_length(G, source=f,
                                                           target=d,
                                                           weight="weight")
    return sp_lengths, G


def pairwise_geodesic(coordinates):
    """ Return pairwise geodesic (great-circle) distances between coordinates.

    Parameters
    ----------
    coordinates : dict
        Maps nodes to (latitude, longitude) tuples

    Returns
    -------
    D : dict
        Maps node tuple to the geodesic distance between them
    """
    R = 6371  # radius of the Earth in km
    nodes = list(coordinates.keys())
    D = {}
    for node1, node2 in itertools.product(nodes, repeat=2):
        lon1, lat1 = coordinates[node1]
        lon2, lat2 = coordinates[node2]
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlong = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlong / 2)**2
        c = 2 * asin(sqrt(a))
        D[node1, node2] = R * c  # great circle distance in km
    return D


def get_angle(A, B, C):
    """
    Return angle between vectors defined by [A, B] and [A, C]

    The angle is positive if [A, C] is a counter-clockwise rotation of [A, B],
    and negative otherwise.

    Parameters
    ----------
    A : tuple, list
        coordinates of tail of vector
    B : tuple, list
        coordinates of head of vector
    C : tuple, list
        coordinates of C

    Returns
    -------
    angle : float64
        angle in radians between vectors AB and AC.
    """
    segment = np.subtract(B, A)
    A_to_C = np.subtract(C, A)
    angle = atan2(np.cross(segment, A_to_C), np.dot(segment, A_to_C))
    return angle


def rotate_vector(A, B, angle):
    """ Rotate vector by given angle.

    Rotation is counterclockwise if angle is positive, and clockwise
    if angle is negative.

    Parameters
    ----------
    A : tuple, list
        coordinates of tail of vector
    B : tuple, list
        coordinates of head of vector
    angle: float64, int
        angle in radians between -pi and pi.
    Returns
    -------
    rotated_translated : 1-D array of size 2.
        coordinates of point resulting from rotating B, anchored at A
    """
    assert angle >= -math.pi and angle <= math.pi
    R = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    rotated = np.dot(R, np.subtract(B, A))
    rotated_translated = np.add(A, rotated)
    return rotated_translated


def get_neighbors(source, coordinates, geodesic_distances, radius,
                  circle_endpoint, max_neighbors=np.inf):
    """Get neighbors of source given a radius and direction.

    Only nodes in the half-circle defined by the radius and direction are
    considered as neighbors.

    Parameters:
    source (tuple) -- landfall node
    node (tuple) -- list of all nodes
    coordinates (dict) -- dictionary whose keys are nodes and values are
                         (latitude, longitude) tuples
    geodesic_distances (dict) -- pairwise great-circle distances between nodes
    radius -- distance within which nodes are considered neighbors
    circle_endpoint

    Returns
    -------
    neighborNodes -- list of nodes neighboring source
    """
    nodes = list(coordinates.keys())
    D = geodesic_distances
    neighborNodes = []
    for n in nodes:
        if n != source:
            if len(neighborNodes) < max_neighbors:
                a = tuple(np.sort((source, n)))
                withinRadius = (D[a] <= radius)
                # angle formed by n-source-circle_endpoint
                angle = get_angle(coordinates[source], circle_endpoint,
                                  coordinates[n])
                withinAngle = degrees(angle) >= -5 or degrees(angle) <= -175
                if withinRadius and withinAngle:
                    neighborNodes.append(n)
    return neighborNodes


def normalize(*args):
    '''
    Normalize each input in args by subtracting the minimum and dividing by
    the range.
    Input: 1-D lists or dictionaries whose values are array-like.
    Output: Normalized data for each input. Lists are returned as a dictionary
    mapping the original value to the normalized one. Dictionaries are returned
    in the same format with values normalized.
    '''
    results = []
    for i, arg in enumerate(args):
        if type(arg) is list:
            arg_normalized = {}
            min_val = np.min(np.abs(arg))
            max_val = np.max(np.abs(arg))
            for v in arg:
                if len(arg) == 1:
                    arg_normalized[v] = v
                else:
                    arg_normalized[v] = (v - min_val) / (max_val - min_val)
            results.append(arg_normalized)
        elif type(arg) is dict:
            data = np.array(list(arg.values()))
            min_lat, min_lon = np.min(data, axis=0)
            max_lat, max_lon = np.max(data, axis=0)
            min_coordinates = np.array([min_lat, min_lon])
            tmp_denom = np.array([max_lat - min_lat, max_lon - min_lon])
            normalized = {k: tuple(np.divide(v - min_coordinates, tmp_denom))
                          for k, v in arg.items()}
            results.append(normalized)
        else:
            sys.exit("in normalize: Arguments must be lists or dictionaries.")
    return tuple(results)


if __name__ == "__main__":
    A = [0, 0]
    B = [1, 1]
    angle = 0
    print(rotate_vector(A, B, angle))
