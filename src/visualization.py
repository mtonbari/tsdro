"""
This file contains tools to visualize first and second stage solutions, and
the network for different scenario demands.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.linalg as npl


dot_size = 300


def initialize_ax(coordinates):
    fig, ax = plt.subplots(figsize=(15, 15))
    for n, (lon, lat) in coordinates.items():
        ax.scatter(lon, lat, s=dot_size, facecolors="none", edgecolors="k")
        ax.annotate(n, (lon, lat), ha="center", va="center")
    return ax


def annotate_nodes(ax, coordinates, supplies, demands=None, item="bundle",
                   plot_edges=False):
    """If item is None, supplies and demands are only indexed by node."""
    facilities = [k if item is None else k[0] for k in supplies.keys()]
    if demands is not None:
        demand_nodes = [k if item is None else k[0] for k in demands.keys()]
        max_demand = max(demands.values())
        nodes = set(facilities).union(set(demand_nodes))
    else:
        nodes = facilities
    for n in nodes:
        lon, lat = coordinates[n]
        plot_it = False  # True if node n should be plotted
        curr_supply = supplies.get((n, item), supplies[n])
        if n in facilities and curr_supply > 1e-6:
            facecolor = "blue"
            edgecolor = "blue"
            alpha = 0.5
            text_coords = (coordinates[n][0], coordinates[n][1] + 0.25)
            ax.annotate(round(curr_supply, 3), text_coords, ha="center",
                        va="center")
            plot_it = True
        if demands is not None:
            curr_demand = demands.get((n, item), demands[n])
            if n in demand_nodes and curr_demand > 1e-6:
                facecolor = "red"
                edgecolor = "red"
                d = curr_demand
                alpha = max(d / max_demand, 0.1)
                # alpha = 1
                plot_it = True
            if n in demand_nodes and n in facilities:
                edgecolor = "blue"
        if plot_it:
            ax.scatter(lon, lat, s=dot_size, facecolors=facecolor,
                       edgecolors=edgecolor, alpha=alpha)
    return ax


def plot_edges(ax, coordinates, edges):
    for e, w in edges.items():
        coords1 = np.array(coordinates[e[0]])
        coords2 = np.array(coordinates[e[1]])
        x = [coords1[0], coords2[0]]
        y = [coords1[1], coords2[1]]
        ax.plot(x, y, linestyle="-", linewidth=1, color="k")

        # Annotate edges
        line_length = npl.norm(coords1 - coords2)
        direction = (coords2 - coords1) / npl.norm(coords2 - coords1)
        text_coords = coords1 + (line_length / 2) * direction
        ax.annotate(w, text_coords, ha="left", va="bottom")
    return ax


def plot_routing(ax, coordinates, t):
    for k, t_val in t.items():
        f = k[0]
        dn = k[1]
        if t_val > 1e-6 and f != dn:
            (tail_lon, tail_lat) = coordinates[f]
            (dlon, dlat) = np.subtract(coordinates[dn], coordinates[f])
            ax.arrow(tail_lon, tail_lat, dlon, dlat, head_width=0.15,
                     length_includes_head=True)
        elif t_val > 1e-6 and f == dn:
            (tail_lon, tail_lat) = coordinates[f]
            (head_lon, head_tail) = np.add(coordinates[f], [0.1, 0.1])
            p = patches.FancyArrowPatch(coordinates[f],
                                        (head_lon, head_tail),
                                        connectionstyle="arc3,rad=6",
                                        arrowstyle="->,head_width=0.15")
            ax.add_patch(p)
    return ax


if __name__ == "__main__":
    import data_process as dat

    open_facilities = [2, 7, 9, 10, 13, 14, 15, 20, 22, 23]
    tmp = [30000, 255, 390, 2215, 229, 100000, 1540, 692, 3394, 1000]
    supplies = {(f, "bundle"): s for f, s in zip(open_facilities, tmp)}
    demands = {(0, "bundle"): 30, (1, "bundle"): 20, (2, "bundle"): 10}
    t = {(0, 0): 1, (1, 2): 1}
    ax = initialize_ax(dat.coordinates)
    print(dat.landfall_nodes)
    # ax = annotate_nodes(ax, dat.coordinates, supplies)
    plot_edges(ax, dat.coordinates, dat.edges)
    plt.show()
