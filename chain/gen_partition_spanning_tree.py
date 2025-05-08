import geopandas as gpd
from gerrychain import Graph, Partition
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import numpy as np

from gerrychain.updaters import Tally, cut_edges

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept

# Load Graph
graph = Graph.from_json("../graph/dual-graph.json")
# print(nx.is_connected(graph))
# print(len(graph))

PARTITIONS = 52

def random_spanning_tree(graph: nx.Graph) -> nx.Graph:
    """
    Builds a spanning tree chosen uniformly from the space of all
    spanning trees of the graph. Uses Wilson's algorithm.
    """
    root = random.choice(list(graph.node_indices))
    tree_nodes = set([root])
    next_node = {root: None}

    for node in graph.node_indices:
        u = node
        while u not in tree_nodes:
            next_node[u] = random.choice(list(graph.neighbors(u)))
            u = next_node[u]

        u = node
        while u not in tree_nodes:
            tree_nodes.add(u)
            u = next_node[u]

    G = nx.Graph()
    for node in tree_nodes:
        if next_node[node] is not None:
            G.add_edge(node, next_node[node])

    return G

def split_spanning_tree(tree, n, alpha=0.5, beta=1.5):
    """
    Computes a set of cut edges to split a spanning tree into n components
    Guarentees that during each component within [alpha*avg_size, beta*avg_size]
    """
    # Initialize the list of cut edges and calculate the total number of nodes
    cut_edges = set()
    remaining_components = n

    current_nodes = len(tree.nodes)
    target_size = current_nodes // remaining_components
    min_size = int(alpha * target_size)
    max_size = int(beta * target_size) 

    while len(cut_edges) < n - 1:
        # Randomly select an edge and simulate the cut
        edge = random.choice(list(tree.edges))
        tree.remove_edge(*edge)  # Temporarily remove the edge
        components = list(nx.connected_components(tree))

        # Determine sizes of the two components
        component_sizes = [len(c) for c in components]
        larger_component_index = 0 if component_sizes[0] >= component_sizes[1] else 1
        smaller_component_size = component_sizes[1 - larger_component_index]

        # Check if the smaller component satisfies the size constraint
        if min_size <= smaller_component_size <= max_size:
            cut_edges.add(edge)  # Accept this edge as a cut
            # Keep only the larger component in the tree
            larger_component_nodes = components[larger_component_index]
            tree = tree.subgraph(larger_component_nodes).copy()
            remaining_components -= 1  # Decrease the number of components to create
            print(f"{len(cut_edges)}/{n-1} cut edges found")
        else:
            # Reconnect the edge if the cut is not valid
            tree.add_edge(*edge)

    return cut_edges

def generate_spanning_tree_partition(graph, num_districts):
    """
    Generates a partition using random spanning tree method.
    
    Args:
        graph: Graph object to partition
        num_districts: Number of districts to create
    
    Returns:
        dict: Mapping of node IDs to district IDs
    """
    # Generate spanning tree
    spanning_tree = random_spanning_tree(graph)
    
    # Get cut edges that split into districts
    cut_edges = split_spanning_tree(spanning_tree, num_districts)
    
    # Remove the cut edges to create districts
    spanning_tree.remove_edges_from(cut_edges)
    
    # Get connected components (districts)
    districts = list(nx.connected_components(spanning_tree))
    
    # Create mapping of nodes to district IDs
    node_to_district = {}
    for district_id, nodes in enumerate(districts):
        for node in nodes:
            node_to_district[node] = district_id
            
    return node_to_district

# spanning_tree = random_spanning_tree(graph)
# cut_edges = split_spanning_tree(spanning_tree, PARTITIONS)
# spanning_tree.remove_edges_from(cut_edges)
# partitions = list(nx.connected_components(spanning_tree))
# partition_map = {i: list(part) for i, part in enumerate(partitions)}

# node_to_district = generate_spanning_tree_partition(graph, PARTITIONS)

# # Assign the 'district_id' attribute to all nodes in the graph
# nx.set_node_attributes(graph, node_to_district, name='district_id')
# graph.to_json("./graphs/shapefile_with_islands.json")

# # PLOT
# gdf = gpd.read_file("./shapefile_with_islands/shapefile_with_islands.shp")

# # Create a new column and initialize it with NaN or a default value
# gdf['district_id'] = None
# base_colors = cm.tab20.colors * 3  # Repeat to ensure at least 52 colors
# shuffled_colors = np.random.permutation(base_colors[:52])  # Randomly shuffle the colors
# random_cmap = ListedColormap(shuffled_colors)

# # Assign district_id based on the mapping
# for district_id, indices in partition_map.items():
#     gdf.loc[indices, 'district_id'] = district_id
# print("Unique districts:", gdf['district_id'].unique())
# gdf.plot("district_id", 
#          cmap=random_cmap,
#          edgecolor="black",
#          linewidth=0.1)  
# plt.show()

import geopandas as gpd
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

def plot_district_map(partition_type, file_id):
    # Load partition assignment
    partition_file = f"./results/chain-final-partitions/{partition_type}_final_partition0.json"
    with open(partition_file) as f:
        assignment = json.load(f)
    assignment = {int(k): v for k, v in assignment.items()}

    # Load shapefile
    gdf = gpd.read_file("../data/shapefile_with_islands/shapefile_with_islands.shp")
    
    # Assign districts
    gdf['district_id'] = gdf.index.map(assignment)

    # Create colormap
    base_colors = cm.tab20.colors * 3  # Repeat to ensure at least 52 colors
    shuffled_colors = np.random.permutation(base_colors[:52])
    random_cmap = ListedColormap(shuffled_colors)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 10))
    gdf.plot("district_id", 
             cmap=random_cmap,
             edgecolor="black",
             linewidth=0.1,
             ax=ax)
    
    plt.title(f"{partition_type.replace('_', ' ').title()} Final Partition")
    plt.axis('off')
    
    # Save plot
    plt.savefig(f"./results/chain-final-partitions/{partition_type}_final_map0.png", 
                bbox_inches='tight', 
                dpi=300)
    plt.show()

if __name__ == "__main__":
    # Plot maps for each partition type
    plot_district_map("spanning_tree", 0)
    plot_district_map("random_nodes", 0)
    plot_district_map("current_districting", 0)