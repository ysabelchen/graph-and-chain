import geopandas as gpd
from gerrychain import Graph, Partition
import json
from collections import defaultdict
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def process_results(partition_type):
    shapefile = "../data/shapefile_with_islands/shapefile_with_islands.shp"
    dem_votes = []
    rep_votes = []

    if partition_type == "spanning_tree" or partition_type == "random_nodes":
        num_files = 10
    elif partition_type == "current_districting":
        num_files = 1

    for file_id in range(num_files):
        if partition_type == "spanning_tree":
            partition_assignment_file = f"./results/chain-final-partitions/spanning_tree_final_partition{file_id}.json"
        elif partition_type == "random_nodes":
            partition_assignment_file = f"./results/chain-final-partitions/random_nodes_final_partition{file_id}.json"
        elif partition_type == "current_districting":
            partition_assignment_file = f"./results/chain-final-partitions/current_districting_final_partition{file_id}.json"

        with open(partition_assignment_file) as f:
            assignment = json.load(f)
        assignment = {int(k): v for k, v in assignment.items()}

        gdf = gpd.read_file(shapefile)
        gdf.drop([f"district_{i}" for i in range(1, 10)], axis=1, inplace=True)
        gdf.drop([f"district{i}" for i in range(10, 91)], axis=1, inplace=True)

        # gdf.info()

        gdf["district_i"] = gdf.index.map(assignment)
        district_to_index_map = gdf.groupby("district_i").apply(lambda x: x.index.tolist()).to_dict()

        dem_vote = defaultdict(int)
        rep_vote = defaultdict(int)
        for district_id, parts in district_to_index_map.items():
            for part in parts:
                dem_vote[district_id] += gdf.iloc[part]["PRES_Dem"]
                rep_vote[district_id] += gdf.iloc[part]["PRES_Rep"]
        print(f"partition: {file_id}")
        print(sum(dem_vote[i] > rep_vote[i] for i in range(52)))
        print(sum(dem_vote[i] < rep_vote[i] for i in range(52)))
        dem_votes.append(sum(dem_vote[i] > rep_vote[i] for i in range(52)))
        rep_votes.append(sum(dem_vote[i] < rep_vote[i] for i in range(52)))

    print(dem_votes)
    print(rep_votes)

    plot_seat_distribution(dem_votes, partition_type, "Dem")
    plot_seat_distribution(rep_votes, partition_type, "Rep")

    # count, bins, _ = plt.hist(dem_votes, bins=12, density=True, alpha=0.6, color='b', edgecolor='black')
    # mu, sigma = np.mean(dem_votes), np.std(dem_votes)
    # x = np.linspace(40, 52, 100)
    # pdf = norm.pdf(x, mu, sigma)
    # plt.plot(x, pdf, 'r', linewidth=2, label=f'Normal Fit ($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})')

    # # Labels and title
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.title('Dem Seats')
    # plt.legend()

    # os.makedirs("./results/experiment1", exist_ok=True)
    # plt.savefig(f"./results/experiment1/{partition_type}_seat_distribution.png")
    # plt.show()

    #TODO: save seats to csv for each party

def plot_seat_distribution(votes, partition_type, party_name):
    """Plot seat distribution with normal fit for given party's votes."""
    plt.figure(figsize=(10, 6))

    # Set bin range based on party
    if party_name == 'Dem':
        bin_range = np.arange(39.75, 52.75, 0.5)  # For Dem: 40-52
        x_range = np.linspace(40, 52, 100)
    else:
        bin_range = np.arange(2.25, 8.25, 0.25)    # For Rep: 4-8
        x_range = np.linspace(2, 9, 100)

    # Create histogram centered on whole numbers
    plt.hist(votes, bins=bin_range, density=False,  
             color='blue' if party_name == 'Dem' else 'red', 
             alpha=0.6, label=f'{party_name} Seats')

    # Calculate mean and standard deviation
    mean = np.mean(votes)
    std = np.std(votes)

    # Create normal distribution curve
    y = norm.pdf(x_range, mean, std)
    plt.plot(x_range, y, 'r-', label=f'Normal Fit (μ={mean:.2f}, σ={std:.2f})')

    plt.title(f'{party_name} Seats: {partition_type.capitalize()}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    os.makedirs("./results/experiment1_initial_partitions", exist_ok=True)
    plt.savefig(f"./results/experiment1_initial_partitions/{partition_type}_{party_name.lower()}_seats_distribution.png")
    plt.show()

if __name__ == "__main__":
    # options: spanning_tree, random_nodes, current_districting
    process_results("spanning_tree")
    process_results("random_nodes")
    process_results("current_districting")


# Use function for both parties
# plot_seat_distribution(dem_votes, 'Dem')
# plot_seat_distribution(rep_votes, 'Rep')