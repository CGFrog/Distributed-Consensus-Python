import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

def update_weighted_values(G):
    w_neighbor = 1
    w_agent = 1
    new_values = {}
    #Arithmetic Weighted average = sum(w_ix_i)/sum(w_i)
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        values = [G.nodes[n]['value'] for n in neighbors]
        total = w_neighbor*sum(values) + (w_agent * G.nodes[node]['value'])
        new_values[node] = total/((len(values) * w_neighbor) + w_agent) 
    #After all new values have been calculated replace each agents current value with the new average.
    for node, value in new_values.items():
        G.nodes[node]['value'] = value
        
def has_converged(G, global_average):
    for node in G.nodes:
        if abs(G.nodes[node]['value'] - global_average) > 0.001:
            return False
    return True

def has_converged_islands(G):
    connected_components = nx.connected_components(G)
    
    for component in connected_components:
        nodes_in_component = list(component)
        local_average = sum(G.nodes[node]['value'] for node in nodes_in_component) / len(nodes_in_component)
        
        for node in nodes_in_component:
            if abs(G.nodes[node]['value'] - local_average) > 0.001:
                return False
    
    return True

def display_final_graph(G):
    values = [G.nodes[node]['value'] for node in G.nodes]
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=False, node_color=values, cmap=plt.cm.viridis, node_size=700)
    
    labels = {node: f"{node}\n{G.nodes[node]['value']:.2f}" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="white")
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Node Values")
    plt.title("Graph")
    plt.show()

def global_average(G):
    values = [G.nodes[n]['value'] for n in G.nodes]
    return np.mean(values)

def track_values_and_averages(G):
    node_values_over_time = {node: [] for node in G.nodes}
    global_averages = []
    
    for node in G.nodes:
        node_values_over_time[node].append(G.nodes[node]['value'])
    global_averages.append(global_average(G))
    
    i = 0
    
    if nx.nx.is_connected(G):
        while not has_converged(G, global_averages[i]):
            update_weighted_values(G)
            i+=1
            for node in G.nodes:
                node_values_over_time[node].append(G.nodes[node]['value'])
            global_averages.append(global_average(G))
    # Try to rewrite this to reduce code duplication. 
    else:
        while not has_converged_islands(G):
            update_weighted_values(G)
            i+=1
            for node in G.nodes:
                node_values_over_time[node].append(G.nodes[node]['value'])
            global_averages.append(global_average(G))


    return node_values_over_time, global_averages

def plot_values_and_global_average(node_values_over_time, global_averages):
    plt.figure(figsize=(12, 8))
    
    for node, values in node_values_over_time.items():
        plt.plot(values, label=f'Node {node}', linestyle='--', alpha=0.7)

    plt.plot(global_averages, label='Global Average', color='red', linewidth=2, marker='o')
    
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Node Values/Global Average')
    plt.legend()
    plt.grid(True)
    plt.show()

def set_rand_node_values(G):
    initial_values = np.random.rand(len(G.nodes))
    node_mapping = {node: i for i, node in enumerate(G.nodes)}
    nx.set_node_attributes(G, {node: {'value': initial_values[node_mapping[node]]} for node in G.nodes})

def custom_graph():
    graph_type = input("Select graph type (binomial, cyclic, k-regular): ")

    if graph_type == "binomial":
        n = int(input("Enter number of nodes: "))
        p = float(input("Enter edge probability [0,1]: "))
        G = nx.erdos_renyi_graph(n, p)
    elif graph_type == "cyclic":
        n = int(input("Enter number of nodes: "))
        G = nx.cycle_graph(n)
    elif graph_type == "k-regular":
        n = int(input("Enter number of nodes for: "))
        k = int(input("Enter degree of nodes for: "))
        G = nx.random_regular_graph(k,n)
    else:
        print("Invalid graph type. Exiting...")
        return

    set_rand_node_values(G)

    initial_average = global_average(G)
    print(f"Initial global average: {initial_average:.4f}")
    
    node_values_over_time, global_averages = track_values_and_averages(G)

    final_average = global_average(G)
    print(f"Final global average: {final_average:.4f}")

    plot_values_and_global_average(node_values_over_time, global_averages)
    display_final_graph(G)
    print("Consensus process complete!")



custom_graph()