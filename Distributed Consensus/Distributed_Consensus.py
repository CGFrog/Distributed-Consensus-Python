import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def update_values(G):
    new_values = {}
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        values = [G.nodes[n]['value'] for n in neighbors]
        values.append(G.nodes[node]['value'])
        new_values[node] = np.mean(values) if values else G.nodes[node]['value']
    
    for node, value in new_values.items():
        G.nodes[node]['value'] = value

def display_final_graph(G):
    values = [G.nodes[node]['value'] for node in G.nodes]
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=False, node_color=values, cmap=plt.cm.viridis, node_size=700)
    
    labels = {node: f"{node}\n{G.nodes[node]['value']:.2f}" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="white")
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Node Values")
    plt.title("Final State of the Graph with Node Labels and Values")
    plt.show()

def global_average(G):
    values = [G.nodes[n]['value'] for n in G.nodes]
    return np.mean(values)

def track_values_and_averages(G, iterations):
    node_values_over_time = {node: [] for node in G.nodes}
    global_averages = []
    
    for node in G.nodes:
        node_values_over_time[node].append(G.nodes[node]['value'])
    global_averages.append(global_average(G))
    
    for _ in range(iterations):
        update_values(G)
        
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
    plt.title('Node Values and Global Average Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

def set_rand_node_values(G):
    initial_values = np.random.rand(len(G.nodes))
    node_mapping = {node: i for i, node in enumerate(G.nodes)}
    nx.set_node_attributes(G, {node: {'value': initial_values[node_mapping[node]]} for node in G.nodes})

def main():
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
    
    iterations = int(input("Enter number of iterations to simulate consensus: "))
    node_values_over_time, global_averages = track_values_and_averages(G, iterations)

    final_average = global_average(G)
    print(f"Final global average: {final_average:.4f}")

    plot_values_and_global_average(node_values_over_time, global_averages)
    display_final_graph(G)
    print("Consensus process complete!")


main()
