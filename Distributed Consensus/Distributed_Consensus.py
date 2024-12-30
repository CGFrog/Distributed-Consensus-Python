import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
"""

This code was purely for testing, all of it has been refactored in simulation.py and single_graph_sim.py.
I keep it here to show my process of development to those who may be interested.

"""


def update_values(G):
    new_values = {}
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        values = [G.nodes[n]['value'] for n in neighbors]
        values.append(G.nodes[node]['value'])
        new_values[node] = np.mean(values)
    for node, value in new_values.items():
        G.nodes[node]['value'] = value

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


'''
New Goal:

    - Plot convergence of multiple graphs 
    - x axis corresponding iteration to converge 
    - y axis corresponding to some independent variable.

    
    Step 1: Convergence detection
        We must determine when a graph has succesfully converged.
        We must define an error range for our convergence, i.e, when all nodes are within .05% of global average
        If there are islands we must discard the graph since convergence is impossible.
'''

def independent_variable_selection(graph):
    if graph == "binomial":
        ind_var = input("Select independent variable type (order, probability)")
        return ind_var
    if graph == "k-regular":
        ind_var = input("Select independent variable type (order, degree)")
        return ind_var
    else:
        print ("Cyclic chosen, independent variable set to order.")
        return "order"

def have_islands():
    allow_islands = input("Allow graphs to contain islands? (Type Y for yes, or press any other key for no)")
    if allow_islands == 'Y':
        allow_islands = True
    else:
        allow_islands = False
    return allow_islands

def convergence_point(G, allow_islands):
    iterations = 0
    g_avg = global_average(G)
    if not allow_islands:
        while not has_converged(G, g_avg):
            iterations += 1
            update_weighted_values(G)
    
    elif allow_islands:
        while not has_converged_islands(G):
            iterations += 1
            update_weighted_values(G)
    return iterations


# Definetly should use classes for this instead 
def binomial_sim(ind_var, allow_islands, iterations, data_points, ind_var_data):
    if ind_var == "probability":
        n = int(input("Enter number of nodes: "))
        for d in range(data_points):
            print (d)
            rp = np.random.rand()
            G = nx.erdos_renyi_graph(n,rp)
            set_rand_node_values(G)
            if allow_islands == False and nx.is_connected(G) == False:
                d-1
                continue
            iterations.append(convergence_point(G,allow_islands))
            ind_var_data.append(rp)
    else:
        p = float(input("Enter edge probability [0,1]: "))
        for d in range(data_points):
            print (d)
            rn = np.random.randint(1,100)
            G = nx.erdos_renyi_graph(rn, p)
            set_rand_node_values(G)
            if allow_islands == False and nx.is_connected(G) == False:
                d-1
                continue
            iterations.append(convergence_point(G,allow_islands))
            ind_var_data.append(rn)

#These functions all essentially do the same thing, this is very bad code and needs to be refactored.

def kreg_sim(ind_var, allow_islands, iterations, data_points, ind_var_data):
        if ind_var == "degree":
            n = int(input("Enter number of nodes: "))
            for d in range(data_points):
                print (d)
                rn = np.random.randint(1,n)
                G = nx.random_regular_graph(rn,n)
                set_rand_node_values(G)
                if allow_islands == False and nx.is_connected(G) == False:
                    d-1
                    continue
                iterations.append(convergence_point(G,allow_islands))
                ind_var_data.append(rn)
        else:
            k = int(input("Enter degree of nodes for: "))
            for d in range(data_points):
                print (d)
                rk = np.random.randint(k,k+1000)
                if k*rk%2 == 1:
                    d-=1
                    continue
                G = nx.random_regular_graph(k,rk)
                set_rand_node_values(G)
                if allow_islands == False and nx.is_connected(G) == False:
                    d-1
                    continue
                iterations.append(convergence_point(G,allow_islands))
                ind_var_data.append(rk)

def cyclic_sim(ind_var, allow_islands, iterations, data_points, ind_var_data):
    for d in range(data_points):
        print (d)
        rn = np.random.randint(1,100)
        G = nx.cycle_graph(rn)
        set_rand_node_values(G)
        if allow_islands == False and nx.is_connected(G) == False:
            d-1
            continue
        iterations.append(convergence_point(G,allow_islands))
        ind_var_data.append(rn)

def plot_iterations_vs_data(iterations, ind_var_data, title="Iterations vs Data", xlabel="Independent Variable Data", ylabel="Iterations"):
    plt.figure(figsize=(8, 5))
    # Plot independent variable data (x-axis) vs. iterations (y-axis)
    plt.plot(ind_var_data, iterations, marker='o', linestyle='', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def run_sim(graph_type, ind_var, allow_islands, data_points):
    ind_var_data = []
    iterations = []
    if graph_type == "binomial":
        binomial_sim(ind_var, allow_islands, iterations, data_points, ind_var_data)
    elif graph_type == "k-regular":        
        kreg_sim(ind_var, allow_islands, iterations, data_points, ind_var_data)
    else:
        cyclic_sim(ind_var, allow_islands, iterations, data_points, ind_var_data)
    
    print(f"Length of ind_var_data: {len(ind_var_data)}")
    print(f"Length of iterations: {len(iterations)}")

    plot_iterations_vs_data(
        iterations, 
        ind_var_data, 
        title=f"{graph_type.capitalize()} Simulation Results",
        xlabel=ind_var, 
        ylabel="Iterations"
    )


def init_sim():
    graph_type = input("Select graph type (binomial, cyclic, k-regular): ")
    ind_var = independent_variable_selection(graph_type)
    allow_islands = have_islands()
    data_points = eval(input("How many data points to evaluate?"))
    
    run_sim(graph_type, ind_var, allow_islands, data_points)
    

class simulation:
    def __init__(self, graph_type, ind_var, allow_islands,data_points):
        self.graph_type = graph_type
        self.ind_var = ind_var
        self.allow_islands = allow_islands
        self.data_points = data_points
        
"""
What we are trying to accomplish:
    instead of many if statements for each possible independent variable choice,
    I want a single function that runs the 




"""




init_sim()
