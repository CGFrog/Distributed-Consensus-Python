from math import ceil
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

"""

Part 2: Byzantine Agents

    Our goal is to introduce byzantine agents into the network, these agents could act in various ways:
        * Calculating the optimal wrong value and distributing it to other agents.
        * Sending an arbitrary value to all of its neighbor.
        * Sending an arbitrary value to different neighbors.
        * Never change value.
        * Send a delayed value.
        * ...
    
    TODO:    
        * Select pecentage of nodes that are byzantine agents
        * Randomly set various nodes in the graph as byzantine agents according to percentage
            1. Get total amount of byzantine agents needed.
            2. Generate a set of random numbers that range from 0 to n-1 where n is the amount of nodes in the network. Size of the set must be the size of n*percentage.
            3. Set each node of those indicies to be a byzantine agent
            4. When updating the values of the nodes we check if the agent is byzantine and we run a seperate function to determine its value.
            
"""

class Simulation:
    
    def __init__(self,order, percent_byzantine):
        self.order = order
        self.G = nx.empty_graph()
        self.CONSENSUS_ERROR = 0.01
        self.AGENT_WEIGHT = 1
        self.NEIGHBOR_WEIGHT = 1
        self.percent_byzantine = percent_byzantine
        
    def set_rand_node_values(self):
        initial_values = np.random.rand(len(self.G.nodes))
        node_mapping = {node: i for i, node in enumerate(self.G.nodes)}
        nx.set_node_attributes(self.G, {node: {'value': initial_values[node_mapping[node]]} for node in self.G.nodes})
        nx.set_node_attributes(self.G, {node: {'byzantine': False} for node in self.G.nodes})
        

    def set_byzantine_agents(self):
        n = self.G.order()
        num_byzantine = int(np.floor(n * self.percent_byzantine))
        byzantine_nodes = np.random.choice(list(self.G.nodes), size=num_byzantine, replace=False)
        for node in byzantine_nodes:
            self.G.nodes[node]['byzantine'] = True
            

    def calculate_global_average(self):
        values = [self.G.nodes[n]['value'] for n in self.G.nodes]
        return np.mean(values)

    def has_converged(self):
        gbl_avg = self.calculate_global_average()
        for node in self.G.nodes:
            if abs(self.G.nodes[node]['value'] - gbl_avg) > self.CONSENSUS_ERROR:
                return False
        return True
    
    def display_final_graph(self):
        values = []
        byzantines = []
        for node in self.G.nodes:
            values.append(self.G.nodes[node]['value'])
            if self.G.nodes[node]['byzantine'] is True:    
                byzantines.append('red')
            else:
                byzantines.append('blue')
           
        pos = nx.spring_layout(self.G)

        plt.figure(figsize=(10, 7))
        nx.draw(self.G, pos, with_labels=False, node_color=byzantines, cmap=plt.cm.viridis, node_size=700)
        
        labels = {node: f"{node}\n{self.G.nodes[node]['value']:.2f}\n{self.G.nodes[node]['byzantine']}" for node in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8, font_color="white")
    
        plt.title("Graph")
        plt.show()


    def plot_values_and_global_average(self, node_values_over_time, global_averages):
        plt.figure(figsize=(12, 8))

        for node, values in node_values_over_time.items():
            is_byzantine = self.G.nodes[node]['byzantine']  # Check if the node is Byzantine
            label = f"Node {node} ({'Byzantine' if is_byzantine else 'Normal'})"
            linestyle = '--' if is_byzantine else '-'
            alpha = 0.9 if is_byzantine else 0.7
            color = 'purple' if is_byzantine else None  # Use a distinct color for Byzantine nodes
            plt.plot(values, label=label, linestyle=linestyle, alpha=alpha, color=color)
        
        plt.plot(global_averages, label='Global Average', color='red', linewidth=2, marker='o')

        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Node Values and Global Average')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Legend")
        plt.tight_layout()  # Adjust layout to fit everything nicely
        plt.grid(True)
        plt.show()


    def has_converged_islands(self):
        connected_components = nx.connected_components(self.G)
    
        for component in connected_components:
            nodes_in_component = list(component)
            local_average = sum(self.G.nodes[node]['value'] for node in nodes_in_component) / len(nodes_in_component)
        
            for node in nodes_in_component:
                if abs(self.G.nodes[node]['value'] - local_average) > self.CONSENSUS_ERROR:
                    return False
    
        return True

    def track_values_and_averages(self):
        node_values_over_time = {node: [] for node in self.G.nodes}
        global_averages = []
    
        for node in self.G.nodes:
            node_values_over_time[node].append(self.G.nodes[node]['value'])
        global_averages.append(self.calculate_global_average())
    
        i = 0
    
        # Just learned that python lets you use functions as variables. That is awesome.
        convergence_check = self.has_converged if nx.is_connected(self.G) else self.has_converged_islands
        while not convergence_check():
            self.update_weighted_values()
            i += 1
            for node in self.G.nodes:
                node_values_over_time[node].append(self.G.nodes[node]['value'])
            global_averages.append(self.calculate_global_average())
        return node_values_over_time, global_averages

    def update_weighted_values(self):
        w_neighbor = self.NEIGHBOR_WEIGHT
        w_agent = self.AGENT_WEIGHT
        new_values = {}
        #Arithmetic Weighted average = sum(w_ix_i)/sum(w_i)
        for node in self.G.nodes:
            neighbors = list(self.G.neighbors(node))
            values = [self.G.nodes[n]['value'] for n in neighbors]
            total = w_neighbor*sum(values) + (w_agent * self.G.nodes[node]['value'])
            new_values[node] = total/((len(values) * w_neighbor) + w_agent) 
        #After all new values have been calculated replace each agents current value with the new average.
        for node, value in new_values.items():
            if self.G.nodes[node]['byzantine']:
                self.G.nodes[node]['value'] = 0  # Replace this with a function to handle Byzantine behavior
            else:
                self.G.nodes[node]['value'] = value

            
    def run_sim(self):
        self.set_rand_node_values()
        self.set_byzantine_agents()
        self.global_average = self.calculate_global_average()
        print(f"Initial global average: {self.global_average:.4f}")
    
        node_values_over_time, global_averages = self.track_values_and_averages()

        final_average = self.calculate_global_average()
        print(f"Final global average: {final_average:.4f}")

        self.plot_values_and_global_average(node_values_over_time, global_averages)
        self.display_final_graph()
        print("Consensus process complete!")
    
class Cyclic(Simulation):
    def __init__(self,order, percent_byzantine):
        super().__init__(order, percent_byzantine)
        self.G=nx.cycle_graph(order)
        self.run_sim()
        

class Kregular(Simulation):
    def __init__(self,order, percent_byzantine, degree):
        super().__init__(order, percent_byzantine)
        self.degree=degree
        self.G=nx.random_regular_graph(degree,order)
        self.run_sim()
        
class Binomial(Simulation):
    def __init__(self,order,percent_byzantine, probability):
        super().__init__(order, percent_byzantine)
        self.probability = probability
        self.G=nx.erdos_renyi_graph(order,probability)
        self.run_sim()
        
#Cyclic(10)
Kregular(20,0.1,3)
#Binomial(20,.1,.1)