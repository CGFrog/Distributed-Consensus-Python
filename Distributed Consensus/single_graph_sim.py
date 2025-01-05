from math import ceil
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

class Agent:
    def __init__(self, value):
        self.value = value
        self.shared_value = value
        self.self_weight = 1
        self.neighbor_weight = 1
        
    def calculate_average(self, neighbors):
        return (self.neighbor_weight * sum(neighbors) + self.value * self.self_weight)/(len(neighbors)*self.neighbor_weight+self.self_weight)
    
    def get_shared_value(self):
        return self.shared_value
    
    def shared_to_value(self):
        print(self.value)
        self.sharded_value = self.value

class Byzantine(Agent):
    def __init__(self, value):
        super().__init__(value)
        
    def get_shared_value(self):
        return 0



class Simulation:
    
    def __init__(self,order, percent_byzantine):
        self.order = order
        self.G = nx.empty_graph()
        self.CONSENSUS_ERROR = .2
        self.AGENT_WEIGHT = 1
        self.NEIGHBOR_WEIGHT = 1
        self.percent_byzantine = percent_byzantine
        
    def set_rand_node_values(self):
        initial_values = np.random.rand(len(self.G.nodes))
        # Map nodes to their index
        node_mapping = {}
        i = 0
        for node in self.G.nodes:
            node_mapping[node] = i
            i += 1
        for node in self.G.nodes:
            nx.set_node_attributes(self.G, {node: {'agent': Agent([node_mapping[node]])}})
        # Set the 'value' attribute for each node's 'agent'
        for node in self.G.nodes:
            self.G.nodes[node]['agent'].value = initial_values[node]

        

    def set_byzantine_agents(self):
        n = self.G.order()
        num_byzantine = int(np.floor(n * self.percent_byzantine))
        byzantine_nodes = np.random.choice(list(self.G.nodes), size=num_byzantine, replace=False)
        for node in byzantine_nodes:
            val = self.G.nodes[node]['agent'].value
            self.G.nodes[node]['agent'] = Byzantine(val)
    
            """
            Currently the byzantine agents automatically set the value to 0. T
            """

    def calculate_global_average(self):
        values = [self.G.nodes[n]['agent'].value for n in self.G.nodes]
        return np.mean(values)

    def has_converged(self):
        gbl_avg = self.calculate_global_average()
        for node in self.G.nodes:
            if abs(self.G.nodes[node]['agent'].value - gbl_avg) > self.CONSENSUS_ERROR:
                return False
        return True
    
    def display_final_graph(self):
        values = []
        byzantines = []
        for node in self.G.nodes:
            values.append(self.G.nodes[node]['agent'].value)
            if isinstance(self.G.nodes[node]['agent'], Byzantine):    
                byzantines.append('red')
            else:
                byzantines.append('blue')
           
        pos = nx.spring_layout(self.G)

        plt.figure(figsize=(10, 7))
        nx.draw(self.G, pos, with_labels=False, node_color=byzantines, cmap=plt.cm.viridis, node_size=700)
        
        labels = {node: f"{node}\n{self.G.nodes[node]['agent'].value:.2f}\n{isinstance(self.G.nodes[node]['agent'],Byzantine)}" for node in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8, font_color="white")
    
        plt.title("Graph")
        plt.show()


    def plot_values_and_global_average(self, node_values_over_time, global_averages):
        plt.figure(figsize=(12, 8))

        for node, values in node_values_over_time.items():
            is_byzantine = isinstance(self.G.nodes[node]['agent'], Byzantine)  # Check if the node is Byzantine
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
            local_average = sum(self.G.nodes[node]['agent'].value for node in nodes_in_component) / len(nodes_in_component)
        
            for node in nodes_in_component:
                if abs(self.G.nodes[node]['agent'].value - local_average) > self.CONSENSUS_ERROR:
                    return False
    
        return True

    def track_values_and_averages(self):
        node_values_over_time = {node: [] for node in self.G.nodes}
        global_averages = []
    
        for node in self.G.nodes:
            node_values_over_time[node].append(self.G.nodes[node]['agent'].value)
        global_averages.append(self.calculate_global_average())
    
        i = 0
    
        # Just learned that python lets you use functions as variables. That is awesome.
        convergence_check = self.has_converged if nx.is_connected(self.G) else self.has_converged_islands
        while not convergence_check():
            self.update_weighted_values()
            i += 1
            for node in self.G.nodes:
                node_values_over_time[node].append(self.G.nodes[node]['agent'].value)
            global_averages.append(self.calculate_global_average())
        return node_values_over_time, global_averages

    def update_weighted_values(self):
        w_neighbor = self.NEIGHBOR_WEIGHT
        w_agent = self.AGENT_WEIGHT
        #Arithmetic Weighted average = sum(w_ix_i)/sum(w_i)
        for node in self.G.nodes:
            self.G.nodes[node]['agent'].value = self.G.nodes[node]['agent'].calculate_average(list(self.G.neighbors(node)))
        for node in self.G.nodes:
            self.G.nodes[node]['agent'].shared_to_value()
                

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