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
        self.iterations = 0
        self.period = 4
        self.node_values_over_time = []
        self.global_averages = []
        self.agent_averages = []
        self.special_var = ""

    # Weighted Average.
    def calculate_average(self, neighbors):
        neighbor_values = [neighbor.get_shared_value() for neighbor in neighbors]
        self.iterations += 1
        return (self.neighbor_weight * sum(neighbor_values) + self.value * self.self_weight) / (len(neighbor_values) * self.neighbor_weight + self.self_weight)

    #Trimmed Weighted Average, removes max and min element.
    def calculate_trimmed_average(self, neighbors):
        neighbor_values = [neighbor.get_shared_value() for neighbor in neighbors]
        neighbor_values.remove(max(neighbor_values))
        neighbor_values.remove(min(neighbor_values))
        self.iterations += 1
        return (self.neighbor_weight * sum(neighbor_values) + self.value * self.self_weight) / (len(neighbor_values) * self.neighbor_weight + self.self_weight)
    
    #Trimmed Weighted Average, removes max and min element including itself.
    def calculate_all_trimmed_average(self, neighbors):
        all_values = [neighbor.get_shared_value() for neighbor in neighbors]
        all_values.append(self.shared_value)  # Add the self-value

        all_values.remove(max(all_values))
        all_values.remove(min(all_values))

        weighted_values = [
            value * (self.self_weight if value == self.shared_value else self.neighbor_weight)
            for value in all_values
        ]
        total_weight = sum(
            self.self_weight if value == self.shared_value else self.neighbor_weight
            for value in all_values
        )
        return sum(weighted_values) / total_weight

     #Trimmed Weighted Average, removes element that has the largest difference.
    def calculate_relative_trimmed_average(self, neighbors):
        neighbor_values = [neighbor.get_shared_value() for neighbor in neighbors]
        max_diff = neighbor_values[0] 
        for n in neighbor_values:
            if (abs(n-self.value) > abs(max_diff-self.value)):
                max_diff = n                
        neighbor_values.remove(max_diff)
        self.iterations += 1
        return (self.neighbor_weight * sum(neighbor_values) + self.value * self.self_weight) / (len(neighbor_values) * self.neighbor_weight + self.self_weight)

    def get_shared_value(self):
        return self.shared_value
    
    def shared_to_value(self):
        self.shared_value = self.value

class Byzantine(Agent):
    def __init__(self, value):
        super().__init__(value)
    
    # This function determines what value the byzantine agent sends.
    def get_shared_value(self):
        return np.random.rand()

class Simulation:
    def __init__(self,order, percent_byzantine):
        self.order = order
        self.G = nx.empty_graph()
        self.CONSENSUS_ERROR = .001
        self.AGENT_WEIGHT = 1
        self.NEIGHBOR_WEIGHT = 1
        self.percent_byzantine = percent_byzantine
        self.iterations = 0
        self.init_global_average = 0
        self.end_global_average = 0
        self.MAX_ITERATIONS = 10000 # If simulation cannot converge, it ends at this many iterations.
    
    def set_consensus_error(self, consensus_error):
        self.CONSENSUS_ERROR = consensus_error


    # Initializes all agents with a random value. 
    def set_rand_node_values(self):
        initial_values = np.random.rand(len(self.G.nodes))
        # Map nodes to their index
        node_mapping = {}
        i = 0
        for node in self.G.nodes:
            node_mapping[node] = i
            i += 1
        for node in self.G.nodes:
            nx.set_node_attributes(self.G, {node: {'agent': Agent(initial_values[node])}})
        for node in self.G.nodes:
            self.G.nodes[node]['agent'].value = initial_values[node]

    # Converts some percentage of agents to byzantine agents.
    def set_byzantine_agents(self):
        n = self.G.order()
        num_byzantine = int(np.floor(n * self.percent_byzantine))
        byzantine_nodes = np.random.choice(list(self.G.nodes), size=num_byzantine, replace=False)
        for node in byzantine_nodes:
            val = self.G.nodes[node]['agent'].value
            self.G.nodes[node]['agent'] = Byzantine(val)
    
    # Calculates global average with byzantine agents.
    def calculate_global_average(self):
        values = [self.G.nodes[n]['agent'].value for n in self.G.nodes]
        return np.mean(values)

    # Calculates global average without byzantine agents.
    def calculate_nonbyzantine_average(self):
        values = [self.G.nodes[n]['agent'].value for n in self.G.nodes if not isinstance(self.G.nodes[n]['agent'], Byzantine)]
        return sum(values) / len(values)

    # Checks if all agent's values are within some margin of error of the global average, if they are, the simulation ends.
    def has_converged(self):
        gbl_avg = self.calculate_nonbyzantine_average()
        for node in self.G.nodes:
            if abs(self.G.nodes[node]['agent'].value - gbl_avg) > self.CONSENSUS_ERROR:
                return False
        return True

    def display_combined_plots(self):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        ax1 = axes[0]
        for node, values in self.node_values_over_time.items():
            is_byzantine = isinstance(self.G.nodes[node]['agent'], Byzantine)  # Check if the node is Byzantine
            label = f"Node {node} ({'Byzantine' if is_byzantine else 'Normal'})"
            linestyle = '--' if is_byzantine else '-'
            alpha = 0.9 if is_byzantine else 0.7
            color = 'purple' if is_byzantine else None  # Use a distinct color for Byzantine nodes
            ax1.plot(values, label=label, linestyle=linestyle, alpha=alpha, color=color)

        ax1.plot(self.global_averages, label='Global Average', color='red', linewidth=2, marker='o')
        ax1.plot(self.agent_averages, label='Agent Average', color='blue', linewidth=2, marker='o')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Value')
        ax1.set_title(f'Node Values and Global Average\nDifference: {abs(self.init_global_average - self.end_global_average)*100:.4f}%\n Init: {self.init_global_average:.4f} End: {self.end_global_average:.4f}')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Legend")
        ax1.grid(True)

        ax2 = axes[1]
        values = []
        byzantines = []
        for node in self.G.nodes:
            values.append(self.G.nodes[node]['agent'].value)
            if isinstance(self.G.nodes[node]['agent'], Byzantine):    
                byzantines.append('red')
            else:
                byzantines.append('blue')
    
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=False, node_color=byzantines, cmap=plt.cm.viridis, node_size=700, ax=ax2)
        labels = {
            node: f"{node}\n{self.G.nodes[node]['agent'].value:.2f}\n{'Byzantine' if isinstance(self.G.nodes[node]['agent'], Byzantine) else 'Normal'}"
            for node in self.G.nodes
        }
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8, font_color="white", ax=ax2)
        ax2.set_title(f"Graph\n Order: {self.order} | Iterations:{self.iterations}\n Error: {self.CONSENSUS_ERROR} | Byzantines: {np.floor(self.percent_byzantine * self.order)}\n Degree/Probability: {self.special_var}")

        #plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    # If a graph contains islands, we will check for convergence of each island.
    def has_converged_islands(self):
        connected_components = nx.connected_components(self.G)
        for component in connected_components:
            nodes_in_component = list(component)
            local_average = sum(self.G.nodes[node]['agent'].value for node in nodes_in_component) / len(nodes_in_component) 
            for node in nodes_in_component:
                if abs(self.G.nodes[node]['agent'].value - local_average) > self.CONSENSUS_ERROR:
                    return False
        return True

    def get_initial_global_average(self):
        return self.init_global_average

    def get_final_global_average(self):
        return self.end_global_average

    def track_values_and_averages(self):
        node_values_over_time = {node: [] for node in self.G.nodes}
        global_averages = []
        agent_averages = []
        for node in self.G.nodes:
            node_values_over_time[node].append(self.G.nodes[node]['agent'].value)
        global_averages.append(self.calculate_global_average())
        agent_averages.append(self.calculate_nonbyzantine_average())
        i = 0
        
        # Just learned that python lets you use functions as variables. That is awesome.
        convergence_check = self.has_converged if nx.is_connected(self.G) else self.has_converged_islands
        while not convergence_check():
            if i >= self.MAX_ITERATIONS:
                #print("Maximum iterations reached. Exiting...")
                break
            self.update_weighted_values()
            i += 1
            for node in self.G.nodes:
                node_values_over_time[node].append(self.G.nodes[node]['agent'].value)
            global_averages.append(self.calculate_global_average())
            agent_averages.append(self.calculate_nonbyzantine_average())
        self.iterations = i
        return node_values_over_time, global_averages, agent_averages

    # Iteratively averages each agent's value with its neighbors.
    def update_weighted_values(self):
        w_neighbor = self.NEIGHBOR_WEIGHT
        w_agent = self.AGENT_WEIGHT
        #Arithmetic Weighted average = sum(w_ix_i)/sum(w_i)
        for node in self.G.nodes:
            neighbors = [self.G.nodes[n]['agent'] for n in self.G.neighbors(node)]
            # We can change the averaging method used by our agents here.
            self.G.nodes[node]['agent'].value = self.G.nodes[node]['agent'].calculate_trimmed_average(neighbors)
        for node in self.G.nodes:
            self.G.nodes[node]['agent'].shared_to_value()
    
    def get_iterations(self):
        return self.iterations

    def run_sim(self):
        self.set_rand_node_values()
        self.set_byzantine_agents()
        self.init_global_average = self.calculate_global_average()
        print(f"Initial global average: {self.init_global_average:.4f}")
        self.node_values_over_time, self.global_averages, self.agent_averages = self.track_values_and_averages()
        self.end_global_average = self.calculate_global_average()
        print(f"Final global average: {self.end_global_average:.4f}")

        print("Consensus process complete!")

class Cyclic(Simulation):
    def __init__(self,order, percent_byzantine):
        super().__init__(order, percent_byzantine)
        self.G=nx.cycle_graph(order)

class Kregular(Simulation):
    def __init__(self,order, percent_byzantine, degree):
        super().__init__(order, percent_byzantine)
        self.degree=degree
        self.G=nx.random_regular_graph(degree,order)
        self.special_var = degree

class Binomial(Simulation):
    def __init__(self,order,percent_byzantine, probability):
        super().__init__(order, percent_byzantine)
        self.probability = probability
        self.G=nx.erdos_renyi_graph(order,probability)
        self.special_var = probability

#graph = Binomial(40,.15,.4)
#graph = Cyclic(30,.05)
graph = Kregular(55,.05,6)
graph.run_sim()
graph.display_combined_plots()