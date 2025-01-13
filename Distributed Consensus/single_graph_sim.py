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
        self.self_weight = 5
        self.neighbor_weight = 1
        self.iterations = 0
        self.period = 4


    """
    ALTERNATIVE METHOD OF DOING THIS ALGORITHM

        say we have n agents in total, we create an nxn stochastic matrix such that
    
                   a1 a2 a3 ... an
                a1 [.2 .3 .1 ... .2 ]
                a2 [.5  0 .2 ... .1 ]
                a3 [.4 .2  0 ... .3 ]   
                .  [................]
                .  [................]
                an [.1 .2  0 ... .2 ]

        each entry represents ai's weight of aj.
        We then create a vector of all the "states" or values that each agent is holding
        and we perform matrix multiplication on the vector to get the next iteration of our system.
        we do this.
                        Weights         current state    next state
                   a1 a2 a3 ... an
                a1 [.2 .3 .1 ... .2 ]       [a1]          [a1]
                a2 [.5  0 .2 ... .1 ]       [a2]          [a2]
     agents     a3 [.4 .2  0 ... .3 ]       [a3]          [a3]
                .  [................]    x  [ .]   =      [ .]
                .  [................]       [ .]          [ .]
                an [.1 .2  0 ... .2 ]       [an]          [an]
    """




    def check_validity(self, neighbors):
        """  
        IDEA:
            Each node keeps a record of recieved data
            it will periodically analyze the record and if it notices suspicious behavior it will remove that node from its neighbors.
            It will then recalculate its value using the records of its own value and previously recieved values.
            
            Find value with the most highest difference values compared to self.value.
            if it has the highest difference for some percentage of other agents values we discard it.
            This matrix will tell us the highest difference.
            
               a1 a2 a3
            a1 [0 1 0 ]
            a2 [1 0 0 ]
            a3 [1 0 0 ]         
        """
        

    
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
        self.MAX_ITERATIONS = 1000 # If simulation cannot converge, it ends at this many iterations.
    

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
    
    # Displays the graph network.
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

    # Shows the convergence figure with matplotlib 
    def plot_values_and_global_average(self, node_values_over_time, global_averages, agent_averages):
        plt.figure(figsize=(12, 8))

        for node, values in node_values_over_time.items():
            is_byzantine = isinstance(self.G.nodes[node]['agent'], Byzantine)  # Check if the node is Byzantine
            label = f"Node {node} ({'Byzantine' if is_byzantine else 'Normal'})"
            linestyle = '--' if is_byzantine else '-'
            alpha = 0.9 if is_byzantine else 0.7
            color = 'purple' if is_byzantine else None  # Use a distinct color for Byzantine nodes
            plt.plot(values, label=label, linestyle=linestyle, alpha=alpha, color=color)
        
        plt.plot(global_averages, label='Global Average', color='red', linewidth=2, marker='o')
        plt.plot(agent_averages, label='Agent Average', color='blue', linewidth=2, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Node Values and Global Average')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Legend")
        plt.tight_layout()  # Adjust layout to fit everything nicely
        plt.grid(True)
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
                print("Maximum iterations reached. Exiting...")
                break
            self.update_weighted_values()
            i += 1
            for node in self.G.nodes:
                node_values_over_time[node].append(self.G.nodes[node]['agent'].value)
            global_averages.append(self.calculate_global_average())
            agent_averages.append(self.calculate_nonbyzantine_average())
        return node_values_over_time, global_averages, agent_averages

    # Iteratively averages each agent's value with its neighbors.
    def update_weighted_values(self):
        w_neighbor = self.NEIGHBOR_WEIGHT
        w_agent = self.AGENT_WEIGHT
        #Arithmetic Weighted average = sum(w_ix_i)/sum(w_i)
        for node in self.G.nodes:
            neighbors = [self.G.nodes[n]['agent'] for n in self.G.neighbors(node)]
            # We can change the averaging method used by our agents here.
            self.G.nodes[node]['agent'].value = self.G.nodes[node]['agent'].calculate_relative_trimmed_average(neighbors)
        for node in self.G.nodes:
            self.G.nodes[node]['agent'].shared_to_value()

    def run_sim(self):
        self.set_rand_node_values()
        self.set_byzantine_agents()
        self.global_average = self.calculate_global_average()
        print(f"Initial global average: {self.global_average:.4f}")
        node_values_over_time, global_averages, agent_averages = self.track_values_and_averages()
        final_average = self.calculate_global_average()
        print(f"Final global average: {final_average:.4f}")
        self.plot_values_and_global_average(node_values_over_time, global_averages, agent_averages)
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

#Binomial(40,.1,.05)
#Cyclic(50,.1)
Kregular(30,.05,4)
