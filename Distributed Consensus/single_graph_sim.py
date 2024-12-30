import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

class Simulation:
    def __init__(self,order):
        self.order = order
        self.G = nx.empty_graph()
        self.CONSENSUS_ERROR = 0.001
        
    def set_rand_node_values(self):
        initial_values = np.random.rand(len(self.G.nodes))
        node_mapping = {node: i for i, node in enumerate(self.G.nodes)}
        nx.set_node_attributes(self.G, {node: {'value': initial_values[node_mapping[node]]} for node in self.G.nodes})

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
        values = [self.G.nodes[node]['value'] for node in self.G.nodes]
        pos = nx.spring_layout(self.G)
    
        plt.figure(figsize=(10, 7))
        nx.draw(self.G, pos, with_labels=False, node_color=values, cmap=plt.cm.viridis, node_size=700)
    
        labels = {node: f"{node}\n{self.G.nodes[node]['value']:.2f}" for node in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=10, font_color="white")
    
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Node Values")
        plt.title("Graph")
        plt.show()


    def plot_values_and_global_average(self,node_values_over_time, global_averages):
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
    
        if nx.nx.is_connected(self.G):
            while not self.has_converged():
                self.update_weighted_values()
                i+=1
                for node in self.G.nodes:
                    node_values_over_time[node].append(self.G.nodes[node]['value'])
                global_averages.append(self.calculate_global_average())
        # Try to rewrite this to reduce code duplication. 
        else:
            while not self.has_converged_islands():
                self.update_weighted_values()
                i+=1
                for node in self.G.nodes:
                    node_values_over_time[node].append(self.G.nodes[node]['value'])
                global_averages.append(self.calculate_global_average())


        return node_values_over_time, global_averages

    def update_weighted_values(self):
        w_neighbor = 1
        w_agent = 1
        new_values = {}
        #Arithmetic Weighted average = sum(w_ix_i)/sum(w_i)
        for node in self.G.nodes:
            neighbors = list(self.G.neighbors(node))
            values = [self.G.nodes[n]['value'] for n in neighbors]
            total = w_neighbor*sum(values) + (w_agent * self.G.nodes[node]['value'])
            new_values[node] = total/((len(values) * w_neighbor) + w_agent) 
        #After all new values have been calculated replace each agents current value with the new average.
        for node, value in new_values.items():
            self.G.nodes[node]['value'] = value
            
    def run_sim(self):
        self.set_rand_node_values()
        self.global_average = self.calculate_global_average()
        print(f"Initial global average: {self.global_average:.4f}")
    
        node_values_over_time, global_averages = self.track_values_and_averages()

        final_average = self.calculate_global_average()
        print(f"Final global average: {final_average:.4f}")

        self.plot_values_and_global_average(node_values_over_time, global_averages)
        self.display_final_graph()
        print("Consensus process complete!")
    
class Cyclic(Simulation):
    def __init__(self,order):
        super().__init__(order)
        self.G=nx.cycle_graph(order)
        self.run_sim()
        

class Kregular(Simulation):
    def __init__(self,order,degree):
        super().__init__(order)
        self.degree=degree
        self.G=nx.random_regular_graph(degree,order)
        self.run_sim()
        
class Binomial(Simulation):
    def __init__(self,order,probability):
        super().__init__(order)
        self.probability = probability
        self.G=nx.erdos_renyi_graph(order,probability)
        self.run_sim()
        
#Cyclic(10)
#Kregular(20,3)
Binomial(10,.2)        