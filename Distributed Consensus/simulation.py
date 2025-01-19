from pickletools import read_stringnl_noescape_pair
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

class Simulation:
    def __init__(self,ind_var,data_points):
        self.ind_var = ind_var
        self.data_points = data_points 
        self.iterations = []
        self.ind_var_data = []
        self.CONSENSUS_ERROR = 0.001
        self.NEIGHBOR_WEIGHTS = 5
        self.AGENT_WEIGHT = 1
        
    def update_weighted_values(self, G):
        new_values = {}
        #Arithmetic Weighted average = sum(w_ix_i)/sum(w_i)
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            values = [G.nodes[n]['value'] for n in neighbors]
            total = self.NEIGHBOR_WEIGHTS*sum(values) + (self.AGENT_WEIGHT * G.nodes[node]['value'])
            new_values[node] = total/((len(values) * self.NEIGHBOR_WEIGHTS) + self.AGENT_WEIGHT) 
        #After all new values have been calculated replace each agents current value with the new average.
        for node, value in new_values.items():
            G.nodes[node]['value'] = value
    
    def global_average(self, G):
        values = [G.nodes[n]['value'] for n in G.nodes]
        return np.mean(values)

    def has_converged(self, G, global_average):
        for node in G.nodes:
            if abs(G.nodes[node]['value'] - global_average) > self.CONSENSUS_ERROR:
                return False
        return True
    
    def convergence_point(self, G, allow_islands):
        iterations = 0
        g_avg = self.global_average(G)
        convergence_check = self.has_converged if not allow_islands else self.has_converged_islands
        while not convergence_check(G, g_avg if not allow_islands else None):
            iterations += 1
            self.update_weighted_values(G)

        return iterations


    def has_converged_islands(self, G):
        connected_components = nx.connected_components(G)
        for component in connected_components:
            nodes_in_component = list(component)
            local_average = sum(G.nodes[node]['value'] for node in nodes_in_component) / len(nodes_in_component)
        
            for node in nodes_in_component:
                if abs(G.nodes[node]['value'] - local_average) > self.CONSENSUS_ERROR:
                    return False
        return True
    
    def set_rand_node_values(self, G):
        initial_values = np.random.rand(len(G.nodes))
        node_mapping = {node: i for i, node in enumerate(G.nodes)}
        nx.set_node_attributes(G, {node: {'value': initial_values[node_mapping[node]]} for node in G.nodes})

    def run_sim(self):
        pass
    
    def plot_iterations_vs_data(self, title="Iterations vs Data", xlabel="Independent Variable Data", ylabel="Iterations"):
        plt.figure(figsize=(8, 5))
        # Plot independent variable data (x-axis) vs. iterations (y-axis)
        plt.plot(self.ind_var_data, self.iterations, marker='o', linestyle='', color='b')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()
    
class Cyclic(Simulation):
    def __init__(self, ind_var,data_points):
        super().__init__(ind_var,data_points)
        self.run_sim()
        self.plot_iterations_vs_data(
            title=f"Cyclic Simulation Results",
            xlabel=self.ind_var, 
            ylabel="Iterations"
        )

    def run_sim(self):
        for d in range(self.data_points):
            print(d)
            rn = np.random.randint(1,100)
            G=nx.cycle_graph(rn)
            self.set_rand_node_values(G)
            self.iterations.append(self.convergence_point(G,False))
            self.ind_var_data.append(rn)

class Kregular(Simulation):
    def __init__(self,ind_var, data_points, allow_islands, fixed_value):
        super().__init__(ind_var, data_points)
        self.allow_islands = allow_islands
        self.MAX_VERTICES = 100
        self.fixed_value = fixed_value
        self.run_sim()
        self.plot_iterations_vs_data(
            title=f"K-Regular Simulation Results",
            xlabel=self.ind_var, 
            ylabel="Iterations"
        )
    
    def degree_ind_var(self):
        n = self.fixed_value
        d=0
        while len(self.iterations) < self.data_points:
            print(d)
            rk = np.random.randint(1, n*.85) #.85 is in place so that we dont get any values that take extreme amounts of time for graph generation
            G = nx.random_regular_graph(rk, n)
            self.set_rand_node_values(G)
            if not self.allow_islands and not nx.is_connected(G):
                continue
            self.iterations.append(self.convergence_point(G, self.allow_islands))
            self.ind_var_data.append(rk)
            d+=1
            
    def order_ind_var(self):
        k = self.fixed_value
        d=0
        while len(self.iterations) < self.data_points:
            print(d)
            rn = np.random.randint(k, k + self.MAX_VERTICES)
            if rn <= k:
                continue
            if (k * rn) % 2 == 1:
                continue
            G = nx.random_regular_graph(k, rn)
            self.set_rand_node_values(G)
            if not self.allow_islands and not nx.is_connected(G):
                continue
            self.iterations.append(self.convergence_point(G, self.allow_islands))
            self.ind_var_data.append(rn)
            d+=1

    def run_sim(self):
        if self.ind_var.lower() == "degree":
            self.degree_ind_var()
        else:
            self.order_ind_var()

                
class Binomial(Simulation):
    def __init__(self,ind_var,data_points,allow_islands, fixed_value):
        super().__init__(ind_var,data_points)
        self.allow_islands = allow_islands
        self.fixed_value = fixed_value
        self.run_sim()
        self.plot_iterations_vs_data(
            title=f"Binomial Simulation Results",
            xlabel=self.ind_var, 
            ylabel="Iterations"
        )
    
    def probability_ind_var(self):
        d=0
        n = self.fixed_value
        while len(self.iterations) < self.data_points:
            print (d)
            rp = np.random.rand()
            G = nx.erdos_renyi_graph(n,rp)
            self.set_rand_node_values(G)
            if not self.allow_islands and not nx.is_connected(G):
                continue
            self.iterations.append(self.convergence_point(G,self.allow_islands))
            self.ind_var_data.append(rp)
            d+=1
            
    def order_ind_var(self):
        d=0
        p = self.fixed_value
        while len(self.iterations) < self.data_points:
            print (d)
            rn = np.random.randint(1,100)
            G = nx.erdos_renyi_graph(rn, p)
            self.set_rand_node_values(G)
            if not self.allow_islands and not nx.is_connected(G):
                continue
            self.iterations.append(self.convergence_point(G,self.allow_islands))
            self.ind_var_data.append(rn)
            d+=1
    

    def run_sim(self):
        if self.ind_var.lower() == "probability":
            self.probability_ind_var()
        else:
            self.order_ind_var()


#Cyclic("order",100)

#Kregular("order",1000,False,3)
Kregular("degree",100,False,100) # Does not always display all values

#Very sensitive to convergence error, takes extremely long times to converge to an error of .001 when islands are not allowed

#Binomial("probability", 100, False, 10) 
#Binomial("order", 100, False,.5)         
