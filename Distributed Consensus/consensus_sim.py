from math import ceil
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy as sp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import Counter

class Agent:
    def __init__(self, value):
        self.value = value
        self.initial_value = value
        self.shared_value = value
        self.self_weight = 1
        self.neighbor_weight = 1
        self.iterations = 0
        self.period = 4
        self.global_averages = []
        self.agent_averages = []
        self.special_var = ""
        self.f = 1
        self.byzantine_count = 0
        self.total_neighbors = 0
        self.can_converge = None
        self.n_byz_count = {}

    def calculate_average(self, neighbors):
        neighbor_values = [neighbor.get_shared_value() for neighbor in neighbors]
        self.iterations += 1
        return (self.neighbor_weight * sum(neighbor_values) + self.value * self.self_weight) / (len(neighbor_values) * self.neighbor_weight + self.self_weight)

    def calculate_mode(self, neighbors):
        neighbor_values = [neighbor.get_shared_value() for neighbor in neighbors]
        neighbor_values.append(self.value)
        counts = Counter(neighbor_values) 
        max_count = max(counts.values())
        most_common = [val for val, count in counts.items() if count == max_count]
        if len(most_common) > 1:
            return self.value
    
        return most_common[0]

    def calculate_trimmed_average(self, neighbors):
        neighbor_values = [neighbor.get_shared_value() for neighbor in neighbors]
        for _ in range(self.f):
            if ( not neighbor_values):
                break
            neighbor_values.remove(max(neighbor_values))
            if ( not neighbor_values):
                break
            neighbor_values.remove(min(neighbor_values))
        self.iterations += 1
        return (self.neighbor_weight * sum(neighbor_values) + self.value * self.self_weight) / (len(neighbor_values) * self.neighbor_weight + self.self_weight)
    
    def calculate_all_trimmed_average(self, neighbors):
        all_values = [neighbor.get_shared_value() for neighbor in neighbors]
        all_values.append(self.shared_value)

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

    def calculate_relative_trimmed_average(self, neighbors):
        neighbor_values = [neighbor.get_shared_value() for neighbor in neighbors]
        max_diff = neighbor_values[0] 
        for _ in range(2*self.f):
            if ( not neighbor_values):
                    break
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

    def _byzantine_neighbors(self,neighbors):
        byzantine_count = 0
        for neighbor in neighbors:
            if isinstance(neighbor, Byzantine):
                byzantine_count += 1    
        return byzantine_count
        
    def valid_byzantine_ratio(self, neighbors):
        self.total_neighbors = len(neighbors)
        self.byzantine_count = self._byzantine_neighbors(neighbors)
        if 3*self.byzantine_count+1 <= len(neighbors):
            self.can_converge = True
            return True
        self.can_converge = False
        return False

    def get_neighbor_byzantine_count(self, neighbors, G):
        self.n_byz_count = {} 
        for neighbor in neighbors:
            agent_obj = G.nodes[neighbor]['agent']
            self.n_byz_count[neighbor] = agent_obj.byzantine_count

class Byzantine(Agent):
    def __init__(self, value):
        super().__init__(value)
    def __generate_random_normal(self):
        N = 1000
        mu, sigma = 0.5, 0.15
        data = np.random.normal(mu, sigma)

        data = np.clip(data, 0, 1)
        return data

    def get_shared_value(self):
        return np.random.rand()

    def get_shared_value_gaussian(self):
        return self.__generate_random_normal()

class Simulation:
    def __init__(self,order, amount_byzantine, averaging_function,f):
        self.order = order
        self.G = nx.empty_graph()
        self.CONSENSUS_ERROR = 0
        self.AGENT_WEIGHT = 1
        self.NEIGHBOR_WEIGHT = 1
        self.amount_byzantine = amount_byzantine
        self.iterations = 0
        self.meets_convergence_req = None
        self.init_global_average = 0
        self.end_global_average = 0
        self.MAX_ITERATIONS = 500 # If simulation cannot converge, it ends at this many iterations.
        self.averaging_function = averaging_function
        self.mode = 0
        self.f = f

    def set_consensus_error(self, consensus_error):
        self.CONSENSUS_ERROR = consensus_error


    def _set_agent_values(self):
        if self.averaging_function=="calculate_mode":self.set_rand_node_integer_values()
        else: self.set_rand_node_values()

    def set_rand_node_values(self):
        initial_values = np.random.rand(len(self.G.nodes))
        node_mapping = {node: i for i, node in enumerate(self.G.nodes)}
    
        for node in self.G.nodes:
            nx.set_node_attributes(self.G, {node: {'agent': Agent(initial_values[node_mapping[node]])}})
    
        for node in self.G.nodes:
            self.G.nodes[node]['agent'].value = initial_values[node_mapping[node]]
            self.G.nodes[node]['agent'].f = self.f

    def set_rand_node_integer_values(self, low=0, high=10):
        lambda_ = (high + low) / 2
        initial_values = np.clip(np.random.poisson(lambda_, len(self.G.nodes)), low, high - 1)
        node_mapping = {node: i for i, node in enumerate(self.G.nodes)}
        self.mode = max(set(list(initial_values)), key=list(initial_values).count)
        for node in self.G.nodes:
            nx.set_node_attributes(self.G, {node: {'agent': Agent(initial_values[node_mapping[node]])}})

        for node in self.G.nodes:
            self.G.nodes[node]['agent'].value = initial_values[node_mapping[node]]

    def set_byzantine_agents(self):
        n = len(self.G.nodes)
        node_list = list(self.G.nodes)
        byzantine_nodes = np.random.choice(node_list, size=self.amount_byzantine, replace=False)
        for node in byzantine_nodes:
            node = int(node)
            if node in self.G.nodes:
                val = self.G.nodes[node]['agent'].value 
                self.G.nodes[node]['agent'] = Byzantine(val)
            else:
                print(f"Warning: Node {node} not found in graph!")

    # Calculates global average with byzantine agents.
    def calculate_global_average(self):
        values = [self.G.nodes[n]['agent'].value for n in self.G.nodes]
        return np.mean(values)

    # Calculates global average without byzantine agents.
    def calculate_nonbyzantine_average(self):
        values = [self.G.nodes[n]['agent'].value for n in self.G.nodes if not isinstance(self.G.nodes[n]['agent'], Byzantine)]
        return sum(values) / len(values)

    def calculate_pure_agent_average(self):
        values = [self.G.nodes[n]['agent'].value for n in self.G.nodes if not isinstance(self.G.nodes[n]['agent'] and self.G.nodes[n]['agent'].byzantine_count==0, Byzantine)]
        return sum(values) / len(values)

    # Checks if all agent's values are within some margin of error of the global average, if they are, the simulation ends.
    def has_converged(self):
        gbl_avg = self.calculate_nonbyzantine_average()
        for node in self.G.nodes:
            if abs(self.G.nodes[node]['agent'].value - gbl_avg) > self.CONSENSUS_ERROR:
                return False
        return True

    def __can_converge(self):
        can_converge=True
        for agent in self.G.nodes:
            agent_obj = self.G.nodes[agent]['agent']
            neighbor_agents = [self.G.nodes[n]['agent'] for n in self.G.neighbors(agent)]
            if not agent_obj.valid_byzantine_ratio(neighbor_agents):
                can_converge=False #We dont stop prematurely because this function needs to run for all nodes (bad code on my part but will fix later probably)
        return can_converge

    def _compute_byzantine_counts(self):
        return {
            node: sum(1 for neighbor in self.G.neighbors(node) if isinstance(self.G.nodes[neighbor]['agent'], Byzantine))
            for node in self.G.nodes
        }

    def _compute_node_styles(self, byzantine_counts):
        max_byzantine = max(byzantine_counts.values()) if byzantine_counts else 1
        node_colors = {}
        node_linewidths = {}

        for node in self.G.nodes:
            if isinstance(self.G.nodes[node]['agent'], Byzantine):
                node_colors[node] = "firebrick"
                node_linewidths[node] = 1
            else:
                intensity = byzantine_counts[node] / max_byzantine if max_byzantine > 0 else 0
                node_colors[node] = cm.YlGnBu(0.5 + 0.8 * (intensity / 2))
                node_linewidths[node] = 1 + 2 * intensity
    
        return node_colors, node_linewidths

    def _plot_node_values(self, ax, node_colors, node_linewidths):
        for node, values in self.node_values_over_time.items():
            agent = self.G.nodes[node]['agent']
            is_byzantine = isinstance(agent, Byzantine)

            ax.plot(values, label="_nolegend_",
                    linestyle='--' if is_byzantine else '-', alpha=0.9 if is_byzantine else 0.7,
                    color=node_colors[node], linewidth=node_linewidths[node])

        ax.plot(self.global_averages, label='Global Average', color='red', linewidth=2, marker='o')
        ax.plot(self.agent_averages, label='Agent Average', color='blue', linewidth=2, marker='o')
        ax.plot(self.pure_averages, label='No Byzantine Neighbors Average', color='cyan', linewidth=2, marker='o')

        ax.set(xlabel='Iteration', ylabel='Value',
               title=f'Node Values and Global Average\n'
                     f'Difference: {abs(self.init_global_average - self.end_global_average):.4f} Mode: {self.mode}\n'
                     f'Init: {self.init_global_average:.4f} End: {self.end_global_average:.4f}')
    
        ax.legend(loc='upper center', bbox_to_anchor=(1, 0.5), title="Legend")
        ax.grid(True)


    def _draw_graph(self, ax, byzantine_counts, node_colors, can_converge):
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=False, node_color=list(node_colors.values()), node_size=1350, ax=ax)

        labels = {
            node: (f"{node}\n{self.G.nodes[node]['agent'].value:.2f}\n"
                f"{'Byzantine' if isinstance(self.G.nodes[node]['agent'], Byzantine) else 'Normal'}\n"
                f"F={byzantine_counts[node]}")
            for node in self.G.nodes
        }
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8, font_color="white", ax=ax)
        ax.set_title(f"Order: {self.order} | Iterations: {self.iterations}\n"
            f"Error: {self.CONSENSUS_ERROR} | Byzantines: {self.amount_byzantine}\n"
            f"Degree/Probability: {self.special_var}\n"
            f"$3F+1\leq n$ {can_converge}")

    def save_graph(self,name):
        G_copy = self.G.copy()
        for node in G_copy.nodes:
            agent = G_copy.nodes[node].get('agent')
            if isinstance(agent, Byzantine):
                G_copy.nodes[node]['agent_type'] = "Byzantine"
                G_copy.nodes[node]['agent_value'] = str(agent.initial_value)
            elif isinstance(agent, Agent):
                G_copy.nodes[node]['agent_type'] = "Normal"
                G_copy.nodes[node]['agent_value'] = str(agent.initial_value)
            if 'agent' in G_copy.nodes[node]:
                del G_copy.nodes[node]['agent']

        nx.write_graphml(G_copy, name)
        print("Graph Saved")

    def display_combined_plots(self):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        byzantine_counts = self._compute_byzantine_counts()
        node_colors, node_linewidths = self._compute_node_styles(byzantine_counts)
        self._plot_node_values(axes[0], node_colors, node_linewidths)      
        self._draw_graph(axes[1], byzantine_counts, node_colors, self.meets_convergence_req)
        return fig 
    
    def get_stats(self):
        stats = {}
        stats["Order"]=self.order
        stats["Byzantines"]=self.amount_byzantine
        stats["Iterations"] = self.iterations
        return stats
        
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
        pure_agent_averages = []
        for node in self.G.nodes:
            node_values_over_time[node].append(self.G.nodes[node]['agent'].value)
        global_averages.append(self.calculate_global_average())
        agent_averages.append(self.calculate_nonbyzantine_average())
        i = 0
        convergence_check = self.has_converged if nx.is_connected(self.G) else self.has_converged_islands
        while not convergence_check():
            if i >= self.MAX_ITERATIONS:
                break
            self.update_weighted_values()
            i += 1
            for node in self.G.nodes:
                node_values_over_time[node].append(self.G.nodes[node]['agent'].value)
            global_averages.append(self.calculate_global_average())
            agent_averages.append(self.calculate_nonbyzantine_average())
            pure_agent_averages.append(self.calculate_pure_agent_average())
        self.iterations = i
        return node_values_over_time, global_averages, agent_averages, pure_agent_averages

    # Iteratively averages each agent's value with its neighbors.
    def update_weighted_values(self):
        w_neighbor = self.NEIGHBOR_WEIGHT
        w_agent = self.AGENT_WEIGHT
        for node in self.G.nodes:
            agent = self.G.nodes[node]['agent']
            neighbors = [self.G.nodes[n]['agent'] for n in self.G.neighbors(node)]
            averaging_method = getattr(agent, self.averaging_function)
            agent.value = averaging_method(neighbors)
        for node in self.G.nodes:
            self.G.nodes[node]['agent'].shared_to_value()

    def get_iterations(self):
        return self.iterations

    def run_sim(self):
        if (not isinstance(self,LoadedGraph)):
            self._set_agent_values()
            self.set_byzantine_agents()
        self.init_global_average = self.calculate_global_average()
        print(f"Initial global average: {self.init_global_average:.4f}")
        self.node_values_over_time, self.global_averages, self.agent_averages,self.pure_averages = self.track_values_and_averages()
        self.end_global_average = self.calculate_global_average()
        print(f"Final global average: {self.end_global_average:.4f}")
        print("Consensus process complete!")
        self.meets_convergence_req = self.__can_converge()
        for agent in self.G.nodes:
            agent_data = self.G.nodes[agent]
            agent_data['agent'].get_neighbor_byzantine_count(self.G.neighbors(agent),self.G)

class Cyclic(Simulation):
    def __init__(self,order, amount_byzantine, averaging_function,f):
        super().__init__(order, amount_byzantine, averaging_function=averaging_function,f=f)
        self.G=nx.cycle_graph(order)
        self.special_var=None

class Kregular(Simulation):
    def __init__(self,order, amount_byzantine, averaging_function,f, degree):
        super().__init__(order, amount_byzantine, averaging_function,f)
        self.degree=degree
        self.G=nx.random_regular_graph(degree,order)
        self.special_var = degree

class Binomial(Simulation):
    def __init__(self,order,amount_byzantine, averaging_function, f, probability):
        super().__init__(order, amount_byzantine,averaging_function, f)
        self.probability = probability
        self.G=nx.erdos_renyi_graph(order,probability)
        self.special_var = probability

class Small_World(Simulation):
    def __init__(self, order, amount_byzantine,averaging_function,f, k, p):
        super().__init__(order, amount_byzantine, averaging_function, f)
        self.k = k  # Each node is connected to `k` nearest neighbors
        self.p = p  # Probability of rewiring an edge
        self.G = nx.watts_strogatz_graph(self.order, self.k, self.p)
        self.special_var = f"k={k}, p={p}"

class LoadedGraph(Simulation):
    def __init__(self, filename, averaging_function,f):
        super().__init__(order=0, amount_byzantine=0, averaging_function=averaging_function, f=f)
        self.G = nx.read_graphml(filename)
        self.order = self.G.number_of_nodes()
        self.special_var = f"Loaded from {filename}"
        for node in self.G.nodes:
            agent_type = str(self.G.nodes[node].get("agent_type", "Normal"))
            agent_value = float(self.G.nodes[node].get("agent_value", 0.0))
            if agent_type.strip() == "Byzantine":
                self.G.nodes[node]['agent'] = Byzantine(agent_value)
                self.amount_byzantine+=1
            else:
                self.G.nodes[node]['agent'] = Agent(agent_value)

        print(f"Graph loaded from {filename}. Nodes: {self.order}")