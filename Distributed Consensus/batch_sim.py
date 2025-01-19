import single_graph_sim as sg
import numpy as np
import matplotlib.pyplot as plt

class Batch_Simulation():
    def __init__(self, data_points):
        self.data_points = data_points
        self.CONSENSUS_ERROR = 0.01
        self.iteration_count = []
        self.indvar_count = []
        self.degree = 5
        self.order = 50
        self.byzantine_percent = .1
        self.probability = .3
        
    def run_batch_sim(self):
        for i in range(self.data_points):
            ind_var = np.random.uniform(0, 0.2)
            #ind_var = np.random.randint(4,20)
            graph = sg.Kregular(self.order, ind_var, self.degree)
            graph.set_consensus_error(self.CONSENSUS_ERROR)
            
            graph.run_sim()

            self.iteration_count.append(graph.get_iterations())
            self.indvar_count.append(ind_var)

    def display_batch_sim(self):
        plt.scatter( self.iteration_count,self.indvar_count, label="Simulation Data")
        plt.xlabel("Iteration Count")
        plt.ylabel("Independent Variable")
        plt.title("Batch Simulation Results")
        plt.legend()
        plt.show()
        



bs = Batch_Simulation(50)

bs.run_batch_sim()
bs.display_batch_sim()
