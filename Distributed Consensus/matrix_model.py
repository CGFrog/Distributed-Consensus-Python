import numpy as np

"""
Matrix representation of the consensus algorithm in single_graph_sim.py
"""

class System:
    def __init__(self):
        B = ([1,2,3,4,5])
        n = 1/3
        A = np.array( [
            [n,n,0,0,n],
            [n,n,n,0,0],
            [0,n,n,n,0],
            [0,0,n,n,n],
            [n,0,0,n,n]
            ])
        
        for i in range(1000):
            B = np.matmul(A, B)
        
        print(B)
      
System()


