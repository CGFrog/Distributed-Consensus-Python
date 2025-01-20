import numpy as np

"""
    ALTERNATIVE (BETTER) METHOD OF DOING THIS ALGORITHM - IMPLEMENT THIS VERSION NEXT

        Say we have n agents in total, we create an nxn stochastic matrix such that
        each entry represents ai's weight of aj. (ith row jth column)
        
        Example:
        
                     a1 a2 a3 ... an
                a1 [.2 .3 .1 ... .2 ]
                a2 [.5  0 .2 ... .1 ]
                a3 [.4 .2  0 ... .3 ]   
                .  [................]
                .  [................]
                an [.1 .2  0 ... .2 ]

        We then create a vector of all the "states" or values that each agent is holding
        and we perform matrix multiplication on the vector to get the next iteration of our system.
        we do this. ai^k means the ith agent at the kth state.
        
                         Weights           Current State       Next State
                     w1 w2 w3 ... wn
                a1 [ .2 .3 .1 ... .2 ]         [a1^1]            [a1^2]
                a2 [ .5  0 .2 ... .1 ]         [a2^1]            [a2^2]
     Agents     a3 [ .4 .2  0 ... .3 ]         [a3^1]            [a3^2]
                .  [ ................]    x    [ .^1]     =      [ .^2]
                .  [ ................]         [ .^1]            [ .^2]
                an [ .1 .2  0 ... .2 ]         [an^1]            [an^2]
                
       GENERATING A RANDOM STOCHASTIC MATRIX
       -------------------------------------
       For each agent in our graph we will genrate random weights using the rand function
       we will then normalize them st their sum = 1
       we return this list as a dictionary with each key being the index of the agent in the graph and the value being the stochastic weight.
       
       We create a matrix in simulation class and append the weights of agent ai to the corresponding columns of the other agents weights.
       
       ----------------------
        MATRIX MULTIPLICATION
       ----------------------
            # Define two matrices
            A = np.array([[1, 2], [3, 4]])
            B = np.array([[5, 6], [7, 8]])

            # Perform matrix multiplication
            result = np.matmul(A, B)
        ------------------------------------

"""
    
class System:
    def __init__(self):
        self.one_to_five()
    
    def one_to_five(self):
        B = ([1,2,3,4,5])
        n = 1/3
        A = np.array( [
            [n,n,0,0,n],
            [n,n,n,0,0],
            [0,n,n,n,0],
            [0,0,n,n,n],
            [n,0,0,n,n]
            ])
        for i in range(32):
            B = np.matmul(A, B)
        print(B)

System()
# 1,18,32,49,