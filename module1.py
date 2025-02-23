#Step 0: Prep for synthetic data
import torch

#generate synthetic data using y=0.5x+1+(randomized data)
epsilon = 0.1
one_vector = torch.ones(1,4,dtype=torch.float32)
x = torch.arange(1,5,1, dtype=torch.float32)
y=0.5*x+1
y=y+epsilon*(2*torch.rand(1,4,dtype=torch.float32)-0.5)
x = zip(one_vector,x)
dataset = zip(x,y)
#for (x0,x1),y in dataset:
#   print(x0.item(),x1.item(),y.item())
    
# Step 1: Represent Model
w = torch.tensor([0,1],dtype=torch.float32)
for (x0,x1), y in dataset:
    x=torch.tensor([x0,x1],dtype=torch.float32)
    predict = torch.matmul(w,x)
    print(f"Y={y} Prediction={predict}")