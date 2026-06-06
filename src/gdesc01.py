"""

TODO:

So we've made pytorch function
It has autodetermination of gradient

Determined function does not work

So now what

Address in email
* previous item was not actually anything like gradient descent

Tests to run:
* grad descent on straight up counting function
* test gradient and see if it can lead to an optimum (t1 function)
"""

import torch
import numpy as np
from scipy.optimize import minimize
from src.gdesc import gen_hypercubes, gen_reducehypercubes, get_countfunc

#---- paramse to input
params_to_planes  = lambda h, k, d : h.reshape((k,d+1))[:,:-1]
params_to_end_terms  = lambda h, k, d,l  :torch.column_stack( [h.reshape((k,d+1))[:,-1]]*l)
incident_mat = lambda h_p, h_et, hpts1, hpts2 : torch.mul(torch.matmul(h_p, hpts1)-h_et, torch.matmul(h_p, hpts2)-h_et)
k=3
d=3
hpoints1, hpoints2 = gen_hypercubes(d, no_loading=True)
l=hpoints1.shape[1]
hpts1 = torch.from_numpy(hpoints1)
hpts2 = torch.from_numpy(hpoints2)
#test_func_00 = lambda h  : torch.sum(torch.matmul(params_to_planes(h,k,d), hpts1)-params_to_end_terms(h,k,d,l))
l_ = 1
incmatfunc = lambda h: incident_mat(params_to_planes(h, k, d), params_to_end_terms(h, k, d, l), hpts1, hpts2)
G00 = lambda col : torch.log(1+torch.exp(col))
G = lambda col : torch.log(.5+torch.sigmoid(col))

cost_function = lambda h: torch.sum(torch.vmap(G)(incmatfunc(h)))  #+ l_*torch.sum(h**2)
count_function = lambda h : torch.sum(torch.vmap(torch.max)(incmatfunc(h)>0.0))


# Create input tensor with requires_grad=True to track gradients
h_t = torch.randn(k * (d+1), requires_grad=True)
# Call the function to get output tensor, then call backward
# output = test_func_00(h)#incmatfunc(h)
# print(output)
# print(hpts1)
# print(output.backward())
# print(h.grad)
h = np.random.random(k*(d+1))

cf = get_countfunc(k, d, hpoints1, hpoints2)

# 2. Create the interface bridge for SciPy
def scipy_objective_wrapper(x_numpy):
    # Convert incoming NumPy array from SciPy into a trainable PyTorch tensor
    x_torch = torch.tensor(x_numpy, dtype=torch.float64, requires_grad=True)

    # Forward pass: Compute the loss
    loss = cost_function(x_torch)

    # Backward pass: Compute exact gradients using PyTorch autograd
    loss.backward()

    # Extract values safely back into 1D NumPy arrays for SciPy
    loss_val = loss.item()
    gradient = x_torch.grad.numpy()

    # Return both the scalar loss and the gradient array
    return loss_val, gradient

result = minimize(
    fun=scipy_objective_wrapper,
    x0=h,
    method='BFGS',  # Quasi-Newton gradient descent method
    jac=True,  # Tells SciPy the wrapper returns both (loss, jacobian)
    options={'disp': True}
)

print(cf(result.x))