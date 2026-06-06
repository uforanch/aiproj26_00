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
from torch.nn.functional import cosine_similarity
import numpy as np
from scipy.optimize import minimize

from src.gdesc import gen_hypercubes, gen_reducehypercubes, get_countfunc
def get_funcs(k,d,b=None):
    #---- paramse to input
    params_to_planes  = lambda h, k, d : h.reshape((k,d+1))[:,:-1]
    params_to_end_terms  = lambda h, k, d,l  :torch.column_stack( [h.reshape((k,d+1))[:,-1]]*l)
    incident_mat = lambda h_p, h_et, hpts1, hpts2 : torch.mul(torch.matmul(h_p, hpts1)-h_et, torch.matmul(h_p, hpts2)-h_et)
    if b is None:
        hpoints1, hpoints2 = gen_hypercubes(d, no_loading=True)
        pcounts = None
    else:
        hpoints1, hpoints2, pcounts = gen_reducehypercubes(d, b,  no_loading=True)

    l=hpoints1.shape[1]
    hpts1 = torch.from_numpy(hpoints1)
    hpts2 = torch.from_numpy(hpoints2)
    #test_func_00 = lambda h  : torch.sum(torch.matmul(params_to_planes(h,k,d), hpts1)-params_to_end_terms(h,k,d,l))

    incmatfunc = lambda h: incident_mat(params_to_planes(h, k, d), params_to_end_terms(h, k, d, l), hpts1, hpts2)
    G = lambda col : torch.log(1+torch.exp(col)) * -1
    G00 = lambda col : torch.log(.5+torch.sigmoid(col)) * -1
    if pcounts is None:
        cost_function = lambda h: torch.sum(torch.vmap(G)(incmatfunc(h)))  + (-1)*torch.sum(h**2)
        count_function = lambda h : torch.sum(torch.vmap(torch.max)(incmatfunc(h)>0.0))
    else:
        pcounts = torch.tensor(pcounts)

        cost_function = lambda h: torch.sum(torch.matmul(torch.vmap(G)(incmatfunc(h)), pcounts))  #+ l_*torch.sum(h**2)
        count_function = lambda h : torch.sum(torch.matmul(torch.vmap(torch.max)(incmatfunc(h)>0.0), pcounts))



    #cf = get_countfunc(k, d, hpoints1, hpoints2)

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
    return cost_function, count_function, scipy_objective_wrapper
def exp00():
    k=3
    d=3
    h = np.random.random(k*(d+1))
    cost_function, count_function, scipy_objective_wrapper = get_funcs(k,d)
    result = minimize(
        fun=scipy_objective_wrapper,
        x0=h,
        method='BFGS',  # Quasi-Newton gradient descent method
        jac=True,  # Tells SciPy the wrapper returns both (loss, jacobian)
        options={'disp': True}
    )
    rx = torch.tensor(result.x)
    print(count_function(rx))

def exp01():
    k=5
    d=6
    h = np.random.random(k*(d+1))
    cost_function, count_function, scipy_objective_wrapper = get_funcs(k,d)
    result = minimize(
        fun=scipy_objective_wrapper,
        x0=h,
        method='BFGS',  # Quasi-Newton gradient descent method
        jac=True,  # Tells SciPy the wrapper returns both (loss, jacobian)
        options={'disp': True}
    )
    t1_matrix = np.matrix([[1,1,1,3,3,-4,0],
                           [-2,-2,-2,3,3,-1,0],
                           [3,3,3,1,1,-4,0],
                           [-1,-1,-1,3,3,6,0],
                           [3,3,3,1,1,8,0]])
    t1 = torch.tensor(t1_matrix).double().reshape((k*(d+1)))
    rx = torch.tensor(result.x, requires_grad=True)
    loss = cost_function(rx)

    # Backward pass: Compute exact gradients using PyTorch autograd
    loss.backward()

    # Extract values safely back into 1D NumPy arrays for SciPy
    loss_val = loss.item()
    gradient = rx.grad
    print(count_function(rx))
    print(loss_val, cost_function(t1))
    print(cosine_similarity(t1-rx, gradient, dim=0))

exp01()