"""

TODO:

* move harnesses to scratch doc
* input starting config
* change how regulation works (might need output function for config)
* save items afterwards and put things in scratch
*

"""


import torch
import numpy as np
from scipy.optimize import minimize

from src.gdesc import gen_hypercubes, gen_reducehypercubes

def get_intersections(k, d,h_m):

    hpoints1, hpoints2=gen_hypercubes(d, no_loading=True)

    l = hpoints1.shape[1]
    params_to_planes = lambda h, k, d: h.reshape((k, d + 1))[:, :-1]
    params_to_end_terms = lambda h, k, d, l: torch.column_stack([h.reshape((k, d + 1))[:, -1]] * l)
    incident_mat = lambda h_p, h_et, hpts1, hpts2: torch.mul(torch.matmul(h_p, hpts1) - h_et,
                                                             torch.matmul(h_p, hpts2) - h_et)


    hpts1 = torch.from_numpy(hpoints1).double()
    hpts2 = torch.from_numpy(hpoints2).double()

    incmatfunc = lambda h: incident_mat(params_to_planes(h, k, d), params_to_end_terms(h, k, d, l), hpts1, hpts2)

    return tuple(torch.vmap(torch.sum)(incmatfunc(h_m) < 0).tolist())

def get_funcs(G,k,d,b=None, e_list=None, regularization=None):
    #---- paramse to input
    params_to_planes  = lambda h, k, d : h.reshape((k,d+1))[:,:-1]
    params_to_end_terms  = lambda h, k, d,l  :torch.column_stack( [h.reshape((k,d+1))[:,-1]]*l)
    incident_mat = lambda h_p, h_et, hpts1, hpts2 : torch.mul(torch.matmul(h_p, hpts1)-h_et, torch.matmul(h_p, hpts2)-h_et)
    """
    TODO:
    
    fix function signature so this makes more sense
    """
    if b is None and e_list is None:
        hpoints1, hpoints2 = gen_hypercubes(d, no_loading=True)
        pcounts = None
    elif e_list is not None:
        hpoints1, hpoints2 = zip(*e_list)
        hpoints1 = np.array(hpoints1).T
        hpoints2 = np.array(hpoints2).T
        pcounts = None
    elif b is not None:
        hpoints1, hpoints2, pcounts = gen_reducehypercubes(d, b,  no_loading=True)
        d=len(b)

    l=hpoints1.shape[1]
    hpts1 = torch.from_numpy(hpoints1).double()
    hpts2 = torch.from_numpy(hpoints2).double()
    #test_func_00 = lambda h  : torch.sum(torch.matmul(params_to_planes(h,k,d), hpts1)-params_to_end_terms(h,k,d,l))


    incmatfunc = lambda h: incident_mat(params_to_planes(h, k, d), params_to_end_terms(h, k, d, l), hpts1, hpts2)
    # G = lambda col : torch.log(1+torch.exp(col))
    # G00 = lambda col : torch.log(.5+torch.sigmoid(col))
    output_function = lambda h : h
    if pcounts is None:
        cost_function_ = lambda h: -1  *torch.sum(torch.vmap(G)(incmatfunc(h).T)) #  + (-1)*torch.sum(h**2)
        count_function = lambda h : torch.sum(torch.vmap(torch.max)(incmatfunc(h).T<0.0))
    else:
        pcounts = torch.tensor(pcounts)

        cost_function_ = lambda h: -1*torch.sum(torch.matmul(torch.vmap(G)(incmatfunc(h).T).T, pcounts))  #+ l_*torch.sum(h**2)
        count_function = lambda h : torch.sum(torch.matmul(torch.vmap(torch.max)(incmatfunc(h).T<0.0).double().T, pcounts))

    if regularization == "ridge":
        cost_function = lambda h: cost_function_(h) + torch.sum(h**2)
    elif regularization== "bound_sigmoid":
        cost_function = lambda h  : cost_function_(torch.sigmoid(h)-.5)
        output_function = lambda h : torch.sigmoid(h)-.5
    elif regularization=="bound_softmax":
        cost_function = lambda h: cost_function_(torch.softmax(h,dim=0) - .5)
        output_function = lambda h: torch.softmax(h,dim=0) - .5
    else:
        cost_function = lambda h: cost_function_(h)


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
    return cost_function, count_function, scipy_objective_wrapper, output_function

reg_list = [None, "ridge", "bound_sigmoid", "bound_softmax"]
