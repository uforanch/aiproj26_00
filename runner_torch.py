import torch
from scipy.optimize import minimize
import pandas as pd
import numpy as np
from torch._dynamo.variables import optimizer

from src.gdesc01 import get_funcs, reg_list, get_intersections


g_outer = lambda x : torch.sigmoid(x) + .1*torch.exp(-(x-1)**2)
g_inner = lambda x : torch.sum(torch.sigmoid(-1000*x))
g_dict = {"G_test":lambda col : g_outer(g_inner(col) ),
          #"G_original":lambda col: torch.sum(torch.sum(torch.log(1+torch.relu( -1*col)))),
          "G_sigmoid": lambda col: 1/(1+torch.exp(100*torch.min(col))),
          "G_relu_sum": lambda col: torch.sum(torch.relu(-1*col)),
          "G_relu_prod": lambda col: 1-torch.prod(torch.relu(col)),
          "G_relu_prod_1000": lambda col: 1-torch.prod(torch.relu(1000*col)),}
def harness_small(g_dict, optim = "scipy", init_range=None, filename="harness"):
    # g_dict = {"G_00": lambda col : torch.log(1+torch.exp(-1*col)),
    # "G_01": lambda col : torch.log(.5+torch.sigmoid(-1*col))}
    # g_dict = {"G_00": lambda col: torch.log(1 + torch.sum(torch.exp(-1 * col))),
    #           "G_01": lambda col: torch.log( torch.sum(.5+torch.sigmoid(-1 * col)))}

    hs_df = pd.DataFrame(columns=["(k,d)","d","G","reg","count","filename", "intersections", "performance"])
    for k,d in [(3,3), (4,4), (5,5),(5,6)]:
        for g_name, G in g_dict.items():
            for reg in reg_list:
                if init_range is None:
                    h=np.ones(k*(d+1))
                else:
                    h = np.random.random(k*(d+1))*init_range - init_range/2
                cost_function, count_function, scipy_objective_wrapper, output_function = get_funcs(G, k, d, regularization=reg)
                print("------")
                if optim == "scipy":
                    result = minimize(
                        fun=scipy_objective_wrapper,
                        x0=h,
                        method='BFGS',  # Quasi-Newton gradient descent method
                        jac=True,  # Tells SciPy the wrapper returns both (loss, jacobian)
                        options={'disp': True}
                    )

                    rx = output_function(torch.tensor(result.x))
                    print(k, d, g_name, count_function(rx), d * 2 ** (d - 1), "reg", reg)
                    np_filename = f"data/{filename}_{len(hs_df)}"
                    c =  count_function(rx).item()
                    hs_df.loc[len(hs_df)] = {"(k,d)": str((k, d)), "G": g_name, "reg": reg,
                                             "count": c, "filename": filename, "d": d,
                                             "intersections": get_intersections(k, d, rx), "performance": c/(d*(2**(d-1)))}
                    # if count_function(rx)>=100:
                    #    c_extra = count_function(torch.tensor(result.x))
                    np.save(np_filename, rx.reshape((k, d + 1)))
                elif optim == "adam":
                    rx = torch.tensor(h, requires_grad=True)
                    optimizer = torch.optim.Adam([rx],lr=0.1)
                    for _ in range(60):
                        optimizer.zero_grad()
                        y=cost_function(rx)
                        y.backward()
                        optimizer.step()

                    print(k, d, g_name, count_function(rx), d * 2 ** (d - 1), "reg", reg)
                    np_filename = f"data/{filename}_{len(hs_df)}"
                    c = count_function(rx).item()
                    hs_df.loc[len(hs_df)] = {"(k,d)": str((k, d)), "G": g_name, "reg": reg,
                                             "count": count_function(rx).item(), "filename": filename, "d": d,
                                             "intersections": get_intersections(k, d, rx), "performance": c/(d*(2**(d-1)))}
                    np.save(np_filename, torch.detach(rx).numpy().reshape((k, d + 1)))

    hs_df.to_csv(filename+".csv", index=False)
    return hs_df



def t1_matrix_test(g_dict, init_range=None, optim="scipy", filename="h56"):
    k,d=5,6
    t1_matrix = np.matrix([[1,1,1,3,3,-4,0],
                           [-2,-2,-2,3,3,-1,0],
                           [3,3,3,1,1,-4,0],
                           [-1,-1,-1,3,3,6,0],
                           [3,3,3,1,1,8,0]], dtype=np.float64)
    t1 = torch.tensor(t1_matrix).double().reshape((k*(d+1)))
    h56_df = pd.DataFrame(columns=["G", "iterations", "optimized val", "edges cut by config", "optimize func of t1", "regularization", "filename","intersections", "performance"])

    for g_name, G in g_dict.items():
        for reg in reg_list:
            if init_range is None:
                h=np.ones(k*(d+1))
            elif init_range>0:
                h = np.random.random(k*(d+1))*init_range - init_range/2
            else:
                h = np.array(t1_matrix).reshape(-1)
            cost_function, count_function, scipy_objective_wrapper, output_function = get_funcs(G, k, d, regularization=reg)
            cost_function_t1, _, _, _ = get_funcs(G, k, d, regularization=None)
            print("------")
            if optim=="scipy":
                result = minimize(
                    fun=scipy_objective_wrapper,
                    x0=h,
                    method='BFGS',  # Quasi-Newton gradient descent method
                    jac=True,  # Tells SciPy the wrapper returns both (loss, jacobian)
                    options={'disp': True}
                )

                rx = output_function(torch.tensor(result.x))
                print(k, d, g_name, count_function(rx))
                #result.nit - number of iterations
                #
                print("t1", cost_function(t1))

                np_filename = f"data/{filename}_{len(h56_df)}"
                c = count_function(rx).item()
                h56_df.loc[len(h56_df)] = {"G":g_name, "iterations":result.nit, "optimized val":result.fun, "edges cut by config": c, "optimize func of t1":cost_function_t1(t1).item(), "regularization":reg, "filename":filename, "intersections":get_intersections(k,d,rx), "performance": c/(d*(2**(d-1)))}
                np.save(np_filename, rx)
            elif optim=="adam":
                rx = torch.tensor(h, requires_grad=True)
                optimizer = torch.optim.Adam([rx], lr=0.1)
                for _ in range(60):
                    optimizer.zero_grad()
                    y = cost_function(rx)
                    y.backward()
                    optimizer.step()

                print(k, d, g_name, count_function(rx), d * 2 ** (d - 1), "reg", reg)
                np_filename = f"data/{filename}_{len(h56_df)}"
                c = count_function(rx).item()
                h56_df.loc[len(h56_df)] = {"G":g_name, "iterations":60, "optimized val":cost_function(rx).item(), "edges cut by config": c, "optimize func of t1":cost_function_t1(t1).item(), "regularization":reg, "filename":filename, "intersections":get_intersections(k,d,rx), "performance": c/(d*(2**(d-1)))}

                np.save(np_filename, torch.detach(rx).numpy().reshape((k, d + 1)))
    h56_df.to_csv(filename+".csv",index=False)
    return h56_df



def f_test(g_dict, filename):
    f_df = pd.DataFrame(columns=["tuple", "count"] + list(g_dict.keys()))
    vec_list = [[0]*4, [.01]*4, [-.01]*4,[-.01]*4+[1], [.01]*4+[-1 ], [.01]*4+[1 ], [-.5]*4]
    format_dict = {}
    for g_key in g_dict.keys():
        format_dict[g_key] = "{:.4f}"

    for vec in vec_list:
        vec_t = torch.tensor(vec).double()
        data = {"tuple": str(tuple(vec)), "count": torch.sum(vec_t<0).item()}
        for g_name, G in g_dict.items():
            data[g_name] = G(vec_t).item()
        f_df.loc[len(f_df)] = data
    print(f_df)
    f_df.to_csv(filename + ".csv", index=False)


f_test(g_dict, "f")
t1_matrix_test(g_dict, init_range=-1, optim="scipy", filename="t1_scipy")
t1_matrix_test(g_dict, init_range=-1, optim="adam", filename="t1_adam")
harness_small(g_dict, init_range=10, optim="scipy", filename="h_scipy")
harness_small(g_dict, init_range=10, optim="adam", filename="h_adam")
harness_small(g_dict, init_range=None, optim="adam", filename="h_adam_low")