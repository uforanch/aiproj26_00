#set dimensions d, planes k
# kd variables generated
#do not need to generate cube
import pprint
from collections import Counter

import numpy as np
import scipy

import pandas as pd
import time
import os
import datetime


def gen_hypercubes(d):
    try:
        data = np.load(f"matrices{d}.npz")
        hpoints1 = data['hpoints1']
        hpoints2 = data['hpoints2']
        return hpoints1, hpoints2
    except:
        #non reduced
        hpoints1 = np.ones((d,2**(d-1)*d))
        hpoints2 = np.ones((d,2**(d-1)*d))

        #testing - gen all pairs, gen half
        j=0
        for i1 in range(2**d):
            for i2 in range(d):

                if 1<<i2>i1:
                    break
                if (i1&(1<<i2))!=0:
                    v1=i1
                    v2=i1^(1<<i2)
                    for j2 in range(d):
                        hpoints1[j2, j] = (-2* ((v1&(1<<j2)) !=0 ))+1
                        hpoints2[j2, j] = (-2* ((v2&(1<<j2)) !=0 ))+1
                    j += 1
        np.savez(f"matrices{d}.npz", hpoints1, hpoints2)
        return hpoints1, hpoints2

def gen_reducehypercubes(d, b, no_loading=False):
    b = tuple(b)
    l_b = len(b)
    if sum(b)!=d:
        raise("Invalid reduction (sum of b incorrect)")
    try:
        if no_loading:
            raise("No loading files")
        data = np.load(f"matrices_red{d}_{b}.npz")
        hpoints1 = data['hpoints1']
        hpoints2 = data['hpoints2']
        p_counts = data['p_counts']
        return hpoints1, hpoints2, p_counts
    except Exception as E:
        # non reduced
        edge_count = Counter()
        #should look at the c program to see how reduction accomplished
        def reducepoint(p):
            b_ind = 0
            cur_b = b[0]
            out = [0] * l_b
            for i, c in enumerate(p):
                if i>cur_b-1:
                    b_ind += 1
                    cur_b += b[b_ind]
                out[b_ind] += c
            return tuple(out)
        for i1 in range(2 ** d):
            for i2 in range(d):
                v1 = i1
                v2 = i1 ^ (1 << i2)
                v1, v2 = min(v1, v2), max(v1, v2)
                p1 = reducepoint([(-2 * ((v1 & (1 << j2)) != 0)) + 1 for j2 in range(d)])
                p2 = reducepoint([(-2 * ((v2 & (1 << j2)) != 0)) + 1 for j2 in range(d)])
                edge_count[(p1,p2)]+=1
        l = len(edge_count)
        p_counts = [0] * l
        hpoints1 = np.ones((l_b,l))
        hpoints2 = np.ones((l_b,l))
        for i, P in enumerate(edge_count):
            p1,p2 = P
            p_counts[i] = edge_count[(p1,p2)]/2
            for i2 in range(l_b):
                hpoints1[i2, i] = p1[i2]
                hpoints2[i2, i] = p2[i2]
        p_counts = np.diag(p_counts)
        np.savez(f"matrices_red{d}_{b}.npz", hpoints1, hpoints2, p_counts)
        return hpoints1, hpoints2, p_counts

#hpoints test in small cases - turn the above into a set of pairs
#see if it has all the points



def get_optfunc(k,d,hpoints1, hpoints2,F,G,W=None):
    #F is normalizing (softmax?)
    #G is max or something to make column of 0s to 0
    if W is None:
        W = np.identity(hpoints1.shape[1])
    return lambda h :np.sum(np.apply_along_axis(G,1, F((h.reshape((k,d))@hpoints1-1)*(h.reshape((k,d))@hpoints2-1))@W))

def get_countfunc(k,d,hpoints1, hpoints2, W=None):
    if W is None:
        W = np.identity(hpoints1.shape[1])
    return lambda h :np.sum(np.max(np.where((h.reshape((k,d))@hpoints1-1)*(h.reshape((k,d))@hpoints2-1)@W > 0,1.0,0.0),axis=0))


F_dict = {
    "sum":np.sum,
    "softmax": lambda x:  np.log(1+np.exp(x)),
    "sigmoid": scipy.special.expit,
    "relu": lambda x: np.maximum(0,x),
}

G_dict = {
    "sum":np.sum,
    "softmax": lambda x:  np.log(1+np.exp(np.sum(x))),
    "sigmoid": lambda x: scipy.special.expit(np.sum(x)),
    "relu": lambda x: np.sum(np.maximum(0,x)),
}

def case_solve(k,d,h_init, opt_method, opt_params, reduction, F_name:str, G_name:str, save=False):
    if reduction is not None:
        hpoints1, hpoints2, p_count = gen_reducehypercubes(d,reduction)
    else:
        hpoints1, hpoints2 = gen_hypercubes(d)
        p_count = None
    if h_init is None:
        h_init = "ones"
    if h_init=="ones":
        h=np.ones((k,d))
    elif h_init=="random":
        h=np.random.random((k,d))

    optfunc = get_optfunc(k, d, hpoints1, hpoints2, lambda x: x, np.sum, W=p_count)
    t = time()
    res = scipy.optimize.minimize(optfunc, h.flatten(), method='nelder-mead')
    out_h = res.x.reshape((k,d))
    t = time()-t
    countfunc = get_countfunc(k,d,hpoints1,hpoints2, W=p_count)
    count = countfunc(out_h)

    data = {"date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") ,
            "k":k,
            "d":d,
            "h_init":h_init,
            "opt_method":opt_method,
            "opt_params":str(opt_params),
            "reduction":str(reduction),
            "F":F_name, "G":G_name,
            "time":t,
            "result_count": count}
    print(data)
    if save:
        if not os.path.isfile("results.csv"):
            df = pd.DataFrame([data])
            df.to_csv("results.csv")
        else:
            df=pd.read_csv("results.csv")
            df.loc[len(df)] = data
            df.to_csv("results.csv")
    return count

def case_count(k,d, h, reduction,):
    if reduction is not None:
        hpoints1, hpoints2, p_count = gen_reducehypercubes(d, reduction)
    else:
        hpoints1, hpoints2 = gen_hypercubes(d)
        p_count = None
    countfunc = get_countfunc(k,d,hpoints1,hpoints2, W=p_count)

    return countfunc(h)

