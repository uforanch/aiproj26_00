import pprint
from collections import Counter

import numpy as np
import scipy

import pandas as pd
import time
import os
import datetime


def gen_hypercubes(d, no_loading=False):
    try:
        if no_loading:
            raise(Exception("No loading files"))
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
        np.savez(f"matrices{d}.npz", hpoints1=hpoints1, hpoints2=hpoints2)
        return hpoints1, hpoints2

def gen_reducehypercubes(d, b, no_loading=False):
    b = tuple(b)
    l_b = len(b)
    if sum(b)!=d:
        raise(Exception("Invalid reduction (sum of b incorrect)"))
    try:
        if no_loading:
            raise(Exception("No loading files"))
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
            p_counts[i] = edge_count[(p1,p2)]
            for i2 in range(l_b):
                hpoints1[i2, i] = p1[i2]
                hpoints2[i2, i] = p2[i2]
        p_counts = np.diag(p_counts)/2
        np.savez(f"matrices_red{d}_{b}.npz", hpoints1=hpoints1, hpoints2=hpoints2, p_counts=p_counts)
        return hpoints1, hpoints2, p_counts

def get_optfunc(k,d,hpoints1, hpoints2,F,G,W=None):
    if W is None:
        W = np.identity(hpoints1.shape[1])

    def h_ext_to_h(k, d, h):
        return h.reshape(k, d+1)[:,:-1] / np.linalg.norm( h.reshape(k, d+1)[:,:-1], axis=1, keepdims=True)
    def h_ext_to_terms(k, d, h):
        return np.broadcast_to(h.reshape(k, d+1)[:,-1:], (k,hpoints1.shape[1])) / np.linalg.norm( h.reshape(k, d+1)[:,:-1], axis=1, keepdims=True)

    return lambda h :np.sum(np.apply_along_axis(G,1, F(np.multiply(h_ext_to_h(k, d, h)@hpoints1-h_ext_to_terms(k, d, h), h_ext_to_h(k, d, h).reshape((k,d))@hpoints2-h_ext_to_terms(k, d, h)))@W))

def get_optfunc_termconst(k,d,end_terms, hpoints1, hpoints2,F,G,W=None):
    if W is None:
        W = np.identity(hpoints1.shape[1])
    end_terms_matrix = np.broadcast_to(end_terms.reshape(k,1), (k,hpoints1.shape[1]))
    return lambda h :np.sum(np.apply_along_axis(G,1, F(np.multiply(h.reshape((k,d))@hpoints1-end_terms_matrix, h.reshape((k,d))@hpoints2-end_terms_matrix))@W))


def get_countfunc(k,d,hpoints1, hpoints2, W=None):
    if W is None:
        W = np.identity(hpoints1.shape[1])
    def h_ext_to_h(k, d, h):
        return h.reshape(k, d+1)[:,:-1]
    def h_ext_to_terms(k, d, h):
        return np.broadcast_to(h.reshape(k, d+1)[:,-1:], (k,hpoints1.shape[1]))
    return lambda h :np.sum(np.max(np.where(np.multiply( h_ext_to_h(k, d, h)@hpoints1-h_ext_to_terms(k, d, h), h_ext_to_h(k, d, h).reshape((k,d))@hpoints2-h_ext_to_terms(k, d, h) ) < 0,1.0,0.0),axis=0)@W)


F_dict = {
    "id":lambda x: x,
    "softmax": lambda x:  np.log(1+np.exp(x)),
    "sigmoid": scipy.special.expit,
    "relu": lambda x: np.maximum(0,x),
    "sm_00": lambda x: np.exp(x)
}

G_dict = {
    "sum":np.sum,
    "softmax": lambda x:  np.log(1+np.exp(np.sum(x))),
    "sigmoid": lambda x: scipy.special.expit(np.sum(x)),
    "relu": lambda x: np.sum(np.maximum(0,x)),
    "sm_00": lambda x: np.log(np.sum(x))
}

def case_solve(k,d,h_init, opt_method, opt_params, reduction, F_name:str, G_name:str, save=False, end_terms=None, filename=None):
    """

    :param k:  - number of hyperplanes
    :param d:  - number of
    :param h_init: whether to initiation hyperplane with ones or random
    :param opt_method: method for the sci py optimizer
    :param opt_params:  tuning parameters for such
    :param reduction: whether to reduce the hypercubes or not and how
    :param F_name: function applied to map edge-hyperplane pairs to 0 or 1 continuously
    :param G_name: function applied to
    :param end_terms: whether we optimize the constant terms of the hyperplanes or keep them constant
    :param save: do we save the results to the csv file
    :param filename: file to save to, default results.csv
    :return out_h: returns the config
    """
    if filename is None:
        filename = "results.csv"
    h_d = d
    if reduction is not None:
        hpoints1, hpoints2, p_count = gen_reducehypercubes(d,reduction)
        h_d = len(reduction)
    else:
        hpoints1, hpoints2 = gen_hypercubes(d)
        p_count = None

    if h_init is None:
        h_init = "ones"
    if h_init=="ones" and end_terms is not None:
        h=np.ones((k,h_d))
    elif h_init=="random" and end_terms is not None:
        h=np.random.random((k,h_d))
    elif h_init=="ones" and end_terms is None:
        h=np.ones((k,h_d+1))
    elif h_init=="random" and end_terms is None:
        h=np.random.random((k,h_d+1))
    F = F_dict[F_name]
    G = G_dict[G_name]
    if end_terms is None:
        optfunc = get_optfunc(k, h_d, hpoints1, hpoints2, F, G, W=p_count)
    else:
        optfunc = get_optfunc_termconst(k, h_d, end_terms, hpoints1, hpoints2, F, G, W=p_count)
    t = time.time()
    res = scipy.optimize.minimize(optfunc, h.flatten(), method='nelder-mead')
    print(res.fun)
    if end_terms is not None:
        out_h = np.concat((res.x.reshape((k,h_d)), end_terms.reshape((k,1))), axis=1)
    else:
        out_h = res.x.reshape((k, h_d+1))

    t = time.time()-t
    countfunc = get_countfunc(k,h_d,hpoints1,hpoints2, W=p_count)
    count = countfunc(out_h)

    data = {"date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") ,
            "k":k,
            "d":d,
            "h_init":h_init,
            "opt_method":opt_method,
            "opt_params":str(opt_params),
            "reduction":str(reduction),
            "end terms": str(end_terms),
            "F":F_name, "G":G_name,
            "time":t,
            "result_count": count}
    print(data)
    if save:
        if not os.path.isfile(filename):
            df = pd.DataFrame([data])
            df.to_csv(filename, index=False)
        else:
            df=pd.read_csv(filename)
            df.loc[len(df)] = data
            df.to_csv(filename, index=False)
    return out_h

def case_count(k,d, h, reduction,):
    """

    :param k:  number of hyperplanes
    :param d:  number of dimensions
    :param h: k x d+1 hyperplane coefficients
    :param reduction: whether to reduce hypercube
    :return:
    """
    if reduction is not None:
        hpoints1, hpoints2, p_count = gen_reducehypercubes(d, reduction)
    else:
        hpoints1, hpoints2 = gen_hypercubes(d)
        p_count = None
    countfunc = get_countfunc(k,d,hpoints1,hpoints2, W=p_count)

    return countfunc(h)

