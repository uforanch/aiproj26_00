
from src.gdesc import *
import pytest


@pytest.mark.parametrize("d", [3,4,5])
def test_non_reduced_hyperplanes(d):
    hpoints1, hpoints2 = gen_hypercubes(d)
    s=set()
    h_len = hpoints1.shape[1]
    for i1 in range(h_len):
        t1=tuple(hpoints1[:,i1])
        t2=tuple(hpoints2[:,i1])
        s.add((t1,t2))
        s.add((t2, t1))
    assert 2**d*d == len(s)

def test_reduced_hyperplanes():
    hpoints1, hpoints2, _  = gen_reducehypercubes(6,(3,2,1), no_loading=True)
    correct_dict = {(-3,-2,-1):3,
                    (-3,-2,1):2,
                    (-3,0,-1):3,
                    (-1,-2,-1):3,
                    (-3,0,1):2,
                    (-1,-2,1):2,
                    (1,2,1):1,
                    (3,0,1):1,
                    (3,2,-1):1,
                    (3,2,1):0}
    points = set()
    edges = Counter()
    l = hpoints1.shape[1]
    for i1 in range(l):
        p1 = tuple(hpoints1[:,i1])
        p2 = tuple(hpoints2[:,i1])
        points.add(p1)
        points.add(p2)
        edges[(p1,p2)]+=1
    for k,v in edges.items():
        if k in correct_dict:
            assert correct_dict[k] == v

