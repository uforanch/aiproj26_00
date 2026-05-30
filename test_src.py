
from src.gdesc import *
import pytest


@pytest.mark.parametrize("d", [3,4,5])
def test_non_reduced_hyperplanes(d):
    hpoints1, hpoints2 = gen_hypercubes(d, no_loading=True)
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


def test_count_p1():
    t1_matrix = np.matrix([[1,1,1,3,3,-4,0],
                           [-2,-2,-2,3,3,-1,0],
                           [3,3,3,1,1,-4,0],
                           [-1,-1,-1,3,3,6,0],
                           [3,3,3,1,1,8,0]])
    hpoints1, hpoints2 = gen_hypercubes(6)
    countfunc = get_countfunc(5, 6, hpoints1, hpoints2)

    assert countfunc(t1_matrix)==192.0

    t1_matrix = np.matrix([[-9,-9,-9,-9,-9,-9,7,-16,5,35,.5],
                           [-9,-9,-9,-9,-9,-9,-32,-4,-17,8,.5],
                           [-9,-9,-9,-9,-9,-9,32,5,19,-4,.5],
                           [-9,-9,-9,-9,-9,-9,-3,15,-3,-38,.5],
                           [-9,-9,-9,-9,-9,-9,15,3,-36,4,.5],
                           [-9,-9,-9,-9,-9,-9,8,-35,-2,-12,.5],
                           [-9,-9,-9,-9,-9,-9,-4,33,7,16,.5],
                           [-9,-9,-9,-9,-9,-9,-18,-4,34,-5,.5]])
    hpoints1, hpoints2 = gen_hypercubes(10)
    countfunc = get_countfunc(8, 10, hpoints1, hpoints2)
    assert countfunc(t1_matrix)==5120.0

def test_count_p2():
    t1_matrix = np.matrix([[1,3,-4,0],
                           [-2,3,-1,0],
                           [3,1,-4,0],
                           [-1,3,6,0],
                           [3,1,8,0]])
    hpoints1, hpoints2, p_counts = gen_reducehypercubes(6,(3,2,1), no_loading=True)
    countfunc = get_countfunc(5, 3, hpoints1, hpoints2, W=p_counts)
    assert countfunc(t1_matrix)==192.0
    t1_matrix = np.matrix([[-9,7,-16,5,35,.5],
                           [-9,-32,-4,-17,8,.5],
                           [-9,32,5,19,-4,.5],
                           [-9,-3,15,-3,-38,.5],
                           [-9,15,3,-36,4,.5],
                           [-9,8,-35,-2,-12,.5],
                           [-9,-4,33,7,16,.5],
                           [-9,-18,-4,34,-5,.5]])
    hpoints1, hpoints2, p_counts = gen_reducehypercubes(10,(6,1,1,1,1), no_loading=True)
    countfunc = get_countfunc(8, 5, hpoints1, hpoints2, W=p_counts)
    assert countfunc(t1_matrix) == 5120.0

def test_count_basic():
    hp1 = np.matrix([2 ,0]).T
    hp2 = np.matrix([-1,0]).T
    h = np.array([1,-1,0])
    assert get_countfunc(1,2, hp1,hp2)(h) == 1.0
    h = np.array([1,-1,3])
    assert get_countfunc(1,2, hp1,hp2)(h) == 0.0
    h = np.array([[1,-1,0],[1,-1,0]])
    assert get_countfunc(2,2, hp1,hp2)(h) == 1.0

    h = np.array([[1,-1,3],[1,-1,3]])
    assert get_countfunc(2,2, hp1,hp2)(h) == 0.0
    hp1, hp2 = gen_hypercubes(2)
    h = np.array([1,0,.5])
    assert get_countfunc(1, 2, hp1, hp2)(h) == 2.0
    h = np.array([[1, 0, .5], [1,-1,.5]])

    assert get_countfunc(2, 2, hp1, hp2)(h) == 3.0
