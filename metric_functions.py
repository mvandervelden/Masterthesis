from utils import *
from numpy import *
import logging

log = logging.getLogger(__name__)

def dist_fg(dist_array):
    """ From a distance array, returns the distance to the fg class (=0th col)
    
    doctest:
    >>> dist_fg(np.array([[0,0],[0,1],[1,0]]))
    array([0, 0, 1])
    >>> dist_fg(np.array([[0,0],[0,0.5],[1000.000,9999999.034534]]))
    array([    0.,     0.,  1000.])
    """
    return dist_array[:,0]

def dist_bg(dist_array):
    """ From a distance array, returns the distance to the bg class (=1st col)
    doctest:
    >>> dist_bg(np.array([[0,0],[0,1],[1,0]]))
    array([0, 1, 0])
    >>> dist_bg(np.array([[0,0],[0,0.5],[1000.000,9999999.034534]]))
    array([  0.00000000e+00,   5.00000000e-01,   9.99999903e+06])
    """
    return dist_array[:,1]

def dist_qh(dist_array):
    """ From a distance array, returns the Qh distance: Relative distance
    foreground to background: (bg-fg)/fg (cf Becker)
    
    doctest:
    >>> dist_qh(np.array([[10.,15],[3000,1284],[100,100]]))
    array([ 0.5  , -0.572,  0.   ])
    >>> dist_qh(np.array([[0., 1], [100,0],[0,0]]))
    array([ inf,  -1.,  nan])
    """
    return (dist_array[:,1] - dist_array[:,0]) / dist_array[:,0]

def bb_energy(bb, i, pt_array, dist_array):
    """ Calculate the energy of a Bounding Box: E = sum of distances of
    descriptors inside BB to the fg + sum of distances of descriptors outside BB
    to the bg
    
    doctest:
    >>> bb_energy([10,15,20,25],1,np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    77
    """

    pts_in_bb = points_in_bb(bb, pt_array)
    return dist_array[pts_in_bb,0].sum() + dist_array[~pts_in_bb, 1].sum()

def bb_wenergy(bb, i, pt_array, dist_array):
    """ Calculate the weighted energy of a Bounding Box: E = sum of distances of
    descriptors inside BB to the fg + sum of distances of descriptors outside BB
    to the bg
    
    doctest:
    >>> bb_energy([10,15,20,25],1,np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    
    """
    n=pt_array.shape[0]
    pts_in_bb = points_in_bb(bb, pt_array)
    m = pts_in_bb.shape[0]
    print n, m
    a=dist_array[pts_in_bb,0].sum()/m
    b=dist_array[~pts_in_bb, 1].sum()/(n-m)
    return  a+b

def bb_fg(bb, i, pt_array, dist_array):
    """ Calculate the mean fg-distances of a Bounding Box: D = mean of fg-distances of
    descriptors inside BB
    
    doctest:
    >>> bb_fg([10,15,20,25],1,np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    14.0
    """
    pts_in_bb = points_in_bb(bb, pt_array)
    return dist_array[pts_in_bb,0].mean()

def bb_bg(bb, i, pt_array, dist_array):
    """ Calculate the mean bg-distances of a Bounding Box: D = mean of bg-distances of
    descriptors inside BB
    
    doctest:
    >>> bb_bg([10,15,20,25],1,np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    None
    """
    pts_in_bb = points_in_bb(bb, pt_array)
    return dist_array[pts_in_bb,1].mean()

def bb_qh(bb, i, pt_array, dist_array):
    """ Calculate the mean Qh-distances of a Bounding Box: Sum of Qh dists of
    all descriptors inside BB
    
    doctest:
    >>> bb_qh([10,15,20,25],1,np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10.,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    5.7777777777777786
    """
    pts_in_bb = points_in_bb(bb, pt_array)
    fg_d = dist_array[pts_in_bb,0]
    bg_d = dist_array[pts_in_bb,1]
    # log.debug(' -- bb_qh: input : bb=%s (dtype:%s), pt_array=%s (dtype:%s), dist_array%s (dtype:%s)',bb,bb.dtype, pt_array.shape, pt_array.dtype, dist_array.shape, dist_array.dtype)
    # log.debug(' -- bb_qh: pts_in_bb=%s, fg_d=%s, bg_d=%s',pts_in_bb.shape, fg_d.shape, bg_d.shape)
    if np.sum(pts_in_bb) == 0:
        log.warning(' -- bb_qh: no pts in bb found, so no qh can be determined')
        
    if np.sum(fg_d==0) > 0:
        log.warning(' -- bb_qh: fg_dist = 0 encountered %d times, replacing with %f', np.sum(fg_d==0), fg_d[fg_d>0].min()/1.e99)
        fg_d[fg_d==0] = fg_d[fg_d>0].min()/1.e99
    log.debug(' -- bb_qh: out:%f',( (bg_d - fg_d)/fg_d ).mean())
    
    return ( (bg_d - fg_d)/fg_d ).mean()

def bb_descrqh(bb, i, pt_array, dist_array):
    """ original Qh: hypothesis quality = relative distance of its descriptor
    (does not take into account descriptors other than the one that defines the
    BB)
    
    doctest:
    >>> bb_descrqh([10,15,20,25],2,np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10.,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    -0.66666666666666663
    """
    fg_d = dist_array[i,0]
    return (dist_array[i,1] - fg_d)/fg_d

def bb_uniform(bb, i, pt_array, dist_array):
    """ Define the value of a BB uniformly: each BB has value 1.
    For getting values of hypotheses in the Becker way

    doctest:
    >>> bb_uniform([10,15,20,25],1,np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10.,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    1
    """
    return 1

def det_energy(det, i, pt_array, dist_array):
    """ Define the value of a detection by its BB energy
    
    doctest:
    >>> det_energy([10,15,20,25],1,np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10.,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    77.0
    """
    return bb_energy(det, i, pt_array, dist_array)

def det_qh(det, i, pt_array, dist_array):
    """ Define the value of a detection by its Qh value
    Idea: never supply qh and qd, but give (as i) a list of indexes that
        make up the detection: qh can be computed from that (dist_qh for the sublist)
        and Qd is the lengths of the i-list
    
    doctest:
    >>> det_qh([10,15,20,25],[2,3],np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10.,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    4.166666666666667
    """
    return dist_qh(dist_array[i,:]).mean()

    
def det_qd(det, i, pt_array, dist_array):
    """ Idea: never supply qh/qd, but give (as i) a list of indexes that
    make up the detection: qh can be computed from that (dist_qh for the sublist)
    and Qd is the lengths of the i-list
    
    doctest:
    >>> det_qd([10,15,20,25],[2,3],np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10.,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    2
    """
    if isinstance(i,np.ndarray):
        return i.shape[0]
    elif isinstance(i,list):
        return len(i)
    
def det_becker(det, i, pt_array, dist_array):
    """ Idea: never supply qh/qd, but give (as i) a list of indexes that
    make up the detection: qh can be computed from that (dist_qh for the sublist)
    and Qd is the lengths of the i-list
    
    doctest:
    >>> det_becker([10,15,20,25],[2,3],np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10.,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    (2, 4.166666666666667)
    """
    return (det_qd(det, i, pt_array, dist_array), det_qh(det, i, pt_array, dist_array))

def det_fg(det, i, pt_array, dist_array):
    """
    
    doctest:
    >>> det_fg([10,15,20,25],[2,3],np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10.,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    14.0
    """
    return bb_fg(det, i, pt_array, dist_array)

def det_bg(det, i, pt_array, dist_array):
    """
    
    doctest:
    >>> det_bg([10,15,20,25],[2,3],np.array([[0,0],[10,10],[15,15],[20,20],[25,25],[15,20]]), np.array([[10.,10],[10,20],[30,10],[5,50],[50,5],[7,70]]))
    None
    """
    return bb_bg(det, i, pt_array, dist_array)

if __name__ == "__main__":
    import doctest
    import numpy as np
    doctest.testmod()