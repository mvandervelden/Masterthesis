import numpy as np
import logging
from detection_utils import *
from file_io import *

# Python adaptation of Vedaldi & Fulkersons implementation of Quickshift in C, included in VLFEAT.
# Changed so it can cluster generically, instead of pixel values and coordinates only
log = logging.getLogger(__name__)

def overlap_to_E(indexes):
    
    E = [sum(i.values()) for i in indexes]
    return E

def quickshift(data, tau=np.inf):
    log.debug('-- Running Quickshift algorithm with tau = %f',tau)
    N = data.shape[0]
    dim = data.shape[1]
    tau2 = tau*tau
    log.debug('data: shape %s, first: %s, last: %s',data.shape, data[0], data[-1])    
    # For now, I take no window, and use pairwise overlap for all
    indexes = [dict() for i in xrange(N)]
    
    for i in xrange(N):
        for j in xrange(N):
            di = data[i]
            dj = data[j]
            overlap = get_overlap(di, dj)
            if overlap > 0:
                indexes[i][j] = overlap
    try:
        log.debug('indexes: shape %s, first: %s, last: %s',len(indexes), indexes[0].items()[0], indexes[-1].items()[-1])
    except Exception as e:
        log.error("Something wrong with indexes: %s", indexes)
        log.error(e)
        raise e
    
    E = overlap_to_E(indexes)
    log.debug('E: shape %s, first: %s, last: %s', len(E), E[0], E[-1])
    parents = np.arange(N)
    dists = np.empty(N)
    dists.fill(np.inf)
    
    # Quickshift assigns each i to the closest j which has an increase in the
    # density (E). If there is no j s.t. Ej > Ei, then dists_i == inf (a root
    # node in one of the trees of merges).
    for i in xrange(N):
        overlap_i = indexes[i]
        for j, overlap in overlap_i.items():
            if j%1000 == 0:
                log.debug('j: %d, overlap: %.3f Ei: %.3f Ej: %.3f', j,overlap, E[i], E[j])
            # Possible: hypo[i] == hypo[j], so E[i] == E[j], make sure they are clustered, but not assigned to eachother
            if E[j] > E[i] or (E[i] == E[j] and i<j):
                Eij = overlap
                Dij = 1./Eij if not Eij == 0.0 else np.inf
                # log.debug('Eij: %.3f Dij: %.3f',Eij,Dij)
                if Dij <= tau2 and Dij < dists[i]:
                    dists[i] = Dij
                    parents[i] = j
        # parents is the index of the best pair
        # dists_i is the minimal distance, inf implies no Ej > Ei within
        # distance tau from the point
        
    return parents, dists

def cluster_quickshift(hypotheses, tau, save_tree_path=None):
    N = hypotheses.shape[0]
    try:
        parents, dists = quickshift(hypotheses, tau)
    except FloatingPointError as e:
        print "Floating Point error encountered in quickshift: %s"%e
        log.error("Floating Point error encountered in quickshift: %s", e)
        
    if not save_tree_path is None:
        save_quickshift_tree(save_tree_path, parents, dists)
    # parents = array of length N (no of hypotheses) where the values represent the parent hypothesis of the i'th hypothesis
    # dists = array of length N where the valeus represent the distance to the parent node
    # if parents[i] = i, this is a root node --> a detection
    log.debug('parents: dtype: %s', parents.dtype)
    log.debug('dists, min, max, mean, size: %s, %s, %s, %s', dists.min(), dists.max(), dists.mean(), dists.shape)
    boolroots = np.array([i==p for i,p in enumerate(parents)])
    log.debug('boolroots sum, size: %s, %s', boolroots.sum(), boolroots.shape)
    roots = parents[boolroots]
    log.debug('roots: %s', roots)
    root_node_indexes = np.unique(roots)
    D = root_node_indexes.shape[0]
    detections = hypotheses[root_node_indexes]
    log.debug('No of detections found: %s', detections.shape)
    log.debug('Root node indexes: %s', root_node_indexes)
    # make dist_reference lists:
    while len(np.unique(parents)) > D:
        for i, p in enumerate(parents):
            if not i == p:
                parents[i] = parents[p]
    
    dist_references = [[] for i in range(D)]
    for p_i, p in enumerate(parents):
        for r_i,r in enumerate(root_node_indexes):
            if p == r:
                dist_references[r_i].append(p_i)
    # Convert list of lists to list of arrays
    dist_references = [np.asarray(l) for l in dist_references]
    return detections, dist_references

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger('')
    A = np.array([[1,1,10,10],[10,10,12,12],[1,1,5,5],[8,8,12,12],[1,1,10,10],[1,1,10,10]])
    tau = np.inf
    
    print cluster_quickshift(A,tau, 'tmptree')