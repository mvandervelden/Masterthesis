import numpy as np
from detection_utils import *
# Python adaptation of Vedaldi & Fulkersons implementation of Quickshift in C, included in VLFEAT.
# Changed so it can cluster generically, instead of pixel values and coordinates only

def overlap_to_E(overlap, indexes):
    N = indexes.max()
    E = np.zeros(N+1)
    for i in xrange(N+1):
        mask = np.any(indexes==i,1)
        Eij = overlap[mask]
        print Eij
        E[i] = np.sum(Eij)
        print E[i]
    return E

def quickshift(data, tau):
    N = data.shape[0]
    dim = data.shape[1]
    # TODO tau [max(height, width)/2 in vlfeat where height & width is of image] & sigma [max(2, tau/3) in vlfeat], what is it here???
    tau2 = tau*tau
    sigma = max(2,tau/3)
    
    # For now, I take no window, and use pairwise overlap for all
    # E = np.zeros(N)
    overlap,indexes = get_pairwise_overlap(data)
    print data
    print overlap
    print indexes
    E = overlap_to_E(overlap, indexes)
    parents = np.arange(N)
    dists = np.empty(N)
    dists.fill(np.inf)
    
    # TODO Make some kind of window (based on tau and sigma), iterate over each data point, and within the window around that point iterate over the neighbors
    # for i in xrange(N):
    #     for j in xrange(N):
    #         # VLFEAT: D_ij = d(x_i,x_j)
    #         # VLFEAT: E_ij = exp(- .5 * D_ij / sigma^2) ;
    #         # Overlap is a kind of similarity measure, so we take overlap=E
    #         # 0 = no overlap
    #         # 1 = identical...
    #         Eij = get_overlap(data[i,:], data[j,:])
    #         E[i] += Eij
    
    # Quickshift assigns each i to the closest j which has an increase in the
    # density (E). If there is no j s.t. Ej > Ei, then dists_i == inf (a root
    # node in one of the trees of merges).
    print E
    
    for n in xrange(indexes.shape[0]):
        i = indexes[n,0]
        j = indexes[n,1]
        
        if E[j] > E[i]:
            Eij = overlap[n]
            Dij = 1./Eij if not Eij == 0.0 else np.inf
            if Dij <= tau2 and Dij < dists[i]:
                dists[i] = Dij
                parents[i] = j
        
        # parents is the index of the best pair
        # dists_i is the minimal distance, inf implies no Ej > Ei within
        # distance tau from the point
        
    return parents, dists

def cluster_quickshift(hypotheses, tau):
    parents, dists = quickshift(hypotheses, tau)
    return parents, dists
    # Generate clusters

if __name__ == "__main__":
    A = np.array([[1,1,10,10],[10,10,12,12],[1,1,5,5],[8,8,12,12]])
    tau = 1
    
    print cluster_quickshift(A,tau)