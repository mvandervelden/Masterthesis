# import hcluster
import numpy as np
import logging

log = logging.getLogger("__name__")

def get_pairwise_overlap(hyp):
    # Can do to with scipy.spatial.distance.pdist, but does not give index array back
    
    n = hyp.shape[0]
    
    log.debug('--Trying to get pairwise overlap of %d combinations',n/2.0 * (n-1))
    # Overlap = intersection/union of 2 hypotheses
    
    # make a vector representing the upper right triangle of a distance matrix
    # excluding the diagonal (so for every row i, calculate overlap with every
    # col j>i)
    overlap = np.zeros(n/2.0 * (n-1))
    indexes = np.zeros([n/2.0 * (n-1),2],np.uint32)
    o_idx = 0
    for i in xrange(n-1):
        if i%10 == 0:
            log.debug('---row i= %d, total j=%d cols', i, n-(i+1))
        for j in xrange(i+1,n):
            # The intersection box exists (>0) when the rightmost xmin is
            # smaller than the leftmost xmax and the lowermost ymin is smaller
            # than the uppermost ymax, in that case intersection = right-left *
            # bottom-top
            indexes[o_idx,:] = [i,j]
            # N.B. 1-indexing, because hyp[x,0] = Qh
            overlap[o_idx] = get_overlap(hyp[i,1:], hyp[j,1:])
            o_idx += 1
    return overlap, indexes

def get_overlap(bbox_a, bbox_b):
    # The intersection box exists (>0) when the rightmost xmin is smaller than
    # the leftmost xmax and the lowermost ymin is smaller than the uppermost
    # ymax, in that case intersection = right-left * bottom-top
    # N.B. 1-indexing, because hyp[x,0] = Qh
    int_top   = np.max([bbox_a[0], bbox_b[0]])
    int_left  = np.max([bbox_a[1], bbox_b[1]])
    int_bottom= np.min([bbox_a[2], bbox_b[2]])
    int_right = np.min([bbox_a[3], bbox_b[3]])
    if int_top < int_bottom and int_left < int_right:
        intersection = np.array((int_right-int_left)*(int_bottom-int_top),dtype=np.float32)
        # Union = A + B - intersection
        area_a = (bbox_a[2]-bbox_a[0])*(bbox_a[3]-bbox_a[1])
        area_b = (bbox_b[2]-bbox_b[0])*(bbox_b[3]-bbox_b[1])
        union = area_a + area_b - intersection
        return intersection / union
    else:
        # If intersection = 0, overlap = 0
        return 0.0

def cluster_hypotheses(overlapvals, index_arr, threshold=0.8):
    """Clusters the hypotheses using single-link agglomerative clustering until 
        threshold is reached (pairwise overlap) -> largest cluster is to be chosen,
        these indexes are returned
    """
    # First: sort the overlap values so the most overlapping are in front
    # (reverse sorting)
    srt_idx = overlapvals.argsort()[::-1]
    
    clustered_idx = dict()
    clusters = dict()
    cur_clust = 1
    
    if overlapvals[srt_idx[0]] < threshold:
        # If no hypotheses have overlap >= threshold, return the first one
        return [index_arr[srt_idx[0],[0]]]
    
    for idx in srt_idx:
        if overlapvals[idx] < threshold:
            # Return the then biggest cluster
            return np.array(sorted([(len(v),v) for v in clusters.values()], reverse=True)[0][1],np.uint32)
        else:
            # Add next two overlapping hypotheses to a cluster
            i = index_arr[idx,0]
            j = index_arr[idx,1]
            # log.debug('i=%d,j=%d',i,j)
            if not i in clustered_idx:
                if not j in clustered_idx:
                    # If both not yet clustered, make a new cluster with both
                    # log.debug('Adding cluster %d, merging i=%d and j=%d'%(cur_clust,i,j))
                    clustered_idx[i] = cur_clust
                    clustered_idx[j] = cur_clust
                    clusters[cur_clust] = [i,j]
                    cur_clust += 1
                else:
                    # If j has a cluster, but i not: Add i to cluster of j
                    # log.debug('Adding i=%d to cluster %d of j=%d'%(i,clustered_idx[j],j))
                    clustered_idx[i] = clustered_idx[j]
                    clusters[clustered_idx[j]].append(i)
            else:
                if not j in clustered_idx:
                    # If i has a cluster, but j not: Add j to cluster of i
                    # log.debug('Adding j=%d to cluster %d of i=%d'%(j,clustered_idx[i],i))
                    clustered_idx[j] = clustered_idx[i]
                    clusters[clustered_idx[i]].append(j)
                elif not clustered_idx[i] == clustered_idx[j]:
                    # If both are clustered, but not yet in to the same
                    # cluster, merge them
                    # print 'Merging cluster %d=%s with %d=%s into %d=%s'%(clustered_idx[i],clusters[clustered_idx[i]],clustered_idx[j],clusters[clustered_idx[j]],cur_clust,clusters[clustered_idx[i]] + clusters[clustered_idx[j]])
                    clusters[cur_clust] = clusters[clustered_idx[i]] + clusters[clustered_idx[j]]
                    clustered_idx[i] = cur_clust
                    clustered_idx[j] = cur_clust
                    cur_clust += 1
                #else:
                # already clustered to the same cluster. Not needed to cluster
    # If the for-loop runs out, all overlaps are above threshold:
    return np.array(sorted([(len(v),v) for v in clusters.values()], reverse=True)[0][1])

"""Below the implementation using hcluster
Own implementation seems more useful"""
# def cluster_hypotheses(hypotheses, threshold=0.8):
#     """Clusters the hypotheses using single-link agglomerative clustering
# until 
#     threshold is reached (pairwise overlap) -> largest cluster is to be
# chosen,
#     a mask of these is returned.
#     """
#     # needed: pairwise overlap (1/pairwise dist measure) (lookup how to do
# overlap) -> in pdist form: (Returns a condensed distance matrix Y. For each i
# and j (i<j), the metric dist(u=X[i], v=X[j]) is computed and stored in the
# ij'th entry)
#     n = hypotheses.shape[0]
#     
#     
#     
#     # Cluster the distance vector using Single Link, get a linkage object
#     Z = hcluster.linkage(distvec, method='single', metric='euclidean')
#     # Get the clusters from the linkage object using threshold distance
#     T = hcluster.fcluster(Z, threshold, 'distance')
#     # TODO what to do if no distance is above the threshold? T should have
# all unique numbers of clusters, counting to n. This means we can stop
# iterating (except that overlapping hypotheses should be removed once each
# other is taken as a detection, until none are left). The order of these
# should probably be according to Qh
#     if T.max() == T.shape[0]:
#         return False
#     # Get the largest cluster, find the leader linkings:
# 
#     
#     L, M = hcluster.leaders(Z, T)
#     # get all leaders of >1 clusters (meaning an index > the no of datapoints)
#     # Set the rest to 0
# 
#     L = L-n
#     L[L<0] = 0
#     # Get the cluster sizes of each leading node L from linkage object ZZ
#     # ( Z[L,3] ) and take the argmax, which corresponds to the index of the
#     # cluster number in M. The indexes of the cluster number corresponding to
#     # the data points are to be found in T, so a mask vector can be made
#     mask = T == M[ Z[L, 3].argmax() ]
#     
#     return mask
    
def merge_cluster(cluster_of_hyp, im_id):
    """Make a detection, a tuple of (Qd, Qh, im_id, xmin,ymin,xmax,ymax)
    """
    Qd = cluster_of_hyp.shape[0]
    rest = cluster_of_hyp.mean(0)
    return (Qd,rest[0], im_id, rest[1], rest[2], rest[3], rest[4])
    
def remove_cluster(cluster, det_bbox, hypotheses, overlap, indexes, threshold=0.0):
    """ overlap thereshold theta p, set by Becker to 0, which means everything 
    with just a little overlap is removed
    """
    n = hypotheses.shape[0]
    to_be_removed = []
    for i in xrange(n):
        if i in cluster:
            # All hypotheses in the detection cluster have to be removed
            to_be_removed.append(i)
        elif get_overlap(hypotheses[i,1:], det_bbox) > threshold:
            # If overlap too big, remove too
            to_be_removed.append(i)
    if len(to_be_removed) == n:
        # Nothing to do anymore: everything is clustered.
        return None, None, None
    else:
        # Remove overlap values & indexes for hypotheses that will be removed
        overlap_to_be_removed = []
        for i in xrange(indexes.shape[0]):
            if indexes[i,0] in to_be_removed or indexes[i,1] in to_be_removed:
                overlap_to_be_removed.append(i)

        return hypotheses[~to_be_removed], overlap[~overlap_to_be_removed], indexes[~overlap_to_be_removed]
