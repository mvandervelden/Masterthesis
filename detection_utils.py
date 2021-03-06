# import hcluster
import numpy as np
import logging

# np.seterr(all='raise')
log = logging.getLogger(__name__)

def single_link_clustering(hypotheses, hvalues, overlap, indexes, ranking, DETopts):
    detections = []
    dist_references = []
    while not indexes is None:
        # Cluster hypotheses' overlap to get the biggest cluster (it's hypotheses' indexes)
        left = np.sum(~(hvalues[:] == 0))
        log.debug('  no. of hypothesis left: %d',left)
        log.debug('  no. of overlaps left: %d, should be %d/2 * (%d-1)=%.1f',overlap.shape[0], left, left,left/2.0*(left-1))
        
        if left > 1:
            log.debug('  --  Clustering again')
            best_cluster_idx = cluster_hypotheses(overlap, indexes, DETopts['theta_m'])
        elif left == 1:
            log.debug('  --  No need for clustering, only 1 hypothesis left, val: %f', hvalues.sum())
            best_cluster_idx = np.where(~(hvalues == 0))[0]
            log.debug('   - ID: %s',best_cluster_idx)
            
        # merge the biggest cluster of hypotheses into a detection, and append it
        log.debug(' Best cluster size: %d',best_cluster_idx.shape[0])
        detection = merge_cluster(hypotheses[best_cluster_idx])
        refs = ranking[best_cluster_idx]
        log.debug(' Detection found: %s, refs: %s',detection, refs)
        detections.append(detection)
        dist_references.append(refs)
        # Select hypotheses to remove based on the cluster and the removal threshold theta_p
        hvalues, overlap, indexes = remove_cluster(best_cluster_idx, detection, hypotheses, hvalues, overlap, indexes, DETopts['theta_p'])
    return detections, dist_references

def threshold_distances(exemplars, points, distances, threshold):
    """ threshold distances
    
    doctest:
    >>> threshold_distances(exemplars, points, distances, t)
    
    
    """
    log.debug(" -- Removing exemplars below threshold %s", threshold)
    log.debug(" -- Shapes BEFORE: exemplars: %s, points: %s, dist: %s", exemplars.shape, points.shape, distances.shape)
    log.debug(" -- Distance properties: means: %s, max: %s, min: %s", distances.mean(0), distances.max(0), distances.min(0))
    
    exemplars_th = []
    points_th = []
    distances_th = []
    
    old_dist_size = distances.shape[0]
    N = distances.shape[0]
    if len(distances.shape) == 2:
        k = 1
    else:
        k = distances.shape[2]
        
    for i in xrange(N):
        if len(distances.shape) == 3:

            if threshold == 'ranked':
                # get all distances smaller than the median (median of even amount of values: mean of the middle two)
                dmask = distances[i,0] < np.median(distances[i])
            elif threshold == 'nearest':
                # Get all fg_d < the nearest bg_d
                dmask = distances[i,0] < distances[i,1].min()

            # amount of exemplars to be added
            amount = dmask.sum()
            # log.debug('Amount of exemplars elected: %d', amount)
            if amount == 0:
                # log.debug('Nothing to add')
                continue
            dists = np.zeros((amount,2))
            dists[:,0] = distances[i,0][dmask]
            # For bg_d, choose the closest bg_d:
            dists[:,1] = distances[i,1].min()
            # Add dists to the results
            distances_th.append(dists)
            # Add the exemplars accordingly
            exemplars_th.append(exemplars[i][dmask])
            # Add the point 'amount' times
            points_th.append(np.tile(points[i],(amount,1)))
        else:
            # 1NN case
            if distances[i,0] < distances[i,1]:
                # fg < bg, so add!
                distances_th.append(distances[i])
                points_th.append(points[i])
                exemplars_th.append(exemplars[i])
    if len(exemplars_th) > 0:
        log.info('Adding exemplars for %d descriptors', len(exemplars_th))
        exemplars = np.vstack(exemplars_th)
        points = np.vstack(points_th)
        distances = np.vstack(distances_th)
    else:
        log.info('Adding no exemplars')
        exemplars = np.zeros([0,4])
        points = np.zeros([0,5])
        distances = np.zeros([0,2])
    log.debug(" -- Shapes  AFTER: exemplars: %s, points: %s, dist: %s", exemplars.shape, points.shape, distances.shape)
    log.debug("Removing points where fg_d > bg_d: Keeping %d (=%d? =%d?) points from %d total, because of Becker thresholding",\
        points.shape[0], exemplars.shape[0], distances.shape[0], old_dist_size)
    return exemplars, points, distances

def get_hypotheses(exemplars, points, imwidth, imheight, logger=None):
    """
    Make hypothesis from an exemplar-points pair
    exemplar: [rel_w = obj_w/sigma, rel_h = obj_h/sigma, rel_x_pos = (x-xmin)/obj_w, rel_y_pos = (y-ymin)/obj_h]
    point:    [x,y,sigma]
    hypothesis: [xmin, ymin, xmax, ymax]
    hyp_w = rel_w * sigma
    hyp_h = rel_h * sigma
    hyp_xmin = x-(rel_x_pos * obj_w) = x - (rel_x_pos * rel_w * sigma)
    hyp_ymin = y - (rel_y_pos * rel_h * sigma)
    hyp_xmax = hyp_xmin + hyp_w
    
    NB: doctest not up to date...
    
    doctest:
    >>> get_hypotheses(np.array([[1,1,0.5,0.5],[0.5,1,0.1,0.5]]), np.array([[0.,0., 2.0], [100,100, 1.0]]), 200, 200)
    array([[   0.  ,    0.  ,    1.  ,    1.  ],
           [  99.95,   99.5 ,  100.45,  100.5 ]])
    """
    if not logger is None:
        log = logger
    else:
        log = logging.getLogger(__name__)
    log.info('  -- Getting hypotheses from %s points and %s exemplars (img dimensions: [%d, %d])',\
        points.shape, exemplars.shape, imwidth, imheight)
    log.debug(' --- Check on Points vals: max: %s, min: %s', points.max(0), points.min(0))
    hypotheses = np.zeros([exemplars.shape[0], 4])
    # Make sure hypotheses lie within image bounds!!! (minimum and maximum within image bounds)
    
    hypotheses[:,0] = np.maximum(points[:,0] - (exemplars[:,2] * exemplars[:,0] * points[:,2]), 0)
    hypotheses[:,1] = np.maximum(points[:,1] - (exemplars[:,3] * exemplars[:,1] * points[:,2]), 0)
    hypotheses[:,2] = np.minimum(hypotheses[:,0] + (exemplars[:,0] * points[:,2]), imwidth)
    hypotheses[:,3] = np.minimum(hypotheses[:,1] + (exemplars[:,1] * points[:,2]), imheight)

    log.info('  -- found %s hypotheses', hypotheses.shape)
    if hypotheses.shape[0] > 0:
        if hypotheses[:,0].max() > imwidth:
            log.warning("Hypotheses coords outside imwidth detected:")
            log.warning("hyp: %s", hypotheses[hypotheses[:,0] > imwidth])
            log.warning("pts: %s", points[hypotheses[:,0] > imwidth])
            log.warning("exm: %s", exemplars[hypotheses[:,0] > imwidth])
        if hypotheses[:,1].max() > imheight:
            log.warning("Hypotheses coords outside imheight detected:")
            log.warning("hyp: %s", hypotheses[hypotheses[:,1] > imheight])
            log.warning("pts: %s", points[hypotheses[:,1] > imheight])
            log.warning("exm: %s", exemplars[hypotheses[:,1] > imheight])
        log.info('  - hyp example: %s from point: %s and exemplar: %s', hypotheses[0,:], points[0,:], exemplars[0,:])
        log.info('  - hyp example: %s from point: %s and exemplar: %s', hypotheses[-1,:], points[-1,:], exemplars[-1,:])
    return hypotheses

def get_hypothesis_values(hypotheses, distances, points, metric):
    log.info('  -- get hypothesis values for %s hypotheses (metric:%s, %s distances, %s points)', \
        hypotheses.shape, metric.__name__, distances.shape, points.shape)
    vals = np.zeros(hypotheses.shape[0])
    for h in xrange(hypotheses.shape[0]):
        vals[h] = metric(hypotheses[h,:], h, points, distances)
    return vals

def get_detection_values(detections, reflist, distances, points, metric, logger=None):
    if not logger is None:
        log = logger
    else:
        log = logging.getLogger(__name__)
    
    log.info('  -- get detection values for %s detections (metric:%s, %s references, %s distances, %s points)', \
        detections.shape, metric.__name__, len(reflist),distances.shape, points.shape)
    if not metric.__name__ in  ['det_becker','det_qd_exempfg']:
        vals = np.zeros(detections.shape[0])
    else:
        vals = np.zeros([detections.shape[0],2])
    for h in xrange(detections.shape[0]):
        vals[h] = metric(detections[h,:], reflist[h], points, distances)
    log.debug('  -- found %d values', len(vals))
    return vals

def sort_values(values, logger=None):
    if not logger is None:
        log = logger
    else:
        log = logging.getLogger(__name__)
    
    if len(values.shape) == 1:
        s = values.argsort()
        log.debug(' -- ranking dimensions: %s', s.shape)
        return s
    else:
        # Sort values in increasing order, by first col first, then by sec col
        # NB: Sorts from small to large: Becker: small clusters first, small qh vals first (which are both 'bad' values)
        s = np.lexsort(np.fliplr(values).T)
        log.debug(' -- ranking dimensions: %s', s.shape)
        return s

def make_indexes(size):
    indexes = np.zeros([size/2.0 * (size-1),2], np.uint32)
    # print indexes.shape
    o = 0
    for i in xrange(size-1):
        # The intersection box exists (>0) when the rightmost xmin is
        # smaller than the leftmost xmax and the lowermost ymin is smaller
        # than the uppermost ymax, in that case intersection = right-left *
        # bottom-top

        # print i
        # print o
        # print size-i+1
        # print np.arange(i+1, size)
        # print range(o, o+(size-(i+1)))
        indexes[o:o+(size-(i+1)),0] = i
        indexes[o:o+(size-(i+1)),1] = np.arange(i+1, size)
        # print indexes
        o += (size - (i+1))
    return indexes

def get_pairwise_overlap(hyp):
    # Can do to with scipy.spatial.distance.pdist, but does not give index array back
    
    n = hyp.shape[0]
    
    log.debug('--Trying to get pairwise overlap of %.1f combinations',n/2.0 * (n-1))
    # Overlap = intersection/union of 2 hypotheses
    
    # make a vector representing the upper right triangle of a distance matrix
    # excluding the diagonal (so for every row i, calculate overlap with every
    # col j>i)
    overlap = np.zeros(n/2.0 * (n-1))
    indexes = make_indexes(n)
    o_idx = 0
    for i in xrange(n-1):
        if i%100 == 0:
            log.debug('---row i= %d, total j=%d cols', i, n-(i+1))
        for j in xrange(i+1,n):
            # N.B. 1-indexing, because hyp[x,0] = Qh
            overlap[o_idx] = get_overlap(hyp[i,:], hyp[j,:])
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
    
    if overlapvals[srt_idx[0]] < threshold:
        # If no hypotheses have overlap >= threshold, return the first one
        log.debug("No overlapping hypotheses, returning %s", index_arr[srt_idx[0],[0]])
        return index_arr[srt_idx[0],[0]]
    
    # Remove superfluous indexes (below threshold)
    srt_idx = srt_idx[overlapvals[srt_idx] >= threshold]
    clustered_idx = dict()
    # clusters = dict()
    cur_clust = 1
    log.debug(" --- No of overlaps above threshold: %d",srt_idx.shape[0])
    flag = True
    for idx in srt_idx:
        if cur_clust%100 == 0:
            if flag:
                flag = False
                log.debug("No of clusters: %d, no. of idx covered: %d, current overlap: %.2f", cur_clust-1, len(clustered_idx.keys()), overlapvals[idx])
        else:
            flag = True
        if overlapvals[idx] < threshold:
            # Return the then biggest cluster, not used if correct
            return get_largest_cluster(clustered_idx)
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
                    # clusters[cur_clust] = [i,j]
                    cur_clust += 1
                else:
                    # If j has a cluster, but i not: Add i to cluster of j
                    # log.debug('Adding i=%d to cluster %d of j=%d'%(i,clustered_idx[j],j))
                    clustered_idx[i] = clustered_idx[j]
                    # clusters[clustered_idx[j]].append(i)
            else:
                if not j in clustered_idx:
                    # If i has a cluster, but j not: Add j to cluster of i
                    # log.debug('Adding j=%d to cluster %d of i=%d'%(j,clustered_idx[i],i))
                    clustered_idx[j] = clustered_idx[i]
                    # clusters[clustered_idx[i]].append(j)
                elif not clustered_idx[i] == clustered_idx[j]:
                    # If both are clustered, but not yet in to the same
                    # cluster, merge them
                    # print 'Merging cluster %d=%s with %d=%s into %d=%s'%(clustered_idx[i],clusters[clustered_idx[i]],clustered_idx[j],clusters[clustered_idx[j]],cur_clust,clusters[clustered_idx[i]] + clusters[clustered_idx[j]])
                    # clusters[cur_clust] = clusters[clustered_idx[i]] + clusters[clustered_idx[j]]
                    c1 = clustered_idx[i]
                    c2 = clustered_idx[j]
                    for k,v in clustered_idx.items():
                        if v == c2:
                            # Add c2 to c1
                            clustered_idx[k] = c1
                #else:
                # already clustered to the same cluster. Not needed to cluster
    # If the for-loop runs out, all overlaps are above threshold:
    return get_largest_cluster(clustered_idx)

def get_largest_cluster(idxs):
    log.debug(" --- Calculating largest cluster of %d indexes", len(idxs.keys()))
    clusters = dict()
    for v in idxs.values():
        if not v in clusters:
            clusters[v] = 1
        else:
            clusters[v] += 1
    largest = sorted([(sz,k) for (k,sz) in clusters.items()], reverse=True)[0]
    log.debug(" --- Found %d clusters, largest is %d: sz %d", len(clusters.keys()), largest[1],largest[0])
    cluster = np.array([idx for (idx,clust) in idxs.items() if clust == largest[1]])
    return cluster
    
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
    
def merge_cluster(cluster_of_hyp):
    """ Make a detection [xmin, ymin, xmax, ymax]
    
    """
    log.debug(" --- Merged cluster of size: %s.",cluster_of_hyp.shape)
    log.debug(" ---  max values per row: %s", cluster_of_hyp.max(0))
    log.debug(" ---  min values per row: %s", cluster_of_hyp.min(0))
    
    det = cluster_of_hyp.mean(0)
    log.debug(" ---  mean values per row: %s (= detection)", cluster_of_hyp.mean(0))
    return det
    
def remove_cluster(cluster, det_bbox, hypotheses, hvalues, overlap, indexes, threshold=0.0):
    """ overlap thereshold theta p, set by Becker to 0, which means everything 
    with just a little overlap is removed
    
    """
    
    n = hypotheses.shape[0]
    hyp_left = np.sum(~(hvalues[:]==0))
    to_be_removed = np.zeros(n, dtype=bool)
    for i in xrange(n):
        if i in cluster:
            # All hypotheses in the detection cluster have to be removed
            to_be_removed[i] = True
        elif not hvalues[i] == 0 and get_overlap(hypotheses[i,:], det_bbox) > threshold:
            # If overlap too big, remove too
            to_be_removed[i] = True
    log.debug(' ---  hypotheses to be removed: %d out of %d', to_be_removed.sum(), hyp_left)
    
    hvalues[to_be_removed] = 0
    log.debug(' ---  New length: %d (hypotheses actually removed: %d)', np.sum(~(hvalues[:]==0)), hyp_left-np.sum(~(hvalues[:]==0)))
    if np.sum(~(hvalues[:]==0)) == 0:
        log.debug(" ---  all is clustered, returning None")
        # Nothing to do anymore: everything is clustered.
        return None, None, None
    else:
        # Remove overlap values & indexes for hypotheses that will be removed
        overlap_to_be_removed = np.zeros(overlap.shape[0],dtype=bool)
        for i in xrange(indexes.shape[0]):
            if to_be_removed[indexes[i,0]] or to_be_removed[indexes[i,1]]:
                overlap_to_be_removed[i]=True
        log.debug(' ---  overlaps to be removed: %d out of %d', overlap_to_be_removed.sum(), overlap.shape[0])
        overlap =overlap[~overlap_to_be_removed]
        log.debug(' ---  New length: %d (overlaps actually removed: %d)', overlap.shape[0], indexes.shape[0]-overlap.shape[0])
        
        return hvalues, overlap, indexes[~overlap_to_be_removed]

if __name__ == '__main__':
    import doctest
    import numpy as np
    N = 3
    k = 4
    exemplars = np.zeros([N, k, 4])
    points    = np.zeros([N, 5])
    distances = np.zeros([N, 2, k])
    for n in range(N):
        m=n+1
        points[n] = [m*11,m*111,m*1111,0,0]
        for kk in range(k):
            ll=kk+1
            e_val = m+(ll*.10)
            exemplars[n,kk] = e_val
            distances[n,:,kk] = [ll,ll*(-1)**ll]
    t = 'becker'
    doctest.testmod()