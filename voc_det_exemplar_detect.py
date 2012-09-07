from utils import *
from io import save_to_pickle
from detection_utils import *
from nbnn import *
from nbnn.voc import *
import sys, cPickle
from ConfigParser import RawConfigParser

if __name__ == '__main__':
    # Get config settings
    if len(sys.argv) < 5:
        raise Exception("arguments expected: cfgfile batch_no class")
    configfile = sys.argv[1]
    tmpdir = sys.argv[2]
    batch_no = int(sys.argv[3])
    cls = sys.argv[4]
    im_id = sys.argv[5]
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    DESCRopts, NBNNopts, TESTopts, DETECTIONopts, test_scalings = get_detection_opts(configfile, tmpdir)
    
    # Setup logger
    if batch_no == 1:
        mode = 'w'
    else:
        mode = 'a'
    log = init_log(TESTopts['log_path'], im_id+cls, mode)

    
    log.info("DETECTION cfg:%s, batch_no:%d, im_id:%s, cls:%s",configfile, batch_no, im_id, cls)
    
    log.info('==== LOADING DISTANCES ====')
    with open(DETECTIONopts['hypotheses_path']%(cls,im_id), 'rb') as dfile:
        hypotheses = cPickle.load(dfile)
    
    # Keep only the best 2000 descriptors (largest relative margin d+, d-)
    if not 'no_hypotheses' in DETECTIONopts:
        DETECTIONopts['no_hypotheses'] = 2000
    else:
        DETECTIONopts['no_hypotheses'] = int(DETECTIONopts['no_hypotheses'])
    log.info('Using %d hypotheses, out of %d', DETECTIONopts['no_hypotheses'], hypotheses.shape[0])
    hypotheses = hypotheses[hypotheses[:,0].argsort()[::-1][:DETECTIONopts['no_hypotheses']]]
    
    # get pairwise overlap (don't have to calculate each time)
    overlap, indexes = get_pairwise_overlap(hypotheses)
    log.debug('Mean overlap:%.5f',overlap.mean())
    log.info('  == CLUSTERING HYPOTHESES OF %s==',im_id)
    detections = []
    Qds = []
    Qhs = []
    while not indexes == None:
        # Cluster hypotheses' overlap to get the biggest cluster (it's hypotheses' indexes)
        left = np.sum(~(hypotheses[:,0] == 0))
        log.debug('  no. of hypothesis left: %d',left)
        log.debug('  no. of overlaps left: %d, should be %d/2 * (%d-1)=%.1f',overlap.shape[0], left, left,left/2.0*(left-1))
        
        if left > 1:
            log.debug('  --  Clustering again')
            best_cluster_idx = cluster_hypotheses(overlap, indexes, DETECTIONopts['theta_m'])
        elif left == 1:
            log.debug('  --  No need for clustering, only 1 hypothesis left:')
            best_cluster_idx = np.array([(np.argsort(hypotheses[:,0] > 0))[-1]])
            
        # merge the biggest cluster of hypotheses into a detection, and append it
        log.debug(' Best cluster size: %d',best_cluster_idx.shape[0])
        Qd, Qh, detection = merge_cluster(hypotheses[best_cluster_idx], im_id)
        log.debug(' Detection found: %s, Qd: %d, Qh: %.2f',detection, Qd, Qh)
        detections.append(detection)
        Qds.append(Qd)
        Qhs.append(Qh)
        # Select hypotheses to remove based on the cluster and the removal threshold theta_p
        hypotheses, overlap, indexes = remove_cluster(best_cluster_idx, detection, hypotheses, overlap, indexes, DETECTIONopts['theta_p'])
            
    log.debug(' Found %d Detections', len(detections))
    # Save detections of image to resultsfiles
    # Save detections only, do not rank yet, because of batches...
    log.info('==== SAVE CONFIDENCE VALUES ====')
    save_to_pickle(TESTopts['result_path']%cls, [np.vstack(detections), np.hstack(Qds), np.hstack(Qhs), np.tile(im_id,len(detections))])
    
