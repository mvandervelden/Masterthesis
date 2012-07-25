from utils import *
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
    tmpfile = sys.argv[2]
    batch_no = int(sys.argv[3])
    cls = sys.argv[4]
    im_id = sys.argv[5]
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    DESCRopts, NBNNopts, TESTopts, DETECTIONopts = get_detection_opts(configfile, tmpfile)
    
    # Setup logger
    if batch_no == 1:
        mode = 'w'
    else:
        mode = 'a'
    log = init_log(TESTopts['log_path'], cls, mode)

    
    log.info("DETECTION cfg:%s, batch_no:%d, im_id:%s, cls:%s",configfile, batch_no, im_id, cls)
    
    log.info('==== LOADING DISTANCES ====')
    with open(DETECTIONopts['hypotheses_path']%(cls,im_id), 'wb') as dfile:
        hypotheses = cPickle.load(dfile)
        
        # get pairwise overlap (don't have to calculate each time)
        overlap, indexes = get_pairwise_overlap(hypotheses)
        log.debug('Mean overlap:%.5f',overlap.mean())
        log.info('  == CLUSTERING HYPOTHESES OF %s==',im_id)
        while not hypotheses == None:
            # Cluster hypotheses' overlap to get the biggest cluster (it's hypotheses' indexes)
            log.debug('  no. of hypothesis left: %d',hypotheses.shape[0])
            log.debug('  no. of overlaps left: %d, should be %d/2 * (%d-1)=%.1f',overlap.shape[0], hypotheses.shape[0],hypotheses.shape[0],hypotheses.shape[0]/2.0*(hypotheses.shape[0]-1))
            
            best_cluster_idx = cluster_hypotheses(overlap, indexes, DETECTIONopts['theta_m'])
            # merge the biggest cluster of hypotheses into a detection, and append it
            log.debug(' Best cluster size: %d',best_cluster_idx.shape[0])
            detection = merge_cluster(hypotheses[best_cluster_idx], im_id)
            log.debug(' Detection found: %s',detection)
            detections.append(detection)
            # Select hypotheses to remove based on the cluster and the removal threshold theta_p
            hypotheses, overlap, indexes = remove_cluster(best_cluster_idx, detection[3:], hypotheses, overlap, indexes, DETECTIONopts['theta_p'])
            
    log.debug(' Found %d Detections', len(detections))
    # Save detections of image to resultsfiles
    # Save detections only, do not rank yet, because of batches...
    log.info('==== SAVE CONFIDENCE VALUES ====')
    save_to_pickle(TESTopts['result_path']%cls, detections)
    
