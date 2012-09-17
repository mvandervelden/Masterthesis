import sys, cPickle
from logutils import *
from nbnn import *
from nbnn.voc import *
from utils import *
from file_io import *
from detection_utils import *
from metric_functions import *
import numpy as np

if __name__ == '__main__':
    np.seterr(all='raise')
    # Get config settings
    if len(sys.argv) < 5:
        raise Exception("arguments expected: cfgfile batch_no class im_id")
    configfile = sys.argv[1]
    batch_no = int(sys.argv[2])
    cls = sys.argv[3]
    im_id = sys.argv[4]
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)

    # Setup logger
    log = init_log(GLOBopts['log_path'], 'det_%s_%s'%(im_id, cls), 'w')
    
    log.info("DETECTION cfg:%s, batch_no:%d, im_id:%s, cls:%s",configfile, batch_no, im_id, cls)
    
    log.info('==== LOADING DISTANCES ====')
    #TODO: solve this

    if DETopts[0] == 'single_link':
        DETopts = DETopts[1]
        distances, points, image, nearest_exemplar_indexes = \
            load_distances(DETopts['distances_path']%(im_id, cls))
    
        log.info('==== LOADING NEAREST EXEMPLARS ====')
        exemplars = load_exemplars(DETopts['exemplar_path']%cls, nearest_exemplar_indexes)
    
        log.info('==== GET HYPOTHESES ====')
        exemplars, points, distances = threshold_distances(exemplars, points, distances, DETopts['hyp_threshold'])
        hypotheses = get_hypotheses(exemplars, points, image.width, image.height)
    
        hvalues = get_hypothesis_values(hypotheses, distances, points, eval(DETopts['hypothesis_metric']))
        ranking = sort_values(hvalues)
    
    
        # Keep only the best n descriptors (largest relative margin d+, d-)
        if 'hyp_cutoff' in DETopts:
            log.info('Using %s hypotheses, out of %d', DETopts['hyp_cutoff'], hypotheses.shape[0])
            ranking = ranking[:int(DETopts['hyp_cutoff'])]
        hvalues = hvalues[ranking]
        hypotheses = hypotheses[ranking]
        log.debug(" -- first hyp: (%s, %.2f, last: hyp: (%s, %.2f)", hypotheses[0,:], \
            hvalues[0], hypotheses[-1,:], hvalues[-1])
        if DETopts['method'] == 'single_link':
            # get pairwise overlap (don't have to calculate each time)
            overlap, indexes = get_pairwise_overlap(hypotheses)
            log.debug('Mean overlap:%.5f',overlap.mean())
            log.info('  == CLUSTERING HYPOTHESES OF %s==',im_id)
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
                    log.debug('  --  No need for clustering, only 1 hypothesis left:')
                    best_cluster_idx = np.where(hvalues > 0)[0]
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
            
    log.debug(' Found %d Detections', len(detections))
    # Save detections of image to resultsfiles
    # Save detections only, do not rank yet, because of batches...
    log.info('==== SAVE CONFIDENCE VALUES ====')
    save_detections(GLOBopts['result_path']%(im_id, cls), np.vstack(detections), dist_references)
    log.info('==== FINISHED DETECTION ====')
