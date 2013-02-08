import sys, cPickle
import numpy as np
from scipy.spatial import distance as sc_dist

from logutils import *
from nbnn import *
from nbnn.voc import *
from utils import *
from file_io import *
from detection_utils import *
from metric_functions import *
from quickshift import *

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
    
    DETmode = DETopts[0]
    DETopts = DETopts[1]
    
    log.info('==== LOADING DISTANCES ====')
    distances, points, image, nearest_exemplar_indexes = \
        load_distances(DETopts['distances_path']%(im_id, cls))
    # Only take into account the fg exemplars, there exist no bg exemplars.
    nearest_exemplar_indexes = nearest_exemplar_indexes[:,0]
        
    log.info('==== LOADING NEAREST EXEMPLARS ====')
    exemplars = load_exemplars(DETopts['exemplar_path']%cls, nearest_exemplar_indexes)
    
    log.info('==== GET HYPOTHESES ====')
    exemplars_th, points_th, distances_th = threshold_distances(exemplars, points, distances, DETopts['hyp_threshold'])
    if DETopts['ignore_threshold'] == 'True' and exemplars_th.shape[0] == 0:
        log.debug("== FOUND NO POINTS WITH fg_d < bg_d!!! CONTINUE WITHOUT THRESHOLDED DISTANCES...")
    elif exemplars_th.shape[0] == 0:
        log.debug("== FOUND NO POINTS WITH fg_d < bg_d!!! NO DETECTIONS FOR THIS CLASS-IMAGE PAIR")
        save_detections(GLOBopts['result_path']%(im_id, cls), np.array([]), np.array([]))
        exit()
    else:
        # Use thresholded distances
        exemplars = exemplars_th
        points = points_th
        distances = distances_th
    
    hypotheses = get_hypotheses(exemplars, points, image.width, image.height)
    if hypotheses.shape[0] == 0:
        log.debug("== FOUND NO HYPOTHESES WITH fg_d < bg_d. No clustering possible!")
    hvalues = get_hypothesis_values(hypotheses, distances, points, eval(DETopts['hypothesis_metric']))
    ranking = sort_values(hvalues)
    
    # Keep only the best n descriptors (largest relative margin d+, d-)
    if 'hyp_cutoff' in DETopts:
        log.info('Using %s hypotheses, out of %d', DETopts['hyp_cutoff'], hypotheses.shape[0])
        ranking = ranking[:int(DETopts['hyp_cutoff'])]
    hvalues = hvalues[ranking]
    hypotheses = hypotheses[ranking]
    # Make sure points and distances are selected and sorted in the same way, and saved with the detections
    points = points[ranking]
    distances = distances[ranking]
    log.debug(" -- first hyp: (%s, %.2f, last: hyp: (%s, %.2f)", hypotheses[0,:], \
        hvalues[0], hypotheses[-1,:], hvalues[-1])
    
    if DETopts['method'] == 'single_link':
        # get pairwise overlap (don't have to calculate each time)
        if DETopts['dist'] == 'overlap':
            overlap, indexes = get_pairwise_overlap(hypotheses)
        else:
            dist = sc_dist.pdist(hypotheses, DETopts['dist'])
            overlap = 1-(dist/dist.max())
            indexes = make_indexes(hypotheses.shape[0])
            
        log.debug('Mean overlap:%.5f',overlap.mean())
        log.info('  == CLUSTERING HYPOTHESES OF %s==',im_id)
        
        detections, dist_references = single_link_clustering(hypotheses, hvalues, overlap, indexes, ranking, DETopts)
    elif DETopts['method'] == 'quickshift':
        log.debug('qs_tree_path: %s', DETopts['quickshift_tree_path'])
        qs_path = DETopts['quickshift_tree_path']%(cls, im_id)
        log.debug('qs_tree_path: %s', qs_path)
        detections, dist_references = cluster_quickshift(hypotheses, DETopts['tau'], save_tree_path=qs_path)
    log.debug(' Found %d Detections', len(detections))
    # Save detections of image to resultsfiles
    # Save detections only, do not rank yet, because of batches...
    # dist_references: a list of length 'No_Detections', of lists that refer back to the original hypotheses, distances, points
    log.info('==== SAVE CONFIDENCE VALUES ====')
    save_detections(GLOBopts['result_path']%(im_id, cls), np.vstack(detections), dist_references, descr_distances=distances, descr_points=points)
    log.info('==== FINISHED DETECTION ====')
