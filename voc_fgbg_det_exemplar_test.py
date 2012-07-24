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
    
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    DESCRopts, NBNNopts, TESTopts, DETECTIONopts = get_detection_opts(configfile, tmpfile)
    
    # Setup logger
    if batch_no == 1:
        mode = 'w'
    else:
        mode = 'a'
    log = init_log(TESTopts['log_path'], cls, mode)

    
    log.info("TEST cfg:%s, batch_no:%d, cls:%s",configfile, batch_no, cls)

    log.info('==== LOAD IMAGE PICKLE ====')
    with open(TESTopts['img_pickle_path']%batch_no,'rb') as pklf:
        images = cPickle.load(pklf)
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = descriptor.DescriptorUint8(**DESCRopts)
    log.info('==== INIT ESTIMATOR ====')
    estimator = nbnn.NBNNEstimator(**NBNNopts)
    
    log.info('==== LOAD IMAGE DESCRIPTORS ====')
    descriptors = get_image_descriptors(images, descriptor_function, \
        TESTopts['descriptor_path'])
    image_list = [(im_id, p, d) for im_id, (p, d) in descriptors.items()]
    del descriptors
    num_descriptors = [d.shape[0] for i,p,d in image_list]
    descriptors_array = np.vstack([d for i,p,d in image_list])
    points_list = [p for i,p,d in image_list]
    im_ids = [i for i,p,d in image_list]
    del image_list

    log.info('==== GET ESTIMATES ====')
    
    # Getting fgbg estimates for full image

    log.info("Getting estimates for %s descriptors."%descriptors_array.shape[0])

    cls_dst = np.zeros((descriptors_array.shape[0], 2), np.double)
    cls_dst[:,0], nn_descr_idxs = estimator.get_class_estimates(cls+'_fg', descriptors_array, return_result=True)
    cls_dst[:,1] = estimator.get_class_estimates(cls+'_bg', descriptors_array)

    log.debug("--returning array of shape %s"%(cls_dst.shape,))
    log.debug("--mean estimate for each class: %s"%np.mean(cls_dst,0))
    log.debug("--max estimate for each class: %s"%np.max(cls_dst,0))
    log.debug("--min estimate for each class: %s"%np.min(cls_dst,0))
    log.debug("--min/max descr_idx for fg class:%d/%d",nn_descr_idxs.min(),nn_descr_idxs.max())
    distances = []
    index = 0
    for k in num_descriptors:
        distances.append(cls_dst[index:index+k,:])
        index += k
    del cls_dst
    
    log.info('==== DEFINE WHICH DESCRIPTORS TO USE FOR HYPOTHESES ====')
    
    with open(DETECTIONopts['exemplar_path']%cls, 'rb') as exf:
        exemplars = np.vstack(cPickle.load(exf))
        # exemplars is an np.array, nx4, where n=no of exemplars in a class
        # the cols are [rel_bb_w, rel_bb_h, rel_x, rel_y]
    
    # Each detection is a tuple (Qd, Qh, im_id, xmin, ymin, xmax, ymax)
    detections = []
    
    for i, dist_arr in enumerate(distances):
        log.info("==== FINDING DETECTIONS FOR IMAGE %s", im_ids[i])
        # Select all descriptors where d+ < d- (i.e. second col > first col)
        positives = dist_arr.argmax(1)==1
        log.debug('No of positive descriptors: %d',positives.sum())
        # Get the QH quality measure (relative distance of fg to bg class)
        QH = (dist_arr[positives,1]-dist_arr[positives,0])/dist_arr[positives,0]
        # Get the foreground points
        fg_points = points_list[i][positives,:]
        # get the indexes of the class's exemplars
        exemplar_idxs = nn_descr_idxs[positives]
        # Get the exemplars that are nn of the image's descriptors
        im_exemplars = exemplars[exemplar_idxs, :]
        log.debug('No of positive exemplars in image %s: %s',im_ids[i],im_exemplars.shape)
        # Make hypotheses from the positive points and their exemplars
        # hypotheses = nx5 array, where n=no of positive descriptors, 
        # the cols define a bbox and its quality Qh: [Qh, xmin, ymin, xmax, ymax]
        hypotheses = np.zeros([positives.sum(),5], np.float32)

        # Qh
        hypotheses[:,0] = QH
        # xmin = point_x - (rel_x * rel_bb_w * point_sigma)
        # [which means: start the bbox at the x_location, subtracted by the
        # converted with to the new scale (rel_bb_w * scale) times the relative
        # x_pos of the exemplar to its bbox]
        log.debug("Trying to fill %s hypotheses from %s fg_points and %s im_exemplars",hypotheses.shape, fg_points.shape, im_exemplars.shape)
        hypotheses[:,1] = fg_points[:,0]-(im_exemplars[:,2] * im_exemplars[:,0] * fg_points[:,2])
        # ymin = point_y - (rel_y * rel_bb_h * point_sigma)
        hypotheses[:,2] = fg_points[:,1]-(im_exemplars[:,3] * im_exemplars[:,1] * fg_points[:,2])
        # xmax = point_x + (rel_x * rel_bb_w * point_sigma)
        hypotheses[:,3] = fg_points[:,0]+(im_exemplars[:,2] * im_exemplars[:,0] * fg_points[:,2])
        # ymax = point_y + (rel_y * rel_bb_h * point_sigma)
        hypotheses[:,4] = fg_points[:,1]+(im_exemplars[:,3] * im_exemplars[:,1] * fg_points[:,2])
        
        # get pairwise overlap (don't have to calculate each time)
        overlap, indexes = get_pairwise_overlap(hypotheses)
        log.debug('Mean overlap:%.5f',overlap.mean())
        log.info('  == CLUSTERING HYPOTHESES OF %s==',im_ids[i])
        while not hypotheses == None:
            # Cluster hypotheses' overlap to get the biggest cluster (it's hypotheses' indexes)
            log.debug('  no. of hypothesis left: %d',hypotheses.shape[0])
            log.debug('  no. of overlaps left: %d, should be %d/2 * (%d-1)=%.1f',overlap.shape[0], hypotheses.shape[0],hypotheses.shape[0],hypotheses.shape[0]/2.0*(hypotheses.shape[0]-1))
            
            best_cluster_idx = cluster_hypotheses(overlap, indexes, DETECTIONopts['theta_m'])
            # merge the biggest cluster of hypotheses into a detection, and append it
            log.debug(' Best cluster size: %d',best_cluster_idx.shape[0])
            detections.append(merge_cluster(hypotheses[best_cluster_idxs], im_ids[i]))
            # Select hypotheses to remove based on the cluster and the removal threshold theta_p
            hypotheses, overlap, indexes = remove_clusters(best_cluster_idx, detection[3:], hypotheses,overlap, indexes, DETECTIONopts['theta_p'])
            
    log.debug(' Found %d Detections', len(detections))
    # Save detections of image to resultsfiles
    # Save detections only, do not rank yet, because of batches...
    log.info('==== SAVE CONFIDENCE VALUES ====')
    save_to_pickle(TESTopts['result_path']%cls, detections)
    
    
    # 
    # log.info('==== SAVE DISTANCES ====')
    # with open (TESTopts['res_folder']+'/distances_%s.pkl'%cls, 'wb') as dfile:
    #     cPickle.dump(distances, dfile)
    #     cPickle.dump(points_list, dfile)
    #     cPickle.dump(im_ids, dfile)
    #     cPickle.dump(images, dfile)
    
