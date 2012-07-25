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
    del descriptors_array
    
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
    
    """ get exemplars and hypotheses per image, save these per image"""
    
    log.info('==== GET EXEMPLARS ====')
    
    with open(DETECTIONopts['exemplar_path']%cls, 'rb') as exf:
        exemplars = np.vstack(cPickle.load(exf))
        # exemplars is an np.array, nx4, where n=no of exemplars in a class
        # the cols are [rel_bb_w, rel_bb_h, rel_x, rel_y]
    
    # Each detection will be a tuple (Qd, Qh, im_id, xmin, ymin, xmax, ymax)
        
    for i, dist_arr in enumerate(distances):
        log.info("==== FINDING DETECTIONS FOR IMAGE %s", im_ids[i])
        # Select all descriptors where d+ < d- (i.e. second col > first col)
        positives = dist_arr.argmax(1)==1
        log.debug(' --- No of positive descriptors: %d',positives.sum())
        # Get the QH quality measure (relative distance of fg to bg class)
        QH = (dist_arr[positives,1]-dist_arr[positives,0])/dist_arr[positives,0]
        log.debug(' --- Average QH value (rel dist d+ to d-) of positives: %f', QH.mean())
        log.debug(' --- Max QH value (rel dist d+ to d-) of positives: %f', QH.max())
        log.debug(' --- Min QH value (rel dist d+ to d-) of positives: %f > 0.0', QH.max())
        # Get the foreground points
        fg_points = points_list[i][positives,:]
        # get the indexes of the class's exemplars
        exemplar_idxs = nn_descr_idxs[positives]
        # Get the exemplars that are nn of the image's descriptors
        im_exemplars = exemplars[exemplar_idxs, :]
        log.debug(' --- No of positive exemplars in image %s: %s',im_ids[i],im_exemplars.shape)
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
        log.debug(" --- Trying to fill %s hypotheses from %s fg_points and %s im_exemplars",hypotheses.shape, fg_points.shape, im_exemplars.shape)
        hypotheses[:,1] = fg_points[:,0]-(im_exemplars[:,2] * im_exemplars[:,0] * fg_points[:,2])
        # ymin = point_y - (rel_y * rel_bb_h * point_sigma)
        hypotheses[:,2] = fg_points[:,1]-(im_exemplars[:,3] * im_exemplars[:,1] * fg_points[:,2])
        # xmax = point_x + (rel_x * rel_bb_w * point_sigma)
        hypotheses[:,3] = fg_points[:,0]+(im_exemplars[:,2] * im_exemplars[:,0] * fg_points[:,2])
        # ymax = point_y + (rel_y * rel_bb_h * point_sigma)
        hypotheses[:,4] = fg_points[:,1]+(im_exemplars[:,3] * im_exemplars[:,1] * fg_points[:,2])
        log.debug(" --- First 10 hypotheses:")
        for t in range(10):
            log.debug(" ---    hyp: %s, fg_point: %s, exemplar: %s",(hypotheses[t,:], fg_points[t,:], im_exemplars[t,:]))
        
        log.info('==== SAVING HYPOTHESES ETC. ====')
        with open(DETECTIONopts['hypotheses_path']%(cls,im_ids[i]), 'wb') as dfile:
            cPickle.dump(hypotheses, dfile)
            cPickle.dump(fg_points, dfile)
            cPickle.dump(im_exemplars[:.3])
            cPickle.dump(dist_arr, fdile)
