import sys
import numpy as np
from logutils import *
from nbnn import *
from nbnn.voc import *
from utils import *

from detection_utils import *
from file_io import *

if __name__ == '__main__':
    np.seterr(all='raise')
    
    # Get config settings
    if len(sys.argv) < 4:
        raise Exception("arguments expected: cfgfile batch_no class")
    configfile = sys.argv[1]
    batch_no = int(sys.argv[2])
    cls = sys.argv[3]
    
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)

    # Setup logger
    log = init_log(GLOBopts['log_path'], 'nn_%s_%d'%(cls, batch_no), 'w')
    
    log.info("NN cfg:%s, batch_no:%d, cls:%s",configfile, batch_no, cls)
    
    log.info('==== LOAD IMAGE PICKLE ====')
    images = load_batch(TESTopts['img_pickle_path']%batch_no)
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = init_descriptor(DESCRopts[1])
    
    log.info('==== INIT ESTIMATOR ====')
    estimator = init_estimator(GLOBopts['nbnn_path']%cls, NBNNopts)
    
    log.info('==== LOAD IMAGE DESCRIPTORS ====')
    descriptors = get_image_descriptors(images, descriptor_function, \
        GLOBopts['descriptor_path'])
    
    # Sort descriptors points & images such that they have the same order...
    descriptors_array, points_list, images, num_descriptors = sort_descriptors(descriptors, images)

    log.info('==== GET ESTIMATES ====')
    
    # Getting fgbg estimates for full image
    log.info("Getting estimates for %s descriptors.", descriptors_array.shape[0])

    # Get distances
    if 'setmode' in GLOBopts and GLOBopts['setmode'] == 'becker':
        cls_dst, nn_descr_idxs = estimator.get_estimates([cls,'background'], descriptors_array, return_result=True)
    else:
        cls_dst, nn_descr_idxs = estimator.get_estimates([cls+'_fg',cls+'_bg'], descriptors_array, return_result=True)
    del descriptors_array
    
    log.debug("-- returning array of shape %s"%(cls_dst.shape,))
    log.debug("-- mean estimate for each class: %s"%np.mean(cls_dst,0))
    log.debug("-- max estimate for each class: %s"%np.max(cls_dst,0))
    log.debug("-- min estimate for each class: %s"%np.min(cls_dst,0))
    log.debug("-- min/max descr_idx for fg class:%d/%d",nn_descr_idxs.min(),nn_descr_idxs.max())
    log.debug("-- no of descr_indexes %s",nn_descr_idxs.shape)
    
    # Put distances into a list (per image)
    # and put exemplar_indexes in a list too
    distances = []
    nearest_exemplar_indexes = []
    index = 0
    for k in num_descriptors:
        distances.append(cls_dst[index:index+k,:])
        nearest_exemplar_indexes.append(nn_descr_idxs[index:index+k,:])
        index += k
    del cls_dst
    del nn_descr_idxs
    
    log.info('==== SAVE DISTANCES ====')
    save_distances(DETopts[1]['distances_path'], cls, distances, points_list, \
        images, nearest_exemplar_indexes)
    log.info('==== NN FINISHED ====')
