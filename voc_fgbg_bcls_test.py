from utils import *
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
    DESCRopts, NBNNopts, TESTopts = getopts(configfile, tmpfile)
    
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
    descr_list = [(im_id, p[:,:3], d) for im_id, (p, d) in descriptors.items()]
    log.info('==== GET ESTIMATES ====')
    # Getting fgbg estimates for full image
    distances = estimator.get_estimates([cls+'_fg',cls+'_bg'], [d for i,p,d in descr_list])
    log.info('==== SAVE DISTANCES ====')
    with open (TESTopts['res_folder']+'/distances_%s.pkl'%cls, 'wb') as dfile:
        cPickle.dump(distances, dfile)
        cPickle.dump([p for i,p,d in descr_list], dfile)
        cPickle.dump([im_id for im_id,p,d in descr_list], dfile)
        cPickle.dump(images, dfile)
    # log.info('==== GET OBJECT DESCRIPTORS FROM IMAGES ====')
    # objects = get_objects(images)
    # descriptors = get_bbox_descriptors(objects, descriptors)
    # log.info('==== GET BBOX ESTIMATES ====')
    # distances = estimator.get_estimates([cls+'_fg',cls+'_bg']], descriptors)
    # 
    # log.info('==== GET CONFIDENCE VALUES ====')
    #     conf_vals = get_confidence_values(distances)
    #     log.info('== SAVE CONFIDENCE VALUES ==')
    #     save_results_to_file(TESTopts['result_path']%cls, objects, conf_vals)
    #     
