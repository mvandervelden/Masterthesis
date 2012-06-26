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
    
    VOCopts, DESCRopts, NBNNopts, TESTopts = getopts(configfile, tmpfile)

    # Setup logger
    if batch_no == 1:
        mode = 'w'
    else:
        mode = 'a'
    log = init_log(TESTopts['log_path'], cls, mode)
    
    log.info("TEST cfg:%s, batch_no:%d, cls:%s",configfile, batch_no,cls)

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
    log.info('==== GET ESTIMATES ====')
    distances = estimator.get_estimates([cls], [d for p,d in descriptors.values()])
    log.info('==== GET CONFIDENCE VALUES ====')
    conf_vals = get_confidence_values(distances)
    log.info('== SAVE CONFIDENCE VALUES ==')
    save_results_to_file(TESTopts['result_path']%cls, images, conf_vals)
    
