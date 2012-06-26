import sys, cPickle
from ConfigParser import RawConfigParser
from nbnn import *
from nbnn.voc import *
from utils import *
from cal import *

if __name__ == '__main__':
    # Get config settings
    if len(sys.argv) < 5:
        raise Exception("arguments expected: cfgfile tmpfolder batch_no class")
    configfile = sys.argv[1]
    tmpfolder = sys.argv[2]
    batch_no = int(sys.argv[3])
    cls = sys.argv[4]
    
    CALopts = Caltech.fromConfig(configfile)
    DESCRopts, NBNNopts, TESTopts = getopts(configfile, tmpfolder)

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
    descriptor_function = descriptor.XYDescriptor(**DESCRopts)
    log.info('==== INIT ESTIMATOR ====')
    estimator = nbnn.NBNNEstimator(**NBNNopts)
    
    log.info('==== LOAD IMAGE DESCRIPTORS ====')
    descriptors = get_image_descriptors(images, descriptor_function, \
        TESTopts['descriptor_path'])
    log.info('==== GET ESTIMATES ====')
    distances = estimator.get_estimates([cls], [d for p,d in descriptors.values()])
    log.info('==== SUM ESTIMATES ====')
    # Now: sum, averaged over no of descriptors in image (is this okay?)
    distances = [d.sum()/d.shape[0] for d in distances]
    log.info('==== SAVE DISTANCES TO FILE ====')
    save_results_to_file(TESTopts['result_path']%cls, images, distances)
    
