from utils import *
from nbnn import *
import sys, cPickle
from ConfigParser import RawConfigParser

if __name__ == '__main__':
    # Get config settings
    if len(sys.argv) < 5:
        raise Exception("arguments expected: cfgfile batch_no class logcfg")
    configfile = sys.argv[1]
    batch_no = int(sys.argv[2])
    cls = sys.argv[3]
    logconfig = sys.argv[4]
    
    print 'configfile: ', configfile
    print 'batch_no: ',batch_no
    print 'cls: ', cls
    print 'logcfg:',logconfig
    
    VOCopts = voc.VOC.fromConfig(configfile)
    cfg = RawConfigParser()
    cfg.read(configfile)
    FLANNopts = dict(cfg.items("FLANN"))
    DESCRopts = dict(cfg.items("DESCRIPTOR"))
    NBNNopts = dict(cfg.items("NBNN"))
    TESTopts = dict(cfg.items("TEST"))

    # Setup logger
    log = init_log(logconfig)
    log.info("TEST cfg:%s, logcfg:%s, batch_no:%d, cls:%s",configfile, logconfig, batch_no,cls)
    if False:
        log.info('==== LOAD IMAGE PICKLE ====')
        with open(TESTopts['img_pickle_path']%batch_no,'rb') as pklf:
            images = cPickle.load(pklf)
    
        log.info('==== INIT DESCRIPTOR FUNCTION ====')
        descriptor_function = DescriptorUint8(**DESCRopts)
        log.info('==== INIT ESTIMATOR ====')
        nbnn = NBNNEstimator(**NBNNopts)
    
        log.info('==== LOAD IMAGE DESCRIPTORS ====')
        descriptors = get_image_descriptors(images, descriptor_function, \
            VOCopts['descriptor_path'])
        log.info('==== GET ESTIMATES ====')
        distances = nbnn.get_estimates([cls], descriptors)
        log.info('==== GET CONFIDENCE VALUES ====')
        conf_vals = get_conf_vals(distances)
        log.info('== SAVE CONFIDENCE VALUES ==')
        save_to_file(conf_vals, cls)
    
