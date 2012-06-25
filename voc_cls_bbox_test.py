from utils import *
from nbnn import *
from nbnn.voc import *
import sys, cPickle
from ConfigParser import RawConfigParser

if __name__ == '__main__':
    # Get config settings
    if len(sys.argv) < 5:
        raise Exception("arguments expected: cfgfile batch_no class logcfg")
    configfile = sys.argv[1]
    tmpfile = sys.argv[2]
    batch_no = int(sys.argv[3])
    cls = sys.argv[4]
    logconfig = sys.argv[5]
    
    VOCopts = VOC.fromConfig(configfile)
    cfg = RawConfigParser()
    cfg.read(configfile)
    DESCRopts = dict(cfg.items("DESCRIPTOR"))
    NBNNopts = dict(cfg.items("NBNN"))
    TESTopts = dict(cfg.items("TEST"))

    # Setup logger
    log = init_log(logconfig)
    
    DESCRopts['cache_dir'] = '/'.join([tmpdir, DESCRopts['cache_dir']])
    NBNNopts['nbnn_dir'] = '/'.join([tmpdir, NBNNopts['nbnn_dir']])
    descriptor_dir = DESCRopts['cache_dir']
    TESTopts['descriptor_path'] = '/'.join([tmpdir, TESTopts['descriptor_path']])
    TESTopts['img_pickle_path'] = '/'.join([tmpdir, TESTopts['img_pickle_path']])
    
    
    log.info("TEST cfg:%s, logcfg:%s, batch_no:%d, cls:%s",configfile, logconfig, batch_no,cls)

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
    log.info('==== GET OBJECT DESCRIPTORS FROM IMAGES ====')
    objects = get_objects(images)
    descriptors = get_bbox_descriptors(objects, descriptors)
    
    distances = estimator.get_estimates([cls], [d for p,d in descriptors.values()])
    log.info('==== GET CONFIDENCE VALUES ====')
    conf_vals = get_confidence_values(distances)
    log.info('== SAVE CONFIDENCE VALUES ==')
    save_results_to_file(TESTopts['result_path']%cls, objects, conf_vals)
    
