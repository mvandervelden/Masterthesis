
import sys, os, os.path
from nbnn import *
from nbnn.voc import *
from utils import *
import cPickle
from ConfigParser import RawConfigParser

if __name__ == "__main__":
    
    # Get config settings
    if len(sys.argv) < 4:
        raise Exception("Please give a config file, logconfig file and testinfo" 
            "file as command line argument")
    configfile = sys.argv[1]
    logconfig = sys.argv[2]
    testinfofile = sys.argv[3]
    
    VOCopts = VOC.fromConfig(configfile)
    
    cfg = RawConfigParser()
    cfg.read(configfile)
    DESCRopts = dict(cfg.items("DESCRIPTOR"))
    NBNNopts = dict(cfg.items("NBNN"))
    TESTopts = dict(cfg.items("TEST"))
    
    # Setup logger
    log = init_log(logconfig)
    
    # Make sure some folders exist
    descriptor_dir = '/'.join(TESTopts['descriptor_path'].split('/')[:-1])
    if not os.path.exists(descriptor_dir):
        os.mkdir(descriptor_dir)
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = descriptor.DescriptorUint8(**DESCRopts)
    log.info('==== INIT ESTIMATOR ====')
    estimator = nbnn.NBNNEstimator(**NBNNopts)
    # Perhaps we want to delete the 'checks' parameter to make sure enough
    # precision is got
    if 'target_precision' in NBNNopts:
        del estimator.flann_args['checks']
    
    for i,cls in enumerate(VOCopts.classes):
        log.info('==== GET CLASS %d: %s IMAGES ====', i, cls)
        img_set = read_image_set(VOCopts,cls+'_train')
        log.info('==== GET %s DESCRIPTORS ====', cls)
        descriptors = get_image_descriptors(img_set, descriptor_function, \
            TESTopts['descriptor_path'])
        log.info('==== ADD %s DESCRIPTORS TO ESTIMATOR', cls)
        estimator.add_class(cls, [d for p,d in descriptors.values()])
    log.info('==== REMOVING TRAIN DESCRIPTORS FROM DISK ====')
    delete_descriptor_file(TESTopts['descriptor_path'])
       
    # Save descriptors of test set to disk
    log.info('==== GENERATING TEST IMAGES =====')
    test_images = read_image_set(VOCopts,'test')
    log.info('==== GENERATING AND SAVING TEST DESCRIPTORS =====')
    save_image_descriptors(test_images, descriptor_function, TESTopts['descriptor_path'])
    batches = get_image_batches(VOCopts, test_images, \
        int(TESTopts['batch_size']))
    log.info('==== SAVING IMAGE OBJECTS PER BATCH =====')
    for b,batch in enumerate(batches):
        with open(TESTopts['img_pickle_path']%(b+1), 'wb') as pklfile:
            cPickle.dump(batch, pklfile)
    log.info('==== SAVING TESTINFORMATION =====')
    save_testinfo(testinfofile, batches, VOCopts.classes)
   
    
