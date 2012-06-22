
import sys, os, os.path
from nbnn import *
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
    
    VOCopts = voc.VOC.fromConfig(configfile)
    
    cfg = RawConfigParser()
    cfg.read(configfile)
    FLANNopts = dict(cfg.items("FLANN"))
    DESCRopts = dict(cfg.items("DESCRIPTOR"))
    NBNNopts = dict(cfg.items("NBNN"))
    TESTopts = dict(cfg.items("TEST"))
    
    # Setup logger
    init_log(logconfig)
    
    log.info("TEST cfg:%s, logcfg:%s, testinfofile:%s",configfile, logconfig,testinfofile)
    
    if False:
    
        log.info('==== INIT DESCRIPTOR FUNCTION ====')
        descriptor_function = DescriptorUint8(**DESCRopts)
        log.info('==== INIT ESTIMATOR ====')
        nbnn = NBNNEstimator(**NBNNopts)
    
        for i,cls in enumerate(VOCopts.classes):
            log.info('==== GET CLASS %d: %s IMAGES ====', i, cls)
            img_set = read_image_set(VOCopts,cls+'_train')
            #TODO define descriptor path (save descriptor files temporarily)
            log.info('==== GET %s DESCRIPTORS ====', cls)
            descriptors = get_image_descriptors(img_set, descriptor_function, \
                descriptor_path)
            log.info('==== ADD %s DESCRIPTORS TO ESTIMATOR', cls)
            nbnn.add_class(cls, descriptors)
        log.info('==== REMOVING TRAIN DESCRIPTORS FROM DISK ====')
        delete_descriptor_file(descriptor_path)
    
        # Save descriptors of test set to disk
        log.info('==== GENERATING TEST IMAGES =====')
        test_images = read_image_set(VOCopts,TESTopts['testset'])
        log.info('==== GENERATING AND SAVING TEST DESCRIPTORS =====')
        save_image_descriptors(test_images, descriptor_path)
        batches = get_image_batches(VOCopts, test_images, \
            TESTopts['batch_size'])
        log.info('==== SAVING IMAGE OBJECTS PER BATCH =====')
        for b,batch in enumerate(batches):
            with open(TESTopts['img_pickle_path']%b,'wb') as pklfile:
                cPickle.dump(batch, pklfile)
        log.info('==== SAVING TESTINFORMATION =====')
    else:
        batches = [1,2,3]
        
    with open(testinfofile,'w') as testfile:
        # Save a file with information on how many iterations with how many
        # classes, and which ones they are, for the multiple processes that are
        # gonna run the tests
        testfile.write("%d\n"%len(batches))
        testfile.write("%d\n"%len(VOCopts.classes))
        for cls in VOCopts.classes:
            testfile.write("%s\n"%cls)
    
