
import sys
from nbnn import descriptor, nbnn
from nbnn.voc import VOC
from utils import *
from procedures import *

if __name__ == "__main__":
    
    # Get config settings
    if len(sys.argv) < 3:
        raise Exception("Please give a config file and tmpdir" 
            "file as command line argument")
    configfile = sys.argv[1]
    tmpdir = sys.argv[2]
    
    VOCopts = VOC.fromConfig(configfile)
    DESCRopts, NBNNopts, TESTopts = getopts(configfile, tmpdir)
    
    # Setup logger
    log = init_log(TESTopts['log_path'], 'main', 'w')
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = descriptor.DescriptorUint8(**DESCRopts)
    log.info('==== INIT ESTIMATOR ====')
    estimator = nbnn.NBNNEstimator(**NBNNopts)
    
    # Train
    train_voc(descriptor_function, estimator, 'fgbg',  VOCopts, \
        TESTopts['descriptor_path'])
    
    #log.info('==== REMOVING TRAIN DESCRIPTORS FROM DISK ====')
    #delete_descriptor_file(TESTopts['descriptor_path'])
    
    # Save descriptors of test set to disk
    make_voc_tests(descriptor_function, VOCopts, TESTopts)
    