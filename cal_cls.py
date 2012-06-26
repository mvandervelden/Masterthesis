import sys
from nbnn import descriptor, nbnn
from utils import *
from procedures import *
from cal import *

if __name__ == "__main__":
    
    # Get config settings
    if len(sys.argv) < 3:
        raise Exception("Please give a config file, tmpdir file as command "
            "line argument")
    configfile = sys.argv[1]
    tmpdir = sys.argv[2]
    
    CALopts = Caltech.fromConfig(configfile)
    DESCRopts, NBNNopts, TESTopts = getopts(configfile, tmpdir)
    
    # Setup logger
    log = init_log(TESTopts['log_path'], 'main', 'w')
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = descriptor.XYDescriptor(**DESCRopts)
    log.info('==== INIT ESTIMATOR ====')
    estimator = nbnn.NBNNEstimator(**NBNNopts)
    
    # Select images
    train_images = dict()
    test_images = dict()
    for cls in CALopts.classes:
        train_images[cls], test_images[cls] = read_cal_image_set(CALopts, cls, \
            TESTopts['train_size'], TESTopts['test_size'])
    
    # Train
    train_cal(train_images, descriptor_function, estimator, CALopts, TESTopts)
    
    log.info('==== REMOVING TRAIN DESCRIPTORS FROM DISK ====')
    delete_descriptor_file(TESTopts['descriptor_path'])
    
    # Save descriptors of test set to disk
    make_cal_tests(test_images, descriptor_function, CALopts, TESTopts)
    