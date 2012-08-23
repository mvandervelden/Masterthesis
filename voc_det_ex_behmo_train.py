
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
    DESCRopts, NBNNopts, TESTopts, DETECTIONopts, test_scalings = get_detection_opts(configfile, tmpdir)
    
    # Setup logger
    log = init_log(TESTopts['log_path'], 'main', 'w')
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = descriptor.DescriptorUint8(**DESCRopts)
    nbnn_dir = NBNNopts['nbnn_dir']
    del NBNNopts['nbnn_dir']

    for cls in VOCopts.classes:
        log.info('==== INIT ESTIMATOR FOR CLASS %s ====', cls)
        
        estimator = nbnn.OptNBNNEstimator(nbnn_dir%cls, **NBNNopts)
        # Train
        load_behmo_estimator(descriptor_function, estimator, cls, VOCopts, \
            train_set=TESTopts['train_set'],\
            descriptor_path=TESTopts['descriptor_path'], \
            exemplar_path=DETECTIONopts['exemplar_path'])
    
        train_behmo(descriptor_function, estimator, cls, VOCopts, \
            val_set=TESTopts['val_set'], \
            descriptor_path = TESTopts['descriptor_path'])
    
    #log.info('==== REMOVING TRAIN DESCRIPTORS FROM DISK ====')
    #delete_descriptor_file(TESTopts['descriptor_path'])
    
    
    DESCRopts['ds_scales'] = test_scalings
    # Save descriptors of test set to disk
    make_voc_tests(descriptor_function, VOCopts, TESTopts)
    