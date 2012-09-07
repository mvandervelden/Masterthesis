
import sys
from nbnn.voc import VOC
from utils import *
from procedures import *
from io import *

if __name__ == "__main__":
    
    # Get config settings
    if len(sys.argv) < 2:
        raise Exception("Please give a config file as command line argument")
    configfile = sys.argv[1]
    
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)

    # Setup logger
    log = init_log(GLOBopts['log_path'], 'training', 'w')
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = init_descriptor(DESCRopts[0])
    
    for cls in VOCopts.classes:
        log.info('==== INIT ESTIMATOR FOR CLASS %s ====', cls)
        estimator = init_estimator(GLOBopts['nbnn_path']%cls, NBNNopts)
        
        # Train
        if NBNNopts[0] == 'behmo':
            load_behmo_estimator(descriptor_function, estimator, cls, VOCopts, \
                train_set = GLOBopts['train_set'],\
                descriptor_path = GLOBopts['descriptor_path'], \
                exemplar_path = DETopts[1]['exemplar_path'])
    
            train_behmo(descriptor_function, estimator, cls, VOCopts, \
                val_set = GLOBopts['val_set'], \
                descriptor_path = GLOBopts['descriptor_path'])
        elif NBNNopts[0] == 'boiman':
            train_voc(descriptor_function, estimator, 'fgbg', VOCopts,\
                train_set = GLOBopts['train_set'], \
                exemplar_path = DETopts[1]['exemplar_path'], \
                descriptor_path = GLOBopts['descriptor_path'], \
                cls = cls)
    log.info('==== TRAINING FINISHED ====')
