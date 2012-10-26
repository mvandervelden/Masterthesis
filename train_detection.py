import sys
from logutils import *
from nbnn.voc import VOC
from utils import *
from procedures import *
from file_io import *

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
    
    if ('setmode' in GLOBopts and not GLOBopts['setmode'] == 'becker') or \
            not 'setmode' in GLOBopts:
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
    elif GLOBopts['setmode'] == 'becker':
        log.info('==== INIT ESTIMATOR FOR BECKER TRAINING ====', )
        estimator = init_estimator(GLOBopts['nbnn_path']%'motorbike', NBNNopts)
        if NBNNopts[0] == 'behmo':
            load_becker_estimator(descriptor_function, estimator, VOCopts, \
                train_set = GLOBopts['train_set'],\
                descriptor_path = GLOBopts['descriptor_path'],\
                exemplar_path = DETopts[1]['exemplar_path'])
            train_behmo(descriptor_function, estimator, cls, VOCopts, \
                val_set = GLOBopts['val_set'], \
                descriptor_path = GLOBopts['descriptor_path'])
        elif NBNNopts[0] == 'boiman':
            load_becker_estimator(descriptor_function, estimator, VOCopts, \
                train_set = GLOBopts['train_set'],\
                descriptor_path = GLOBopts['descriptor_path'],\
                exemplar_path = DETopts[1]['exemplar_path'])

    log.info('==== TRAINING FINISHED ====')
