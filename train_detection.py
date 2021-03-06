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
    
    if GLOBopts['setmode'] == 'voc':
        # VOC07 detection
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
                    fg_selection = GLOBopts['train_sel'], \
                    random_bg_images = GLOBopts['randbg'], \
                    random_bg_set = GLOBopts['bg_train_set'], \
                    cls = cls)
    elif GLOBopts['setmode'] == 'becker':
        # 'Becker' detection set (TUDmotorbikes with VOC07 for bg training)
        log.info('==== INIT ESTIMATOR FOR BECKER TRAINING ====', )
        estimator = init_estimator(GLOBopts['nbnn_path']%'motorbike', NBNNopts)
        if NBNNopts[0] == 'behmo':
            load_becker_estimator(descriptor_function, estimator, VOCopts, \
                train_set = GLOBopts['train_set'],\
                descriptor_path = GLOBopts['descriptor_path'],\
                exemplar_path = DETopts[1]['exemplar_path'])
            train_behmo_becker(descriptor_function, estimator, VOCopts, \
                val_set = GLOBopts['val_set'], \
                descriptor_path = GLOBopts['descriptor_path'])
        elif NBNNopts[0] == 'boiman':
            load_becker_estimator(descriptor_function, estimator, VOCopts, \
                train_set = GLOBopts['train_set'],\
                descriptor_path = GLOBopts['descriptor_path'],\
                exemplar_path = DETopts[1]['exemplar_path'])
    elif GLOBopts['setmode'] == 'cls_graz_person':
        # 'Graz01 detection set train/test: 100 negative (50 bg, 50 other class), 100 positive, randomly selected
        pass
    log.info('==== TRAINING FINISHED ====')
