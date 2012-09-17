import sys
from nbnn.voc import VOC
from logutils import *
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
    log = init_log(GLOBopts['log_path'], 'mkbatches', 'w')
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = init_descriptor(DESCRopts[1])
    
    
    log.info('==== INIT BATCHES ====')
    # Save descriptors of test set to disk
    make_voc_batches(descriptor_function, VOCopts, GLOBopts, TESTopts)
    log.info('==== BATCHMAKING FINISHED ====')
    