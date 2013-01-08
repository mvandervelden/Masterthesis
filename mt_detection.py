import sys, cPickle
import numpy as np

from logutils import *
from nbnn import *
from nbnn.voc import *
from utils import *
from file_io import *
from detection import *


if __name__ == '__main__':
    np.seterr(all='raise')
    # Get config settings
    if len(sys.argv) < 5:
        raise Exception("arguments expected: cfgfile batch_no class im_id")
    configfile = sys.argv[1]
    batch_no = int(sys.argv[2])
    cls = sys.argv[3]
    im_id = sys.argv[4]
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)

    # Setup logger
    log = init_log(GLOBopts['log_path'], 'det_%s_%s'%(im_id, cls), 'w')
    
    log.info("DETECTION cfg:%s, batch_no:%d, im_id:%s, cls:%s",configfile, batch_no, im_id, cls)
    
    DETmode = DETopts[0]
    DETopts = DETopts[1]
    
    # TODO Make a multithreaded thing to do detection, later perhaps finding distances too!
