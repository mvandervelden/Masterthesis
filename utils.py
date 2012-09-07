import logging, logging.config
import os, os.path, subprocess, cPickle
from ConfigParser import RawConfigParser
from nbnn.vocimage import *
from nbnn.voc import VOC
from nbnn import descriptor, nbnn
from cal import *

def points_in_bb(bb, pt_array):
    """Returns a binary list of all points falling within the bounding box
    supplied
    
    doctest:
    >>> points_in_bb([0,0,10,10], np.array([[0,0],[0,1],[0,5],[0,10],[0,20],[5,0],[5,5],[5,11],[10,0],[10,5],[10,12],[12,5],[12,12]]))
    array([ True,  True,  True,  True, False,  True,  True, False,  True,
            True, False, False, False], dtype=bool)
    """

    return (pt_array[:,0] >= bb[0]) & (pt_array[:,0] <= bb[2]) & \
        (pt_array[:,1] >= bb[1]) & (pt_array[:,1] <= bb[3])

def point_in_bb(x,y, bb):
    if x >= bb[0]-1 and x <= bb[2]-1 and y >= bb[1]-1 and y <= bb[3]-1:
        return True
    else:
        return False

def assert_dir(path):
    if not os.path.exists(path):
        if not '%' in path:
            os.mkdir(path)
        else:
            raise Exception("Trying to init a directory ''%s'', which is not fully formatted"%path)

def get_confidence_values(distances):
    cv = []
    for d in distances:
        no_descriptors = d.shape[0]
        # Sum or Sum of squared?
        cv.append(-d.sum()/no_descriptors)
    return cv

def sort_descriptors(descriptors, images):
    image_list = [(im_id, p, d) for im_id, (p, d) in descriptors.items()]
    del descriptors
    num_descriptors = [d.shape[0] for i,p,d in image_list]
    descriptors_array = np.vstack([d for i,p,d in image_list])
    points_list = [p for i,p,d in image_list]
    im_ids = [i for i,p,d in image_list]
    imgs = []
    del image_list
    for it1, ref_id in enumerate(im_ids):
        for it2, im in enumerate(images):
            if im.im_id == ref_id:
                # log.debug(' --- descr no %d (id:%s) == list no %d 
                # (id:%s)',it1,ref_id,it2,im.im_id)
                imgs.append(im)
                break
    images = imgs
    return descriptor_array, points_list, images, num_descriptors

def getopts(configfile):
    cfg = RawConfigParser()
    cfg.read(configfile)
    
    GLOBopts = dict(cfg.items("GLOBAL"))
    # Set global datatypes:
    if 'nn_threads' in GLOBopts:
        GLOBopts['nn_threads'] = int(GLOBopts['nn_threads'])
    if 'det_threads' in GLOBopts:
        GLOBopts['det_threads'] = int(GLOBopts['det_threads'])
    # Make sure folders exist:
    assert_dir(GLOBopts['tmp_dir'])
    assert_dir(GLOBopts['res_dir'])
    assert_dir(GLOBopts['res_dir']+'/logs')
    assert_dir(GLOBopts['tmp_dir']+'/descriptors')
    assert_dir(GLOBopts['tmp_dir']+'/nbnn')
    
    # Create important paths
    GLOBopts['result_path'] = GLOBopts['res_dir']+'/%s_%s.pkl'
    GLOBopts['log_path'] = GLOBopts['res_dir']+'/logs/%s.pkl'
    GLOBopts['descriptor_path'] = GLOBopts['tmp_dir']+'/descriptors/%s.dbin'
    GLOBopts['nbnn_path'] = GLOBopts['tmp_dir']+'/nbnn/%s'
    DESCRopts = [('',), ('',)]
    for i,d in enumerate(["TEST-DESCRIPTOR", "TRAIN-DESCRIPTOR"]):
        # Set datatypes
        DESCRopts[i][1] = dict(cfg.items(d))
        DESCRopts[i][0] = DESCRopts[i][1]['dtype']
        del DESCRopts[i][1]['dtype']
        if 'outputformat' in DESCRopts[i][1]:
            DESCRopts[i][1]['outputFormat'] = DESCRopts[i][1]['outputformat']
            del DESCRopts[i][1]['outputformat']
        # Set and create paths
        DESCRopts[i][1]['cache_dir'] = '/'.join([GLOBopts['tmp_dir'], DESCRopts[i][1]['cache_dir']])
        assert_dir(DESCRopts[i][1]['cache_dir'])
    
    NBNNopts = ('',dict(cfg.items("NBNN")))
    # Set datatypes
    if 'target_precision' in NBNNopts:
        NBNNopts[1]['target_precision'] = float(NBNNopts[1]['target_precision'])
    if 'checks' in NBNNopts:
        NBNNopts[1]['checks'] = int(NBNNopts[1]['checks'])
    if 'log_level' in NBNNopts:
        NBNNopts[1]['log_level'] = int(NBNNopts[1]['log_level'])
    if NBNNopts[1]['behmo'] == "True":
        NBNNopts[0] = 'behmo'
    else:
        NBNNopts[0] = 'boiman'
    del NBNNopts[1]['behmo']
    
    TESTopts = dict(cfg.items("TEST"))
    # Set datatypes
    if 'batch_size' in TESTopts:
        TESTopts['batch_size'] = int(TESTopts['batch_size'])
    if 'train_size' in TESTopts:
        TESTopts['train_size'] = int(TESTopts['train_size'])
    if 'test_size' in TESTopts:
        TESTopts['test_size'] = int(TESTopts['test_size'])
    if 'keep_descriptors' in TESTopts:
        TESTopts['keep_descriptors'] = TESTopts['keep_descriptors'] == 'True'
    #  Set Paths
    TESTopts['img_pickle_path'] = '/'.join([tmpdir, TESTopts['img_pickle_path']])
    assert_dir('/'.join(TESTopts['img_pickle_path'].split('/')[:-1]))
    
    # Add the infofile for the test as variable
    TESTopts['infofile'] = '/'.join([tmpdir,'testinfo.txt'])
    
    if "DETECTION" in cfg.items:
        DETopts = ('',dict(cfg.items("DETECTION")))
        DETopts[0] = DETopts[1]['method']
        # Set datatypes
        if 'theta_m' in DETopts[1]:
            DETopts[1]['theta_m'] = float(DETopts[1]['theta_m'])
        if 'theta_p' in DETopts[1]:
            DETopts[1]['theta_p'] = float(DETopts[1]['theta_p'])
        
        # Set paths
        DETopts[1]['exemplar_path'] = '/'.join([tmpdir, DETopts[1]['exemplar_path']])
        DETopts[1]['distances_path'] = '/'.join([tmpdir, DETopts[1]['distances_path']])
        DETopts[1]['hypotheses_path'] = '/'.join([tmpdir, DETopts[1]['hypotheses_path']])
        exemplar_dir = '/'.join(DETopts[1]['exemplar_path'].split('/')[:-1])
        distances_dir = '/'.join(DETopts[1]['distances_path'].split('/')[:-1])
        hypotheses_dir = '/'.join(DETopts[1]['hypotheses_path'].split('/')[:-1])
        assert_dir(exemplar_dir)
        assert_dir(distances_dir)
        assert_dir(hypotheses_dir)
        
        return GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts
    else:
        return GLOBopts, DESCRopts, NBNNopts, TESTopts, None

def init_descriptor(DESCRopts):
    if DESCRopts[0] == 'DescriptorUint8':
        return descriptor.DescriptorUint8(**DESCRopts[1])
    else:
        raise Exception("Descriptor type '%s' unknown, check your cfg file"%DESCRopts[0])

def init_estimator(path, NBNNopts):
    if NBNNopts[0] == 'behmo':
        return nbnn.OptNBNNEstimator(path, **NBNNopts[1])
    elif NBNNopts[0] == 'boiman':
        return nbnn.NBNNEstimator(path, **NBNNopts[1])
    else:
        raise Exception("Unknown estimator type '%s', check your cfg file"%NBNNopts[0])

def init_log(log_path, cls, mode='a'):
    # print "log_path: %s, log_file: %s"%(log_path, log_path%cls)
    # print "mode:", mode
    # Setup a config file
    subprocess.call(["./setlog.sh", log_path%cls, mode])
    
    # Setup logger
    logging.config.fileConfig(log_path%(cls)+'.cfg', \
        disable_existing_loggers=False)
    log = logging.getLogger('')
    # Remove config file
    os.remove(log_path%(cls)+'.cfg')
    f = MemuseFilter()
    log.handlers[0].addFilter(f)
    return log


class MemuseFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.

    Rather than use actual contextual information, we just use random
    data in this demo.
    """
    _proc_status = '/proc/%d/status' % os.getpid()

    _scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
              'KB': 1024.0, 'MB': 1024.0*1024.0}


    def filter(self, record):

        record.memuse = self.str_mem()
        return True

    def _VmB(self,VmKey):
        '''Private.
        '''
         # get pseudo file  /proc/<pid>/status
        try:
            t = open(self._proc_status)
            v = t.read()
            t.close()
        except:
            return 0.0  # non-Linux?
         # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
        i = v.index(VmKey)
        v = v[i:].split(None, 3)  # whitespace
        if len(v) < 3:
            return 0.0  # invalid format?
         # convert Vm value to bytes
        return ''.join(v[1:3])

    def memory(self):
        '''Return memory usage in bytes.
        '''
        return self._VmB('VmSize:')

    def resident(self):
        '''Return resident memory usage in bytes.
        '''
        return self._VmB('VmRSS:')

    def stacksize(self):
        '''Return stack size in bytes.
        '''
        return self._VmB('VmStk:')

    def swapsize(self):
        '''Return swap size in bytes.
        '''
        return self._VmB('VmSwap:')

    def byte_to_mb(self,byte):
        return byte/(1024*1024)

    def str_mem(self):
        return "Tot:%s,Swap:%s"%(self.memory(),self.swapsize() )

if __name__ == '__main__':
    import doctest
    import numpy as np
    doctest.testmod()