import logging, logging.config
import os, os.path, subprocess, cPickle
from ConfigParser import RawConfigParser
from nbnn.vocimage import *
from nbnn.voc import VOC
from cal import *

def getopts(configfile, tmpdir):
    cfg = RawConfigParser()
    cfg.read(configfile)
    DESCRopts = dict(cfg.items("DESCRIPTOR"))
    NBNNopts = dict(cfg.items("NBNN"))
    TESTopts = dict(cfg.items("TEST"))

    DESCRopts['cache_dir'] = '/'.join([tmpdir, DESCRopts['cache_dir']])
    NBNNopts['nbnn_dir'] = '/'.join([tmpdir, NBNNopts['nbnn_dir']])
    
    TESTopts['descriptor_path'] = '/'.join([tmpdir, TESTopts['descriptor_path']])
    TESTopts['img_pickle_path'] = '/'.join([tmpdir, TESTopts['img_pickle_path']])
    # Add the infofile for the test as variable
    TESTopts['infofile'] = '/'.join([tmpdir,'testinfo.txt'])
    
    # Set the datatypes of some variables
    if 'target_precision' in NBNNopts:
        NBNNopts['target_precision'] = float(NBNNopts['target_precision'])
    if 'checks' in NBNNopts:
        NBNNopts['checks'] = int(NBNNopts['checks'])
    if 'batch_size' in TESTopts:
        TESTopts['batch_size'] = int(TESTopts['batch_size'])
    if 'train_size' in TESTopts:
        TESTopts['train_size'] = int(TESTopts['train_size'])
    if 'test_size' in TESTopts:
        TESTopts['test_size'] = int(TESTopts['test_size'])
    if 'keep_descriptors' in TESTopts:
        TESTopts['keep_descriptors'] = TESTopts['keep_descriptors'] == 'True'
    if 'log_level' in NBNNopts:
        NBNNopts['log_level'] = int(NBNNopts['log_level'])
    if 'outputformat' in DESCRopts:
        DESCRopts['outputFormat'] = DESCRopts['outputformat']
        del DESCRopts['outputformat']

    # Make sure some folders exist
    if not os.path.exists(DESCRopts['cache_dir']):
        os.mkdir(DESCRopts['cache_dir'])
    res_folder = '/'.join(TESTopts['result_path'].split('/')[:-1])
    TESTopts['res_folder'] = res_folder
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    log_folder = '/'.join(TESTopts['log_path'].split('/')[:-1])
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    
    return DESCRopts, NBNNopts, TESTopts

def get_detection_opts(configfile, tmpdir):
    DESCRopts, NBNNopts, TESTopts = getopts(configfile, tmpdir)
    cfg = RawConfigParser()
    cfg.read(configfile)
    DETECTIONopts = dict(cfg.items("DETECTION"))
    
    if 'theta_m' in DETECTIONopts:
        DETECTIONopts['theta_m'] = float(DETECTIONopts['theta_m'])
    if 'theta_p' in DETECTIONopts:
        DETECTIONopts['theta_p'] = float(DETECTIONopts['theta_p'])
    
    DETECTIONopts['exemplar_path'] = '/'.join([tmpdir, DETECTIONopts['exemplar_path']])

    exemplar_dir = '/'.join(DETECTIONopts['exemplar_path'].split('/')[:-1])
    if not os.path.exists(exemplar_dir):
        os.mkdir(exemplar_dir)
    DETECTIONopts['hypotheses_path'] = '/'.join([tmpdir, DETECTIONopts['hypotheses_path']])
    hyp_dir = '/'.join(DETECTIONopts['hypotheses_path'].split('/')[:-1])
    if not os.path.exists(hyp_dir):
        os.mkdir(hyp_dir)
    
    return DESCRopts, NBNNopts, TESTopts, DETECTIONopts
    

def get_confidence_values(distances):
    cv = []
    for d in distances:
        no_descriptors = d.shape[0]
        # Sum or Sum of squared?
        cv.append(-d.sum()/no_descriptors)
    return cv

def save_testinfo(filename, batches, classes):
    with open(filename,'w') as testfile:
        # Save a file with information on how many iterations with how many
        # classes, and which ones they are, for the multiple processes that are
        # gonna run the tests
        testfile.write("%d\n"%len(batches))
        testfile.write("%d\n"%len(classes))
        for cls in classes:
            testfile.write("%s\n"%cls)

def save_results_to_file(file, objects, confidence_values):
    log = logging.getLogger("__name__")
    if isinstance(objects[0], VOCImage) or isinstance(objects[0], CalImage):
        log.info('Saving image classification')
        with open(file, 'a') as f:
            for obj, cv in zip(objects,confidence_values):
                f.write('%s %f\n'%(obj.im_id, cv))
    elif isinstance(objects[0], Object):
        log.info('Saving image detection files (by bbox)')
        with open(file, 'a') as f:
            for obj, cv in zip(objects,confidence_values):
                f.write('%s %f %d %d %d %d\n'%(obj.image.im_id, cv, \
                    obj.xmin, obj.ymin, obj.xmax, obj.ymax))
                
    log.info("Saved results to %s",file)

def save_to_pickle(filename,data):
    if not os.path.exists(filename):
        with open(filename,'wb') as pklfile:
            cPickle.dump(data, pklfile)
    else:
        with open(filename,'rb') as pklfile:
            data_old = cPickle.load(pklfile)
        data_old.extend(data)
        with open(filename, 'wb') as pklfile:
            cPickle.dump(data_old, pklfile)


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
