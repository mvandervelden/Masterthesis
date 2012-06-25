import logging, logging.config
import os, os.path
from nbnn.vocimage import *
from nbnn.voc import VOC

def getopts(configfile, tmpdir):
    VOCopts = VOC.fromConfig(configfile)
    
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
    
    # Make sure some folders exist
    if not os.path.exists(DESCR_opts['cache_dir']):
        os.mkdir(descriptor_dir)
    res_folder = '/'.join(DESCR_opts['results_path'].split('/')[:-1])
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    
    return VOCopts, DESCRopts, NBNNopts, TESTopts

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
    if isinstance(objects[0], VOCimage):
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

def init_log(logconfig):
    # Setup logger
    logging.config.fileConfig(logconfig, \
        disable_existing_loggers=False)
    log = logging.getLogger('')
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

if __name__ == "__main__":
    from random import choice
    import logging.config
    levels = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
    
    # logging.basicConfig(level=logging.DEBUG,
    #                     format='%(asctime)-15s %(name)-5s %(levelname)-8s Mem:%(memuse)s %(message)s')
    
    logging.config.fileConfig('blank.log.cfg')
    # create logger
    a1 = logging.getLogger('')
    f = MemuseFilter()
    a1.handlers[0].addFilter(f)
    for x in range(10):
        lvl = choice(levels)
        lvlname = logging.getLevelName(lvl)
        a1.log(lvl, 'A message at %s level with %d %s', lvlname, 2, 'parameters')
