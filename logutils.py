def print_log_record_on_error(func):
    def wrap(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except:
            import sys
            print >>sys.stderr, "Unable to create log message msg=%r, args=%r " % (
                getattr(self, 'msg', '?'), getattr(self, 'args', '?'))
            raise
    return wrap

import logging
logging.LogRecord.getMessage = print_log_record_on_error(logging.LogRecord.getMessage)


import logging.config
import os, os.path, subprocess
import re

def init_log(log_path, cls, mode='a'):
    # print "log_path: %s, log_file: %s"%(log_path, log_path%cls)
    # print "mode:", mode
    # Setup a config file
    
    nlog = log_path%cls
    nlogcfg = nlog+'.cfg'
    with open('blank.log.cfg', 'r') as f:
        blog = f.read()
    blog2 = re.sub("'blank.log','a'", "'%s','%s'"%(log_path%cls, mode), blog)
    with open(nlogcfg, 'w') as f:
        f.write(blog2)
    
    # Setup logger
    logging.config.fileConfig(nlogcfg, \
        disable_existing_loggers=False)
    log = logging.getLogger('')
    # Remove config file
    os.remove(nlogcfg)
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
