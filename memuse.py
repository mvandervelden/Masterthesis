import os,logging

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
        return float(v[1]) * self._scale[v[2]]

    def memory(self,since=0.0):
        '''Return memory usage in bytes.
        '''
        return self._VmB('VmSize:') - since

    def resident(self,since=0.0):
        '''Return resident memory usage in bytes.
        '''
        return self._VmB('VmRSS:') - since

    def stacksize(self,since=0.0):
        '''Return stack size in bytes.
        '''
        return self._VmB('VmStk:') - since

    def swapsize(self,since=0.0):
        '''Return swap size in bytes.
        '''
        return self._VmB('VmSwap:') - since

    def byte_to_mb(self,byte):
        return byte/(1024*1024)

    def str_mem(self):
        return "Tot:%.0fM,Swap:%.0fM"%(\
            self.byte_to_mb(self.memory()),self.byte_to_mb(self.swapsize()) )

if __name__ == "__main__":
    from random import choice
    import logging.config
    levels = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
    
    # logging.basicConfig(level=logging.DEBUG,
    #                     format='%(asctime)-15s %(name)-5s %(levelname)-8s Mem:%(memuse)s %(message)s')
    
    logging.config.fileConfig('logging.conf')
    # create logger
    a1 = logging.getLogger('')
    f = MemuseFilter()
    a1.handlers[0].addFilter(f)
    for x in range(10):
        lvl = choice(levels)
        lvlname = logging.getLevelName(lvl)
        a1.log(lvl, 'A message at %s level with %d %s', lvlname, 2, 'parameters')