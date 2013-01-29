from multiprocessing import Pool
from time import sleep
from logutils import *

LOGPATH = '%s.logtmp'

def f(a):
    (x,i) = a
    log = init_log(LOGPATH, i, 'w')
    log.info('RUNNING F FOR THREAD %s',x)
    with open(i+'.tmp','w') as ff:
        ff.write('bla: "%d: %s"\n'%(x,i))
    sleep(x*2)

def cb(i):
    print i

if __name__ == "__main__":
    
    log = init_log(LOGPATH, 'main', 'w')
    log.info('START MAIN')
    a = enumerate(['a','b','c','d','e','f','g'])
    pool = Pool(processes = 4)
    pool.map(f, a)
    log.info('FINISH MAIN')

