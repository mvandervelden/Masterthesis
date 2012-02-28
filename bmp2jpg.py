import os
from subprocess import call

if __name__ == '__main__':
    dirn = 'im/graz01/bikes_and_persons'
    fl = os.listdir(dirn)
    for f in fl:
        if not f[0] == '.':
            d = f.split('.')
            out = dirn+'/'+d[0]+'.'+'jpg'
            fpath = dirn+'/'+f
            if call(['cjpeg', '-outfile', out, fpath]) == 1:
                print 'Could not convert ', fpath, ', somethings wrong'
            else:
                print f, 'is converted to', out, 'and removed'
                os.remove(fpath)

