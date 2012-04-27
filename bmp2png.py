import os
from subprocess import call

if __name__ == '__main__':
    dirs = ['../im/graz01png/bikes','../im/graz01png/bikes_and_persons', '../im/graz01png/persons','../im/graz01png/no_bike_no_person']
    for dirr in dirs:
        fl = os.listdir(dirr)
        for f in fl:
            if not f[0] == '.':
                d = f.split('.')
                out = dirr+'/'+d[0]+'.'+'png'
                fpath = dirr+'/'+f
                if call(['convert', fpath, out]) == 1:
                    print 'Could not convert ', fpath, ', somethings wrong'
                else:
                    print f, 'is converted to', out, 'and removed'
                    os.remove(fpath)

