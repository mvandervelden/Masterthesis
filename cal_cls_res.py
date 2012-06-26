import sys
from cal import *
from utils import *

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("argument expected: cfgfile, tmpfolder")
    
    CALopts = Caltech.fromConfig(sys.argv[1])
    DESCRopts, NBNNopts, TESTopts = getopts(sys.argv[1], sys.argv[2])
    
    
    imdict = dict()
    for cls in CALopts.classes:
        with open(TESTopts['result_path']%cls, 'r') as rf:
            for line in rf:
                elems = line.split(' ')
                if not elems[0] in imdict:
                    imdict[elems[0]] = ('', float('inf'))
                if float(elems[1]) < imdict[elems[0]][1]:
                    imdict[elems[0]] = (cls, float(elems[1]))
    with open(TESTopts['result_path']%'classification_by_image', 'w') as rf:
        for im,(cls, dist) in imdict.items():
            rf.write('%s %s %f\n'%(im, cls, dist))
    
    confmat = [[0 for i in CALopts.classes] for i in CALopts.classes]
    for im,(cls, dist) in imdict.items():
        gt = im.split('__')[0]
        gt_idx = CALopts.classes.index(gt)
        cl_idx = CALopts.classes.index(cls)
        confmat[gt_idx][cl_idx] += 1
    
    with open(TESTopts['result_path']%'confmat', 'w') as rf:
        for row in confmat:
            rf.write('%s\n'%(' '.join(['%d'%e for e in row])))
    print "Finished"