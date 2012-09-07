import cPickle, sys
import numpy as np

if __name__ == "__main__":
    
    inputf = sys.argv[1]
    outputf = sys.argv[2]
    
    with open(inputf, 'rb') as f:
        detections = cPickle.load(f)
        Qds = cPickle.load(f)
        Qhs = cPickle.load(f)
        im_ids = cPickle.load(f)
    
    # rank all detections found by sorting and enumerating (it means it sorts first by QD, then by QH. lowest values get first (lower=less confidence, get lower ranks)
    ranking = np.hstack([Qds, Qhs]).argsort()

    with open(outputf, 'w') as f:
        for num, r in enumerate(ranking):
            f.write('%s %f %f %f %f %f\n'%(im_ids[r],num,det[0], det[1], det[2], det[3]))
        
    
