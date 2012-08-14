import cPickle, sys

if __name__ == "__main__":
    
    inputf = sys.argv[1]
    outputf = sys.argv[2]
    
    with open(inputf, 'rb') as f:
        detections = cPickle.load(f)
    
    # rank all detections found by sorting and enumerating (it means it sorts first by QD, then by QH. lowest values get first (lower=less confidence, get lower ranks)
    detections.sort()
    
    with open(outputf, 'w') as f:
        for num, det in enumerate(detections):
            f.write('%s %f %f %f %f %f\n'%(det[2],num,det[3], det[4], det[5], det[6]))
        
    
