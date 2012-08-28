import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from pylab import *
import cPickle, sys
from nbnn import vocimage

if __name__ == '__main__':
    im_id = sys.argv[1]
    cls = sys.argv[2]
    folder = sys.argv[3]

    impath = 'VOCdevkit/VOC2012/JPEGImages/%s.jpg'
    annotation_file = 'VOCdevkit/VOC2012/Annotations/%s.xml'%im_id
    vimage = vocimage.VOCImage(impath%im_id, im_id,annotation_file)

    with open(folder+'/comp3_det_val_%s.txt'%cls, 'r') as f:
        content = f.read()
    imlines = [line for line in content.split('\n') if im_id in line]
    if len(imlines) <1:
        print 'No detections available, exiting'
        exit()
    bbs = [line.split(' ')[1:] for line in imlines]
    n = len(bbs)
    print "showing image %s, with file %s (%d detections)"%(im_id, impath%im_id, n) 
    
    
    imf = Image.open(vimage.path)

    imarr = np.asarray(imf)
    xsize = vimage.width
    plt.imshow(imarr)
    plt.title('DETECTIONS: im: %s, class %11s'%(im_id, cls))
    legend = []
    colors = ['r','b','c','m','y','k','violet','grey','orange', 'salmon']
    for obj in vimage.objects:
        if obj.class_name == cls:
            rect = Rectangle((obj.xmin-1,obj.ymin-1), obj.xmax-obj.xmin, obj.ymax-obj.ymin, \
                facecolor='none', edgecolor='g', linewidth=3 , label='Ground Truth BB')
            gca().add_patch(rect)

    for i, bbb in enumerate(bbs):
        print bbb
        bb = [float(s) for s in bbb[1:]]
        conf = int(float(bbb[0]))
        print conf
        rect = Rectangle((bb[0]-1,bb[1]-1),bb[2]-bb[0], bb[3]-bb[1], facecolor='none', edgecolor=colors[i], label='DET %d: conf: %d'%(i,conf))
        gca().add_patch(rect)
    # plt.legend(legend)
    plt.legend( bbox_to_anchor=(1.1, 0.05),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig(folder+'/'+im_id+'_'+cls+'_detections.png')
    plt.clf()
    
