import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from pylab import *
import cPickle, sys
from nbnn import vocimage


def point_in_bb(x,y, bb):
    if x >= bb[0]-1 and x <= bb[2]-1 and y >= bb[1]-1 and y <= bb[3]-1:
        return True
    else:
        return False

if __name__ == '__main__':
    im_id = sys.argv[1]
    cls = sys.argv[2]
    folder = sys.argv[3]
    if len(sys.argv) > 4:
        make_heatmap = True
    else:
        make_heatmap = False

    impath = 'VOCdevkit/VOC2012/JPEGImages/%s.jpg'
    annotation_file = 'VOCdevkit/VOC2012/Annotations/%s.xml'%im_id
    vimage = vocimage.VOCImage(impath%im_id, im_id,annotation_file)

    with open(folder+'/cls_%s_imid_%s.pkl'%(cls,im_id),'rb') as f:
        hypotheses = cPickle.load(f)
        fg_points = cPickle.load(f)
        im_exemplars = cPickle.load(f)
    
    n = fg_points.shape[0]
    print "showing image %s, with file %s (%d points)"%(im_id, impath%im_id, n) 
    print "no of hypotheses: %s, no of fg_points: %s, no of im_exemplars: %s"%(hypotheses.shape, fg_points.shape, im_exemplars.shape)
    
    # Sort rows of hypotheses, and points:
    no_hyps = 10
    idxs = hypotheses[:,0].argsort()
    sort_hyp = hypotheses[idxs[-no_hyps:]]
    sort_points = fg_points[idxs[-no_hyps:]]
    
    imf = Image.open(vimage.path)

    imarr = np.asarray(imf)
    xsize = vimage.width
    plt.imshow(imarr)
    plt.title('BEST HYPOTHESES im: %s, class %11s'%(im_id, cls))
    legend = []
    colors = ['','r','b','c','m','y','k','violet','grey','orange', 'salmon']
    for obj in vimage.objects:
        if obj.class_name == cls:
            rect = Rectangle((obj.xmin-1,obj.ymin-1), obj.xmax-obj.xmin, obj.ymax-obj.ymin, \
                facecolor='none', edgecolor='g', linewidth=3 , label='Ground Truth BB')
            gca().add_patch(rect)
            legend.append('Ground Truth BB')

    for i in xrange(1, sort_hyp.shape[0]+1):#hypotheses.shape[0]):
        bb = sort_hyp[-i,1:]
        pt = sort_points[-i,:2]
        rect = Rectangle((bb[0]-1,bb[1]-1),bb[2]-bb[0], bb[3]-bb[1], facecolor='none', edgecolor=colors[i], label='BB %d: Qh=%.2f'%(i,sort_hyp[-i,0]))
        plt.scatter(pt[0], pt[1],s=20,color=colors[i])
        gca().add_patch(rect)
        legend.append('BB %d: Qh=%.2f'%(i,sort_hyp[-i,0]))
    # plt.legend(legend)
    plt.legend( bbox_to_anchor=(1.1, 0.05),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig(folder+'/'+im_id+'_'+cls+'.png')
    plt.clf()
    
    if make_heatmap:
        # Make a heatmap
        hm = np.zeros([vimage.height, vimage.width],np.int)
        for h in xrange(1,11):#hypotheses.shape[0]):
            bb = hypotheses[idxs[-h],1:]
            print 'hyp: ', h, ': ', bb
            for i in xrange(hm.shape[0]):
                for j in xrange(hm.shape[1]):
                    if point_in_bb(j, i, bb):
                        hm[i,j] += 1
        X,Y = meshgrid(range(vimage.width), range(vimage.height))
        plt.imshow(imarr)
        plt.scatter(X,Y,s=1,c=hm, marker=',', alpha=0.5,edgecolor='none')
        plt.title('HYPOTHESES HEATMAP im: %s, class %11s'%(im_id, cls))
        plt.colorbar()
        plt.savefig(folder+'/'+im_id+'_'+cls+'_heatmap.png')
    
    
