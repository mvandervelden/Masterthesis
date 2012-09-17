import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pylab import *
import sys
from nbnn import vocimage
from utils import *
from metric_functions import *
from file_io import *
from detection_utils import *
import logging


def visualize_distances(vimage, cls, metric, DETopts, res_path):
    print "Performing visualize_distance with image: %s, class: %s, metric: %s, res_path: %s"%(vimage.im_id, cls, metric.__name__, res_path)
    distances, points, image, nn_exemplar_indexes = load_distances(DETopts['distances_path']%(im_id,cls))
    
    imarr = load_imarray(vimage.path)
    
    plt.imshow(imarr)
    dists = metric(distances)
    plt.hexbin(points[:,0], points[:,1],C=dists,alpha=.2,gridsize=imarr.shape[0]/10)
    cb = plt.colorbar()
    cb.set_label(metric.__name__)
    plt.title("Im %s, cls %s, Descriptor %s distances "%(vimage.im_id, cls, metric.__name__))
    plt.savefig(res_path)
    plt.clf()

def visualize_detections(vimage, cls, metric, det_n, detsrc_path, DETopts, res_path):
    print "Performing visualize_detection with image: %s, class: %s, metric: %s, det_n: %d, res_path: %s"%(vimage.im_id, cls, metric.__name__, det_n, res_path)
    distances, points, image, nn_exemplar_indexes = load_distances(DETopts['distances_path']%(im_id,cls))
    detections, reflist = load_detections(detsrc_path, cls, vimage.im_id)
    detection_vals = get_detection_values(detections, reflist, distances, points, metric)
    ranking = sort_values(detection_values)

    imarr = load_imarray(vimage.path)
    plt.imshow(imarr)
    
    plt.title('Im %s, cls %s, Detections %s top %d'%(im_id, cls, metric.__name__, det_n))

    colors = ['r','b','c','m','y','k','violet','grey','orange', 'salmon']
    for obj in vimage.objects:
        if obj.class_name == cls:
            rect = Rectangle((obj.xmin-1,obj.ymin-1), obj.xmax-obj.xmin, obj.ymax-obj.ymin, \
                facecolor='none', edgecolor='g', linewidth=3 , label='Ground Truth BB')
            gca().add_patch(rect)
    total_d = detections.shape[0]
    for d in xrange(total_d-1, total_d-1-det_n, -1):
        det = detection[ranking[d],:]
        val = detection_vals[ranking[d]]
        print 'Detection: %s, cf: %f'%(det, val)
        rect = Rectangle((det[0]-1,det[1]-1),det[2]-det[0], det[3]-det[1], \
            facecolor='none', edgecolor=colors[i], \
            label='DET %d: conf (%s): %.2f'%(d,metric.__name__, val))
        gca().add_patch(rect)
    # plt.legend(legend)
    plt.legend( bbox_to_anchor=(1.1, 0.05), ncol=3, fancybox=True)
    plt.savefig(res_path)
    plt.clf()
    
    
def visualize_hypotheses_heatmap(vimage, cls, metric, DETopts, res_path):
    print "Performing visualize_hyp_heatmap with image: %s, class: %s, metric: %s, res_path: %s"%(vimage.im_id, cls, metric.__name__, res_path)
    distances, points, image, nn_exemplar_indexes = load_distances(DETopts['distances_path']%(im_id,cls))
    exemplars = load_exemplars(DETopts['exemplar_path']%(cls), nn_exemplar_indexes)
    hypotheses = get_hypotheses(exemplars, points, vimage.width, vimage.height)
    hyp_values = get_hypothesis_values(hypotheses, distances, points, metric)
    ranking = sort_values(hyp_values)
    
    imarr = load_imarray(vimage.path)
    
    # Make a heatmap
    hm = np.zeros([vimage.height, vimage.width])

    print "printing %d hypotheses in heat map. Metric: %s)"%(hypotheses.shape[0],metric.__name__)
    for h in xrange(ranking.shape[0]-1,-1,-1):
        bb = hypotheses[ranking[h],:]
        # Betere implementatie:
        hm[bb[1]-1:bb[3]-1,bb[0]-1:bb[2]-1] += hyp_values[ranking[h]]
    X,Y = meshgrid(range(vimage.width), range(vimage.height))

    plt.imshow(imarr)
    plt.scatter(X,Y,s=1,c=hm, marker=',', alpha=0.5,edgecolor='none')
    plt.title('Im %s, cls %s, Hypotheses %s heatmap'%(im_id, cls, metric.__name__))
    cb = plt.colorbar()
    cb.set_label(metric.__name__)
    plt.savefig(res_path)
    
def visualize_hypotheses_top(vimage, cls, metric, hyp_n, DETopts, res_path):
    print "Performing visualize_hyp_top with image: %s, class: %s, metric: %s, hyp_n: %d res_path: %s"%(vimage.im_id, cls, metric.__name__, hyp_n, res_path)
    distances, points, image, nn_exemplar_indexes = load_distances(DETopts['distances_path']%(im_id,cls))
    exemplars = load_exemplars(DETopts['exemplar_path']%(cls), nn_exemplar_indexes)
    hypotheses = get_hypotheses(exemplars, points, vimage.width, vimage.height)
    hyp_values = get_hypothesis_values(hypotheses, distances, points, metric)
    ranking = sort_values(hyp_values)
    
    imarr = load_imarray(vimage.path)
    
    plt.imshow(imarr)
    plt.title('Im %s, cls %s, Hypotheses %s top %d'%(im_id, cls, metric.__name__, hyp_n))

    for obj in vimage.objects:
        if obj.class_name == cls:
            rect = Rectangle((obj.xmin-1,obj.ymin-1), \
                obj.xmax-obj.xmin, obj.ymax-obj.ymin, \
                facecolor='none', edgecolor='g', linewidth=3 , \
                label='Ground Truth BB')
            gca().add_patch(rect)
    
    colors = ['r','b','c','m','y','k','violet','grey','orange', 'salmon']
    total_h = ranking.shape[0]
    for h in xrange(total_h-1, total_h-1-hyp_n, -1):
        bb = hypotheses[ranking[h],:]
        pt = points[ranking[h],:2]
        label = 'BB %d: %s=%.2f'%(total_h-h,metric.__name__, hyp_values[h])
        rect = Rectangle((bb[0]-1,bb[1]-1),bb[2]-bb[0], bb[3]-bb[1], \
            facecolor='none', edgecolor=colors[total_h - h], label=label)
        plt.scatter(pt[0], pt[1],s=20,color=colors[total_h - h])
        gca().add_patch(rect)
    plt.legend( bbox_to_anchor=(1.1, 0.05), ncol=3, fancybox=True)
    plt.savefig(res_path)




if __name__ == '__main__':
    usage = """ Usage: python visualize method im_id cls configfile [options]
        method = [distance | detections | hypotheses]
        options [ distance: [fg | bg | Qh] ;
                  detections: n(-1...x) [Becker | Qh | fg | Energy]
                  hypotheses: [ heat: [uniform | Qh | fg | Energy]
                                top: n(-1...x) [Qh | fg | Energy]
                              ]
                ]
    """
    try:
        method = sys.argv[1]
        im_id = sys.argv[2]
        cls = sys.argv[3]
        cfgfile = sys.argv[4]
    except IndexError:
        print "Not enough command line arguments:"
        print usage
        exit(1)
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    log = logging.getLogger(__name__)
    
    VOCopts = VOC.fromConfig(cfgfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(cfgfile)
    
    im_filename = VOCopts.image_path%im_id
    annotation_file = VOCopts.annotation_path%im_id
    vimage = vocimage.VOCImage(im_filename, im_id, annotation_file)
    
    if len(sys.argv) > 5:
        options = sys.argv[5:]
    else:
        options = None
    if method == 'distance':
        if not options is None:
            metric = eval('dist_' + options[0])
        else:
            metric = dist_fg
        
        res_dir = GLOBopts['res_dir'] + '/dist_images'
        assert_dir(res_dir)
        visualize_distances(vimage, cls, metric, \
            DETopts[1], \
            res_dir + '/%s_%s_%s.png' % (im_id, cls, metric.__name__))
    elif method == 'detections':
        if not options is None:
            det_n = int(options[0])
            if len(options) > 1:
                metric = eval('det_'+options[1])
            else:
                metric = det_becker
        else:
            det_n = 10
            metric = det_becker
        res_dir = GLOBopts['res_dir'] + '/top%d_det_images' % det_n
        assert_dir(res_dir)
        visualize_detections(vimage, cls, metric, det_n,  \
            GLOBopts['result_path'] % (im_id, cls), \
            DETopts[1], \
            res_dir + '/%s_%s_%s.png' % (im_id, cls, metric.__name__))
    elif method == 'hypotheses':
        if not options is None:
            h_vis = options[0]
            if h_vis == 'heat':
                if len(options) > 1:
                    metric = eval('bb_' + options[1])
                else:
                    metric = bb_qh
                res_dir = GLOBopts['res_dir'] + '/hyp_heatmaps'
                assert_dir(res_dir)
                visualize_hypotheses_heatmap(vimage, cls, metric, \
                    DETopts[1], \
                    res_dir + '/%s_%s_%s.png'%(im_id, cls, metric.__name__))
            elif h_vis == 'top':
                if len(options) > 1:
                    hyp_n = int(options[1])
                else:
                    hyp_n = 10
                if len(options) > 2:
                    metric = eval('bb_' + options[2])
                else:
                    metric = bb_qh
                res_dir = GLOBopts['res_dir'] + '/top%d_hyp_images' % hyp_n
                assert_dir(res_dir)
                visualize_hypotheses_top(vimage, cls, metric, hyp_n,  \
            DETopts[1], \
            res_dir + '/%s_%s_%s.png' % (im_id, cls, metric.__name__))
            else:
                raise Exception("Wrong Parameters")
        else:
            raise Exception("Not enough parameters")
    else:
        raise Exception("Unknown mode")
        
