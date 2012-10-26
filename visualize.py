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
    if metric.__name__ in ['dist_qh']:
        dists = -dists
        label = '-'+ metric.__name__
    else:
        label = metric.__name__
    plt.hexbin(points[:,0], points[:,1],C=dists,alpha=.2,gridsize=imarr.shape[0]/10)
    cb = plt.colorbar()
    cb.set_label(label)
    plt.title("Im %s, cls %s, Descriptor %s distances "%(vimage.im_id, cls, metric.__name__))
    plt.savefig(res_path)
    plt.clf()

def visualize_detections(vimage, cls, metric, det_n, detsrc_file, DETopts, res_path):
    print "Performing visualize_detection with image: %s, class: %s, metric: %s, det_n: %d, res_path: %s"%(vimage.im_id, cls, metric.__name__, det_n, res_path)
    distances, points, image, nn_exemplar_indexes = load_distances(DETopts['distances_path']%(im_id,cls))
    detections, reflist = load_detections(detsrc_file, vimage.im_id)
    detection_vals = get_detection_values(detections, reflist, distances, points, metric)
    print "det shape=%s, det_vals shape=%s"%(detections, detection_vals)
    ranking = sort_values(detection_vals)
    print "ranking shape:", ranking
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
    print 'totald:', total_d, 'ranking.shp:',ranking.shape, 'dvals.shape:', detection_vals.shape, \
        'colors.shp:', len(colors)
    
    if metric.__name__ in ['det_qh', 'det_becker', 'det_qd', 'det_bg']:
        # flip ranking to make it ranked correctly
        ranking = np.flipud(ranking)
    for d in xrange(min(total_d, det_n)):
        print 'd:', d
        det = detections[ranking[d],:]
        val = detection_vals[ranking[d]]
        print 'Detection: %s, cf: %s'%(det, val)
        if metric.__name__ == 'det_becker':
            label = 'DET %d: conf (%s): [%.0f, %.2f]'%(d,metric.__name__, val[0], val[1])
        else:
            label = 'DET %d: conf (%s): %.2f'%(d,metric.__name__, val )
        rect = Rectangle((det[0]-1,det[1]-1),det[2]-det[0], det[3]-det[1], \
            facecolor='none', edgecolor=colors[d], \
            label=label)
        gca().add_patch(rect)
    # plt.legend(legend)
    plt.legend( bbox_to_anchor=(1.1, 0.05), ncol=3, fancybox=True)
    plt.savefig(res_path)
    plt.clf()
    
    
def visualize_hypotheses_heatmap(vimage, cls, metric, DETopts, res_path):
    print "Performing visualize_hyp_heatmap with image: %s, class: %s, metric: %s, res_path: %s"%(vimage.im_id, cls, metric.__name__, res_path)
    distances, points, image, nn_exemplar_indexes = load_distances(DETopts['distances_path']%(im_id,cls))
    nn_exemplar_indexes = nn_exemplar_indexes[:,0]
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
    
    if metric.__name__ in ['bb_energy', 'bb_fg']:
        hm = -hm
        label = '-'+metric.__name__
    else:
        label = metric.__name__
    
    plt.imshow(imarr)
    plt.scatter(X,Y,s=1,c=hm, marker=',', alpha=0.5,edgecolor='none')
    plt.title('Im %s, cls %s, Hypotheses %s heatmap'%(im_id, cls, metric.__name__))
    cb = plt.colorbar()
    cb.set_label(label)
    plt.savefig(res_path)
    
def visualize_hypotheses_top(vimage, cls, metric, hyp_n, DETopts, res_path):
    print "Performing visualize_hyp_top with image: %s, class: %s, metric: %s, hyp_n: %d res_path: %s"%(vimage.im_id, cls, metric.__name__, hyp_n, res_path)
    distances, points, image, nn_exemplar_indexes = load_distances(DETopts['distances_path']%(im_id,cls))
    nn_exemplar_indexes = nn_exemplar_indexes[:, 0]
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
    print 'totalh:', total_h, 'ranking.shp:',ranking.shape, 'hvals.shape:', hyp_values.shape, \
        'colors.shp:', len(colors)
    if metric.__name__ in ['bb_qh', 'bb_bg']:
        # flip ranking to make it ranked correctly
        ranking = np.flipud(ranking)
    for h in xrange(min(total_h, hyp_n)):
        print 'h:', h
        bb = hypotheses[ranking[h],:]
        pt = points[ranking[h],:2]
        label = 'BB %d: %s=%.2f'%(h,metric.__name__, hyp_values[ranking[h]])
        rect = Rectangle((bb[0]-1,bb[1]-1),bb[2]-bb[0], bb[3]-bb[1], \
            facecolor='none', edgecolor=colors[h], label=label)
        plt.scatter(pt[0], pt[1],s=20,color=colors[h])
        gca().add_patch(rect)
    plt.legend( bbox_to_anchor=(1.1, 0.05), ncol=3, fancybox=True)
    plt.savefig(res_path)


def run_visualize(method, im_id, cls, cfgfile, options):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    log = logging.getLogger(__name__)
    
    VOCopts = VOC.fromConfig(cfgfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(cfgfile)
    if 'setmode' in GLOBopts and GLOBopts['setmode'] == 'becker':
        VOCopts.image_path = VOCopts.image_path[:-4]+'.png'
    
    im_filename = VOCopts.image_path%im_id
    annotation_file = VOCopts.annotation_path%im_id
    vimage = vocimage.VOCImage(im_filename, im_id, annotation_file)
    
    
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

if __name__ == '__main__':
    usage = """ Usage: python visualize method im_id cls configfile [options]
        method = [distance | detections | hypotheses]
        options [ distance: [fg | bg | qh] ;
                  detections: n(-1...x) [becker | qh | fg | bg | energy]
                  hypotheses: [ heat: [uniform | qh | descrqh | fg | bg | energy]
                                top: n(-1...x) [ qh | descrqh | fg | bg | energy]
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
    
    if len(sys.argv) > 5:
        options = sys.argv[5:]
    else:
        options = None
    
    run_visualize(method, im_id, cls, cfgfile, options)

        
