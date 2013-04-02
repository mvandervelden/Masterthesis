import sys, traceback
import numpy as np
from multiprocessing import Pool
import multiprocessing

from logutils import *
from nbnn import *
from nbnn.voc import *

from utils import *
from procedures import *
from file_io import *
from detection_utils import *
from metric_functions import *
from quickshift import *

""" Multiprocessing error handling """
# Shortcut to multiprocessing's logger
def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable
        return

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result
    pass

""" LOCAL NBNN FUNCTIONS """

def train_local(classes, descriptors_function, estimator, VOCopts, GLOBopts, NBNNopts, TESTopts, DETopts, log):

    DETmode = DETopts[0]
    DETopts = DETopts[1]

    for i, cls in enumerate(classes):
        # Get classes images, descriptors, and add it to the estimator
        log.info('==== GET CLASS %d: %s IMAGES ====', i, cls)
        if cls == 'background':
            set_name = GLOBopts['train_set']
        else:
            set_name = cls+'_'+GLOBopts['train_set']
        img_set = read_image_set(VOCopts,set_name)
        exemplar_path = DETopts['exemplar_path']%cls
        
        if GLOBopts['train_sel'] == 'segment':
            log.info('==== LOADING OBJECT SEGMENTATION MASKS ====')
            load_object_segmentations(VOCopts.object_mask_path, img_set)
        if not GLOBopts['descriptor_path'] is None:
            for im in img_set:
                if os.path.exists(GLOBopts['descriptor_path']%im.im_id):
                    # Make sure not to calculate descriptors again if they already exist
                    im.descriptor_file.append(GLOBopts['descriptor_path']%im.im_id)
        log.info('==== GET %s DESCRIPTORS ====', cls)
        descriptors = get_image_descriptors(img_set, descriptor_function, \
            GLOBopts['descriptor_path'])
                
        if GLOBopts['train_sel'] == 'bbox':
            # Get descriptor's objects (bboxes):
            log.info('==== GET %s OBJECTS ====', cls)
            if not cls == 'background':
                objects = get_objects_by_class(img_set, cls)
                if not exemplar_path is None:
                    cls_descriptors, exemplars = get_bbox_descriptors(objects, \
                            descriptors, exemplars=True)
                    estimator.add_class(cls, cls_descriptors)
                    log.info('==== SAVING EXEMPLARS to %s ====', exemplar_path)
                    save_exemplars(exemplar_path, exemplars)
            else:
                # Add bg descriptors
                bg_descriptors = get_bbox_bg_descriptors(objects, descriptors)
                estimator.add_class(cls, bg_descriptors)
                        
        elif GLOBopts['train_sel'] == 'segment':
            cls_descr = []
            exemplars = []
            for im in img_set:
                object_idxs = list(np.unique(im.object_segmentation))
                log.info('=== Image %s, bb [%d,%d,%d,%d] partitioning ===', \
                        im.im_id, im.objects[0].xmin, im.objects[0].ymin, \
                        im.objects[0].xmax, im.objects[0].ymax)
                log.info(' --- object idxs in image: %s', object_idxs)
                log.info(' --- pts: %s, descr: %s', \
                        descriptors[im.im_id][0].shape, \
                        descriptors[im.im_id][1].shape)
                if not cls == 'background':
                    # Find which objects are cls
                    object_ids = [o.object_id for o in im.objects]
                    cls_obj_ids = [o.object_id for o in im.objects \
                            if o.class_name == cls]
                else:
                    cls_obj_ids = [0]
                log.info(' --- objects in image: %s, cls ids: %s', \
                        [(o.object_id, o.class_name) for o in im.objects], \
                        cls_obj_ids)
                if not cls == 'background':
                    # Get object_segmentation
                    imdescr, impts = partition_descriptors(\
                            np.matrix(descriptors[im.im_id][0]), \
                            descriptors[im.im_id][1], \
                            im.object_segmentation, exemplars=True)
                    log.info(' --- No of seg_objects found: %d (=%d)', \
                            len(imdescr), len(impts))
                    # Index to the right objects
                    im_exemplars = []
                    for object_id in cls_obj_ids:
                        cls_object = im.objects[object_id - 1]
                        no_descr_in_obj = imdescr[object_id].shape[0]

                        log.info(' --- get exemplars for object_id: %s, obj: %s, no_descr: %d',\
                                object_id, (cls_object.object_id, \
                                cls_object.class_name), no_descr_in_obj)
                        exmps = get_exemplars(cls_object, \
                                np.array(impts[object_id]))
                        log.info(' --- adding %s exemplars, %s descr, %s points', \
                                exmps.shape, no_descr_in_obj, \
                                impts[object_id].shape)
                        im_exemplars.append(exmps)
                    exemplars.append(np.vstack(im_exemplars))
                else:
                    # Get background segment descriptors
                    log.info("Adding descriptors of bg-class")
                    # Get object_segmentation
                    imdescr = partition_descriptors(\
                            np.matrix(descriptors[im.im_id][0]), \
                            descriptors[im.im_id][1], \
                            im.object_segmentation, exemplars=False)
                    log.info(' --- No of seg_objects found: %d', \
                            len(imdescr))
                for object_id in cls_obj_ids:
                    log.info(' --- adding descr for object %s', object_id)
                    cls_descr.append(imdescr[object_id])
                    
            log.info('--- Adding %s descriptor arrays to class %s', \
                    len(cls_descr), cls)
            estimator.add_class(cls, cls_descr)
            if not cls == 'background':
                log.info('==== SAVING EXEMPLARS to %s ====', exemplar_path)
                save_exemplars(exemplar_path, exemplars)
        log.info('=== CLS %s (%d/%d) ADDED ===',cls, i, len(classes))


def get_detection_dists((batch_no, cls, batch, configfile)):
    images = batch
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)

    # Setup logger
    log = init_log(GLOBopts['log_path'], 'nn_%s_%d'%(cls, batch_no), 'w')
    
    log.info("NN cfg:%s, batch_no:%d, cls:%s",configfile, batch_no, cls)
        
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = init_descriptor(DESCRopts[1], logger=log)
    
    log.info('==== INIT ESTIMATOR ====')
    estimator = init_estimator(GLOBopts['nbnn_path']%'estimator', NBNNopts, logger=log)
    
    log.info('==== LOAD IMAGE DESCRIPTORS ====')
    # log.debug('  -- NO images: %d, descr_path: %s', len(images), GLOBopts['descriptor_path'])
    descriptors = get_image_descriptors(images, descriptor_function, \
        GLOBopts['descriptor_path'])
    
    # Sort descriptors points & images such that they have the same order...
    descriptors_array, points_list, images, num_descriptors = sort_descriptors(descriptors, images)

    log.info('==== GET ESTIMATES ====')
    
    # Getting fgbg estimates for full image
    log.info("Getting estimates for %s descriptors.", descriptors_array.shape[0])

    # Get distances
    cls_dst, nn_descr_idxs = estimator.get_estimates([cls], descriptors_array, k=TESTopts['k'], return_result=True,logger=log)

    del descriptors_array
    
    log.debug("-- returning array of shape %s",cls_dst.shape)
    log.debug("-- mean estimate: %s",np.mean(cls_dst))
    log.debug("-- max estimate: %s",np.max(cls_dst))
    log.debug("-- min estimate: %s",np.min(cls_dst))
    log.debug("-- min/max descr_idx:%d/%d",nn_descr_idxs.min(),nn_descr_idxs.max())
    log.debug("-- no of descr_indexes %s",nn_descr_idxs.shape)
    
    # Put distances into a list (per image)
    # and put exemplar_indexes in a list too
    distances = []
    nearest_exemplar_indexes = []
    index = 0
    for k in num_descriptors:
        distances.append(cls_dst[index:index+k,:])
        nearest_exemplar_indexes.append(nn_descr_idxs[index:index+k,:])
        index += k
    del cls_dst
    del nn_descr_idxs
    
    log.info('==== SAVE DISTANCES ====')
    save_distances(DETopts[1]['distances_path'], cls, distances, points_list, \
        images, nearest_exemplar_indexes, logger=log)
    log.info('==== NN FINISHED ====')


def get_knn((image, configfile)):
    im_id = image.im_id
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)

    # Setup logger
    log = init_log(GLOBopts['log_path'], 'local_nn_%s'%(im_id), 'w')
    
    log.info("Local NN cfg:%s, im:%s",configfile, im_id)
    
    if GLOBopts['setmode'] == 'voc':
        classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',\
                'chair','cow','diningtable','dog','horse','motorbike','person',\
                'pottedplant','sheep','sofa','train','tvmonitor', 'background']
    elif GLOBopts['setmode'] == 'becker':
        classes = ['motorbike', 'background']
    
    no_classes = len(classes)
    log.info("Local NN cfg:%s, im:%s, no_cls:%d",configfile, im_id, no_classes)
    distlist = []
    ptslist = []
    exemp_idxlist = []
    
    for cls in classes:
        log.info("Handling class: %s", cls)
        distances, allpoints, im, nearest_exemplar_indexes = load_distances( \
                DETopts[1]['distances_path']%(im_id, cls), logger=log)
        log.info('--Adding cls %s: %s dists, %s pts, im: %s, %s ex_idxs', \
                cls, distances.shape, allpoints.shape, im.im_id, \
                nearest_exemplar_indexes.shape)
        if not im.im_id == im_id:
            log.warning("WRONG im_id, wrong distance pkl loaded!!: %s != %s",im_id, im.im_id)
        distlist.append(distances)
        ptslist.append(allpoints)
        exemp_idxlist.append(nearest_exemplar_indexes)
    
    distances = np.hstack(distlist)
    exemplar_indexes = np.hstack(exemp_idxlist)
    log.info('Got %s distance matrix, and %s exemplar_idx', distances.shape, exemplar_indexes.shape)
    # Save points, same over all classes of an image, so ptslist[0]
    save_points(DETopts[1]['knn_path']%(im_id, 'points'), ptslist[0], logger=log)
    
    log.info("Reshape distance & exemplar idxs")
    k = TESTopts['k']
    # n= no of descriptors in test image
    N = distances.shape[0]
    # Make a selection of the k nearest neighbors overall
    
    distances = np.reshape(distances,[N, no_classes*k])
    exemplar_indexes = np.reshape(exemplar_indexes, [N,no_classes*k])
    asort = np.argsort(distances)
    
    cls_idxs = np.array([i for i in range(no_classes) for kkk in range(k)])
    
    cls_dists = [[[] for nn in range(N)] for c in classes]
    cls_bg_dists = [[np.inf for nn in range(N)] for c in classes]
    cls_exempl = [[[] for nn in range(N)] for c in classes]
    
    log.info('Reshaped dist & ex_ind: %s & %s', distances.shape, exemplar_indexes.shape)
    log.info('Argsort shape: %s', asort.shape)
    log.info('cls_idxs shape: %s', cls_idxs.shape)
    log.info('cls_dists (%d lists of %d lists) & cls_exempl (%d lists of %d lists)', \
            len(cls_dists), len(cls_dists[0]), len(cls_exempl), len(cls_exempl[0]))
    log.info('cls_bg_dists (%d lists of %d lists)', len(cls_bg_dists), len(cls_bg_dists[0]))
    
    for i in range(N):
        # Iterate over all descriptors i, and get the k nearest of these
        sort_dists = distances[i,asort[i,:k]]
        # Get their classes
        sort_cls = cls_idxs[asort[i,:k]]
        # Get their exemplar indexes
        sort_exempl = exemplar_indexes[i,asort[i,:k]]
        # Get nearest bg_dist for each class
        for j, d in enumerate(sort_dists):
            for cl in range(no_classes):
                if (not cl == sort_cls[j]) and cls_bg_dists[cl][i] > d:
                    cls_bg_dists[cl][i] = d
        
        # Get near dists & exemplars for each class
        for j, cl in enumerate(sort_cls):
            cls_dists[cl][i].append(sort_dists[j])
            cls_exempl[cl][i].append(sort_exempl[j])
        
    log.info('Built lists of cls_dists and cls_exempl')
    total_dists = 0
    goal = k*N
    for c in range(no_classes):
        cls = classes[c]
        no_cls_dists = sum([len(n) for n in cls_dists[c]])
        no_bg_dists = len(cls_bg_dists[c])
        total_dists += no_cls_dists
        log.info('Saving knn for cls %s, %d dists, %d bg_dists', cls, no_cls_dists, no_bg_dists)
        save_knn(DETopts[1]['knn_path']%(im_id, cls), cls_dists[c], cls_bg_dists[c], cls_exempl[c], logger=log)
        log.info('Saved %d of %d distances', total_dists, goal)
    log.info("===FINISHED kNN SUBDIVISION===")
    # compare nearest neighbors

def detection((image, cls, configfile)):
    im_id = image.im_id
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)

    # Setup logger
    log = init_log(GLOBopts['log_path'], 'detection_%s_%s'%(im_id, cls), 'w')
    
    log.info("DETECTION cfg:%s, cls: %s, im:%s",configfile, cls, im_id)
    
    DETmode = DETopts[0]
    DETopts = DETopts[1]
    log.info('=== LOADING POINTS ===')
    points = load_points(DETopts['knn_path']%(im_id, 'points'),logger=log)
    
    log.info('==== LOADING kNN DISTANCES ====')
    distances, bg_distances, nearest_exemplar_indexes = load_knn(DETopts['knn_path']%(im_id, cls), \
            logger=log)
    
    no_distances = sum([len(d) for d in distances])
    no_bg_distances = len(bg_distances)
    log.debug("Got %d distance_lists, %d bg_distance lists, %d ex_indexes_lists, element of idx_list: %s", len(distances), len(bg_distances), len(nearest_exemplar_indexes), nearest_exemplar_indexes[0])
    log.debug("No of distances: %d, no of bg_dists: %d, no of ex_ix: %d", no_distances, no_bg_distances, sum([len(n) for n in nearest_exemplar_indexes]))
    if no_distances == 0:
        log.debug('No distances found for im %s, cls %s, k=%d. NO DETECTIONS TO BE FOUND', im_id, cls, TESTopts['k'])
        save_detections(GLOBopts['result_path']%(im_id, cls), np.zeros([0,4]), np.zeros([0,2]))
        log.info('==== FINISHED DETECTION ====')
        return
    log.debug('Number of distances found %d > 0', no_distances)
    pointslist = [[points[i,:] for k in n] for i,n in enumerate(nearest_exemplar_indexes)]
    log.debug('Built a pointslist: len = %d, inner list sum: %d', len(pointslist), sum([len(p) for p in pointslist]))
    points = np.vstack([p for ppp in pointslist for p in ppp])
    log.debug('Built a pointsarray: shape: %s', points.shape)
    nearest_exemplar_indexes = np.hstack([e for eee in nearest_exemplar_indexes for e in eee])
    log.debug('Built n_ex_ind array: %s', nearest_exemplar_indexes.shape)
    bg_dist = []
    for i, d in enumerate(distances):
        for j, ddd in enumerate(d):
            bg_dist.append(bg_distances[i])
    bg_distances = np.hstack(bg_dist)
    log.debug('Built bg_distances array: %s', bg_distances.shape)
    distances = np.hstack([d for ddd in distances for d in ddd])
    log.debug('Built distances array: %s', distances.shape)

    distances = np.vstack([distances, bg_distances]).T
    log.debug('Built distances array: %s', distances.shape)
    
    log.info('==== LOADING NEAREST EXEMPLARS ====')
    exemplars = load_exemplars(DETopts['exemplar_path']%cls, nearest_exemplar_indexes, logger=log)
    
    log.info('==== GET HYPOTHESES ====')
    
    hypotheses = get_hypotheses(exemplars, points, image.width, image.height, logger=log)
    if hypotheses.shape[0] == 0:
        log.debug("== FOUND NO HYPOTHESES WITH fg_d < bg_d. No clustering possible!")
    log.info('==== GET HYPOTHESIS VALUES ====')
    hvalues = get_hypothesis_values(hypotheses, distances, points, eval(DETopts['hypothesis_metric']))
    log.debug('HVALS shape: %s', hvalues.shape)
    ranking = sort_values(hvalues)
    log.debug('Ranking shape: %s', ranking.shape)
    # Keep only the best n descriptors (largest relative margin d+, d-)
    if 'hyp_cutoff' in DETopts:
        log.info('Using %s hypotheses, out of %d', DETopts['hyp_cutoff'], hypotheses.shape[0])
        ranking = ranking[-int(DETopts['hyp_cutoff']):]
    hvalues = hvalues[ranking]
    hypotheses = hypotheses[ranking]
    # Make sure points and distances are selected and sorted in the same way, and saved with the detections
    points = points[ranking]
    distances = distances[ranking]
    log.debug(" -- first hyp: (%s, %.2f, last: hyp: (%s, %.2f)", hypotheses[0,:], \
        hvalues[0], hypotheses[-1,:], hvalues[-1])
    
    if DETopts['method'] == 'single_link':
        # get pairwise overlap (don't have to calculate each time)
        if DETopts['dist'] == 'overlap':
            overlap, indexes = get_pairwise_overlap(hypotheses)
        else:
            dist = sc_dist.pdist(hypotheses, DETopts['dist'])
            overlap = 1-(dist/dist.max())
            indexes = make_indexes(hypotheses.shape[0])
            
        log.debug('Mean overlap:%.5f',overlap.mean())
        log.info('  == CLUSTERING HYPOTHESES OF %s==',im_id)
        
        detections, dist_references = single_link_clustering(hypotheses, hvalues, overlap, indexes, ranking, DETopts)
    elif DETopts['method'] == 'quickshift':
        log.debug('qs_tree_path: %s', DETopts['quickshift_tree_path'])
        qs_path = DETopts['quickshift_tree_path']%(cls, im_id)
        log.debug('qs_tree_path: %s', qs_path)
        detections, dist_references = cluster_quickshift(hypotheses, DETopts['tau'], save_tree_path=qs_path)
    log.debug(' Found %d Detections', len(detections))
    # Save detections of image to resultsfiles
    # Save detections only, do not rank yet, because of batches...
    # dist_references: a list of length 'No_Detections', of lists that refer back to the original hypotheses, distances, points
    log.info('==== SAVE CONFIDENCE VALUES ====')
    save_detections(GLOBopts['result_path']%(im_id, cls), np.vstack(detections), dist_references, descr_distances=distances, descr_points=points)
    log.info('==== FINISHED DETECTION ====')
    

def rank_detections((cls, configfile)):
    
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)
    DETmethod = DETopts[0]
    DETopts = DETopts[1]
    # Setup logger
    log = init_log(GLOBopts['log_path'], 'ranking_%s'%cls, 'w')
    
    log.info("Making VOC results files cfg:%s, cls:%s",configfile, cls)
    
    vimages = read_image_set(VOCopts, GLOBopts['test_set'])
    log.info('Ranking %s images. ',len(vimages))
    
    det_metrics = DETopts['detection_metric']
    if 'qs_density' in det_metrics and not DETmethod == 'quickshift':
        log.debug("Skipping qs_density, because not clustered with quickshift")
        det_metrics.remove('qs_density')
    
    ranking_path = DETopts['ranking_path'] 
    for det_metric in det_metrics:
        log.debug("Ranking for metric: %s, path: %s", det_metric, ranking_path)
        outputf = ranking_path%(det_metric, GLOBopts['test_set'], cls)
        if 'hyp_' in det_metric:
            hyp = True
            metric = det_metric[4:]
        else:
            hyp = False
            metric = det_metric
        all_detections = []
        all_det_vals = []
        all_det_imids = []
        
        for vimage in vimages:
            im_id = vimage.im_id
            log.info('Parsing image %s detections...', im_id)
            detfile = GLOBopts['result_path']%(im_id, cls)
            
            if hyp:
                # hypothesis based ranking method
                log.setLevel(logging.WARNING)
                image = vimage
                log.info('=== LOADING POINTS ===')
                points = load_points(DETopts['knn_path']%(im_id, 'points'),logger=log)
                log.info('==== LOADING kNN DISTANCES ====')
                distances, bg_distances, nearest_exemplar_indexes = load_knn(DETopts['knn_path']%(im_id, cls), \
                        logger=log)
                no_distances = sum([len(d) for d in distances])
                no_bg_distances = len(bg_distances)
                log.debug("Got %d distance_lists, %d bg_distance lists, %d ex_indexes_lists, element of idx_list: %s", len(distances), len(bg_distances), len(nearest_exemplar_indexes), nearest_exemplar_indexes[0])
                log.debug("No of distances: %d, no of bg_dists: %d, no of ex_ix: %d", no_distances, no_bg_distances, sum([len(n) for n in nearest_exemplar_indexes]))
                if no_distances == 0:
                    log.warning('No distances found for im %s, cls %s, k=%s. NO RANKING TO BE FOUND', im_id, cls, TESTopts['k'])
                    log.warning('==== FINISHED RANKING im %s, cls %s ====', image.im_id, cls)
                    continue
                log.debug('Number of distances found %d > 0', no_distances)
                pointslist = [[points[i,:] for k in n] for i,n in enumerate(nearest_exemplar_indexes)]
                log.debug('Built a pointslist: len = %d, inner list sum: %d', len(pointslist), sum([len(p) for p in pointslist]))
                points = np.vstack([p for ppp in pointslist for p in ppp])
                log.debug('Built a pointsarray: shape: %s', points.shape)
                nearest_exemplar_indexes = np.hstack([e for eee in nearest_exemplar_indexes for e in eee])
                log.debug('Built n_ex_ind array: %s', nearest_exemplar_indexes.shape)
                bg_dist = []
                for i, d in enumerate(distances):
                    for j, ddd in enumerate(d):
                        bg_dist.append(bg_distances[i])
                bg_distances = np.hstack(bg_dist)
                log.debug('Built bg_distances array: %s', bg_distances.shape)
                distances = np.hstack([d for ddd in distances for d in ddd])
                log.debug('Built distances array: %s', distances.shape)
                distances = np.vstack([distances, bg_distances]).T
                log.debug('Combined distances array: %s', distances.shape)
    
                log.info('==== LOADING NEAREST EXEMPLARS ====')
                exemplars = load_exemplars(DETopts['exemplar_path']%cls, nearest_exemplar_indexes, logger=log)
    
                log.info('==== GET HYPOTHESES ====')
                hypotheses = get_hypotheses(exemplars, points, image.width, image.height, logger=log)
                if hypotheses.shape[0] == 0:
                    log.warning("== FOUND NO HYPOTHESES . No ranking possible!")
                
                detections = hypotheses
                reflist = [[i] for i in range(hypotheses.shape[0])]
                log.setLevel(logging.DEBUG)
            else:
                detections, reflist, distances, points = load_detections(detfile,im_id, logger=log)
                if detections.shape[0] == 0:
                    log.warning("No detections for image %s, skip this image",im_id)
                    continue
                if not isinstance(reflist[0], np.ndarray):
                    # If reflist is a lst of lists instead of a list of ndarrays, convert
                    reflist = [np.array(l) for l in reflist]
                log.info(" Detections: %s, Reflist: %s (max: %d), distances: %s, points: %s", \
                        detections.shape, len(reflist), max([l.max() for l in reflist]), \
                        distances.shape, points.shape)
            
            if metric == 'qs_density' and DETmethod == 'quickshift':
                qs_parents, qs_dists, qs_E = load_quickshift_tree(DETopts['quickshift_tree_path']%(cls, im_id))
                boolroots = np.array([i==p for i,p in enumerate(qs_parents)])
                log.debug('boolroots sum, size: %s, %s', boolroots.sum(), boolroots.shape)
                qs_E = np.array(qs_E)
                detection_vals = qs_E[boolroots]
                log.info('Quickshift density Estimates of detections: size: %s', detection_vals.shape)
            else:
                detection_vals = get_detection_values(detections, reflist, distances, \
                        points, eval(metric), logger=log)
            log.info("im %s: det shape=%s, det_vals shape=%s"%(im_id, \
                    detections.shape, detection_vals.shape))
            all_detections.append(detections)
            all_det_vals.append(detection_vals)
            imids = np.array([im_id for i in range(detections.shape[0])])
            all_det_imids.append(imids)
            log.info("stored imids shape:%s", imids.shape)
        all_detections = np.vstack(all_detections)
        if len(all_det_vals[0].shape) > 1:
            all_det_vals = np.vstack(all_det_vals)
        else:
            all_det_vals = np.hstack(all_det_vals)
        all_det_imids = np.hstack(all_det_imids)
        log.info("Found %s detections, %s vals, %s imids", all_detections.shape, \
                all_det_vals.shape, all_det_imids.shape)
        if len(all_det_vals.shape) > 1:
            ranking = sort_values(all_det_vals, logger=log)
            log.info("ranking shape: %s", ranking.shape)
            save_voc_results(outputf, all_detections[ranking], all_det_vals[ranking], \
                    all_det_imids[ranking], logger=log)
        else:
            save_voc_results(outputf, all_detections, all_det_vals, all_det_imids, logger=log)
    
    log.info('FINISHED')

if __name__ == "__main__":
    
    # Get config settings
    if len(sys.argv) < 2:
        raise Exception("Please give a config file as command line argument")
    configfile = sys.argv[1]
    if len(sys.argv) == 3:
        if sys.argv[2] == '--rankingonly':
            print 'rankingonly'
            rankingonly = True
            testlogging = False
        elif sys.argv[2] == '--testlogging':
            print 'testlogging'
            testlogging = True
            rankingonly = False
    else:
        rankingonly = False
        testlogging = False
    
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)

    # Setup multiprocessing logging
    multiprocessing.log_to_stderr()
    mplogger = multiprocessing.get_logger()
    mplogger.setLevel(logging.INFO)
    
    # Setup logger
    log = init_log(GLOBopts['log_path'], 'training', 'w')
    
    nn_threads = GLOBopts['nn_threads']
    det_threads = GLOBopts['det_threads']
    rank_threads = GLOBopts['rank_threads']
    set_mode = GLOBopts['setmode']
    if set_mode == 'voc':
        test_classes = VOCopts.classes
        train_classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',\
            'chair','cow','diningtable','dog','horse','motorbike','person',\
            'pottedplant','sheep','sofa','train','tvmonitor', 'background']
    elif set_mode == 'becker':
        test_classes = ['motorbike', 'background']
        train_classes = ['motorbike', 'background']
    
    no_test_classes = len(test_classes)
    no_train_classes = len(train_classes)
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = init_descriptor(DESCRopts[0])
    
    if not rankingonly:
        log.info('==============================')
        log.info('========== TRAINING ==========')
        log.info('==============================')
            
        # VOC07 detection
            
        log.info('==== INIT ESTIMATOR FOR CLASS ====')
        estimator = init_estimator(GLOBopts['nbnn_path']%'estimator', NBNNopts)
        
        if set_mode == 'voc':
            train_local(train_classes, descriptor_function, estimator, VOCopts, GLOBopts, NBNNopts, TESTopts, DETopts, log)
        elif set_mode == 'becker':
            load_becker_estimator(descriptor_function, estimator, VOCopts, \
                train_set = GLOBopts['train_set'],\
                descriptor_path = GLOBopts['descriptor_path'],\
                exemplar_path = DETopts[1]['exemplar_path'])
        
        log.info('==== TRAINING FINISHED ====')
        
        log.info('==============================')
        log.info('======== MAKE BATCHES ========')
        log.info('==============================')
        
        # Save descriptors of test set to disk
        orig_impath = VOCopts.image_path
        
        if GLOBopts['setmode'] == 'becker':
            VOCopts.image_path = VOCopts.image_path[:-4]+'.png'
        batches = make_voc_batches(descriptor_function, VOCopts, GLOBopts, TESTopts)
        VOCopts.image_path = orig_impath
        log.info('==== BATCHMAKING FINISHED ====')
        
        """ Now, Do stuff per batch and per class, so multithread!"""
        
        no_batches = len(batches)
        
        log.info("No of NN-threads: %d:",nn_threads)
        log.info("No of batches: %d",no_batches)
        log.info("No of train classes: %d", no_train_classes)
        log.info("No of test classes: %d", no_test_classes)
        
        log.info('==============================')
        log.info('===== NN for all BATCHES =====')
        log.info('==============================')
        
        nn_pool = Pool(processes = nn_threads)
        argtuples = []
        for batch_no, batch in enumerate(batches):
            for cls in train_classes:
                log.info('ADD BATCH NO: %d, CLS: %s to the pool', batch_no, cls)
                argtuples.append((batch_no, cls, batch, configfile))
        nn_pool.map(LogExceptions(get_detection_dists), argtuples)
        nn_pool.close()
        
        # GET THE OVERALL kNN
        log.info('==============================')
        log.info('===== K-NN for all IMAGES ====')
        log.info('==============================')
            
        knn_pool = Pool(processes = det_threads)
        argtuples = []
        for batch in batches:
            for im in batch:
                argtuples.append((im, configfile))
        knn_pool.map(LogExceptions(get_knn), argtuples)
        knn_pool.close()
        # DETECTION PER IMAGE
        log.info('==============================')
        log.info('== DETECTION FOR ALL IMAGES ==')
        log.info('==============================')
        det_pool = Pool(processes = det_threads)
        argtuples = []
        for batch in batches:
            for im in batch:
                for cls in test_classes:
                    if not cls == 'background':
                        argtuples.append((im, cls, configfile))
        det_pool.map(LogExceptions(detection), argtuples)
        det_pool.close()
    
    log.info('==============================')
    log.info('======= RANK DETECTIONS ======')
    log.info('==============================')
    
    rank_pool = Pool(processes = rank_threads)
    argtuples = []
    for cls in test_classes:
        if not cls == 'background':
            argtuples.append((cls, configfile))
    rank_pool.map(LogExceptions(rank_detections), argtuples)
    rank_pool.close()
    
    log.info('==============================')
    log.info('======== FINISHED TEST =======')
    log.info('==============================')
    