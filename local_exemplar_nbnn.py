import sys
import numpy as np
from multiprocessing import Pool

from logutils import *
from nbnn import *
from nbnn.voc import *

from utils import *
from procedures import *
from file_io import *
from detection_utils import *

def train_local(classes, descriptors_function, estimator, VOCopts, GLOBopts, NBNNopts, TESTopts, DETopts, log):
    for i, cls in enumerate(classes):
        # Get classes images, descriptors, and add it to the estimator
        log.info('==== GET CLASS %d: %s IMAGES ====', i, cls)
        img_set = read_image_set(VOCopts,cls+'_'+GLOBopts['train_set'])
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
                    save_exemplars(exemplar_path%cls, exemplars)
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
                # Find which objects are cls
                object_ids = [o.object_id for o in im.objects]
                cls_obj_ids = [o.object_id for o in im.objects \
                        if o.class_name == cls]
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
                save_exemplars(exemplar_path%cls, exemplars)
        log.info('=== CLS %s (%d/%d) ADDED ===',cls, i, len(classes))


def get_detection_dists((batch_no, cls, batch, configfile)):
    
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
    
    log.debug("-- returning array of shape %s"%(cls_dst.shape,))
    log.debug("-- mean estimate: %s"%np.mean(cls_dst))
    log.debug("-- max estimate: %s"%np.max(cls_dst))
    log.debug("-- min estimate: %s"%np.min(cls_dst))
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
    
    classes = VOCopts.classes + ['background']
    no_classes = len(classes)
    
    distlist = []
    ptslist = []
    exemp_idxlist = []
    
    for cls in classes:
        distances, allpoints, im, nearest_exemplar_indexes = load_distances( \
                DETopts['distances_path']%(im_id, cls), logger=log)
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
    # Save points, same over all classes of an image
    save_points(DETopts['knn_path']%(im_id, 'points'), ptslist[0], logger=log)
    
    k = TESTopts['k']
    # n= no of descriptors in test image
    n = distances.shape[0]
    # Make a selection of the k nearest neighbors overall
    
    distances = np.reshape(distances,[n, no_classes*k])
    exemplar_indexes = np.reshape(exemplar_indexes, [n,no_classes*k])
    asort = np.argsort(distances)
    
    cls_idxs = np.array([i for i in range(no_classes) for kkk in range(k)])
    
    cls_dists = [[[] for nn in range(n)] for c in classes]
    cls_exempl = [[[] for nn in range(n)] for c in classes]
    
    log.info('Reshaped dist & ex_ind: %s & %s', distances.shape, exemplar_indexes.shape)
    log.info('Argsort shape: %s'. asort.shape)
    log.info('cls_idxs shape: %s', cls_idxs.shape)
    log.info('cls_dists (%d lists of %d lists) & cls_exempl (%d lists of %d lists)', \
            len(cls_dists), len(cls_dists[0]), len(cls_exempl), len(cls_exempl[0]))
    
    for i in range(n):
        sort_dists = distances[i,asort[i,:k]]
        sort_cls = cls_idxs[asort[i,:k]]
        sort_exempl = exemplar_indexes[i,asort[i,:k]]
        
        for j, cl in enumerate(sort_cls):
            cls_dists[cl][i].append(sort_dists[j])
            cls_exempl[cl][i].append(sort_exempl[j])
    total_dists = 0
    goal = k*n
    for c in range(no_classes):
        cls = classes[c]
        no_cls_dists = sum([len(n) for n in cls_dists[c]])
        total_dists += no_cls_dists
        log.info('Saving knn for cls %s, %d dists', cls, no_cls_dists)
        save_knn(DETopts['knn_path']%(im_id, cls), cls_dists[c], cls_exempl[c], logger=log)
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
    distances, nearest_exemplar_indexes = load_knn(DETopts['knn_path']%(im_id, cls), \
            logger=log)
    pointslist = [[points[i] for k in n] for i,n in enumerate(nearest_exemplar_indexes)]
    points = np.vstack([p for ppp in pointslist for p in ppp])
    nearest_exemplar_indexes = np.vstack([e for eee in nearest_exemplar_indexes for e in eee])
    distances = np.vstack([d for ddd in distances for d in ddd])
    log.info('==== LOADING NEAREST EXEMPLARS ====')
    exemplars = load_exemplars(DETopts['exemplar_path']%cls, nearest_exemplar_indexes, logger=log)
    
    log.info('==== GET HYPOTHESES ====')
    
    hypotheses = get_hypotheses(exemplars, points, image.width, image.height, logger=log)
    if hypotheses.shape[0] == 0:
        log.debug("== FOUND NO HYPOTHESES WITH fg_d < bg_d. No clustering possible!")
    hvalues = get_hypothesis_values(hypotheses, distances, points, eval(DETopts['hypothesis_metric']))
    ranking = sort_values(hvalues)
    
    # Keep only the best n descriptors (largest relative margin d+, d-)
    if 'hyp_cutoff' in DETopts:
        log.info('Using %s hypotheses, out of %d', DETopts['hyp_cutoff'], hypotheses.shape[0])
        ranking = ranking[:int(DETopts['hyp_cutoff'])]
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
    

if __name__ == "__main__":
    
    # Get config settings
    if len(sys.argv) < 2:
        raise Exception("Please give a config file as command line argument")
    configfile = sys.argv[1]
    
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)
    
    # Setup logger
    log = init_log(GLOBopts['log_path'], 'training', 'w')
    
    nn_threads = GLOBopts['nn_threads']
    det_threads = GLOBopts['det_threads']
    classes = VOCopts.classes + ['background']
    no_classes = len(classes)
    
    
    log.info('==== INIT DESCRIPTOR FUNCTION ====')
    descriptor_function = init_descriptor(DESCRopts[0])
    
    log.info('==============================')
    log.info('========== TRAINING ==========')
    log.info('==============================')
    
    # VOC07 detection
    log.info('==== INIT ESTIMATOR FOR CLASS ====')
    estimator = init_estimator(GLOBopts['nbnn_path']%'estimator', NBNNopts)
    
    train_local(classes, descriptors_function, estimator, VOCopts, GLOBopts, NBNNopts, TESTopts, DETopts, log)
    
    log.info('==== TRAINING FINISHED ====')
    
    log.info('==============================')
    log.info('======== MAKE BATCHES ========')
    log.info('==============================')
    
    # Save descriptors of test set to disk
    batches = make_voc_batches(descriptor_function, VOCopts, GLOBopts, TESTopts)
    log.info('==== BATCHMAKING FINISHED ====')
    
    """ Now, Do stuff per batch and per class, so multithread!"""
    
    no_batches = len(batches)

    log.info("No of NN-threads: %d:",nn_threads)
    log.info("No of batches: %d",no_batches)
    log.info("No of classes: %d", no_classes)
    
    log.info('==============================')
    log.info('===== NN for all BATCHES =====')
    log.info('==============================')
    
    nn_pool = Pool(processes = nn_threads)
    argtuples = []
    for batch_no, batch in enumerate(batches):
        for cls in classes:
            log.info('ADD BATCH NO: %d, CLS: %d to the pool')
            argtuples.append((batch_no, cls, batch, configfile))
    nn_pool.map(get_detection_dists, argtuples)
    
    # GET THE OVERALL kNN
    
    log.info('==============================')
    log.info('===== NN for all IMAGES ======')
    log.info('==============================')
    
    knn_pool = Pool(processes = det_threads)
    argtuples = []
    for batch in batches:
        for im in batches:
            argtuples.append((im, configfile))
    knn_pool.map(get_knn, argtuples)
    
    # DETECTION PER IMAGE
    log.info('==============================')
    log.info('== DETECTION FOR ALL IMAGES ==')
    log.info('==============================')
    det_pool = Pool(processes = det_threads)
    argtuples = []
    for batch in batches:
        for im in batches:
            for cls in classes:
                if not cls == 'background':
                    argtuples.append((im, cls, configfile))
    det_pool.map(detection, argtuples)
    log.info('==============================')
    log.info('======== FINISHED TEST =======')
    log.info('==============================')
    