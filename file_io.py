import cPickle
import numpy as np
import Image
import logging

log = logging.getLogger(__name__)

def save_testinfo(filename, batches, classes):
    """ Save a file with information on how many iterations with how many
    classes, and which ones they are, for the multiple processes that are
    gonna run the tests
    
    """
    log.info('++ SAVING testinfo (batches:%d, classes:%s) to %s ++', len(batches), classes, filename)
    with open(filename,'w') as testfile:
        testfile.write("%d\n"%len(batches))
        testfile.write("%d\n"%len(classes))
        for cls in classes:
            testfile.write("%s "%cls)

def save_batch(filename, batch):
    log.info('++ SAVING batch: (len=%d), to %s ++', len(batch), filename)
    
    with open(filename, 'wb') as pklfile:
        cPickle.dump(batch, pklfile)
    with open(filename + '.txt', 'w') as txtf:
        for im in batch:
            txtf.write(im.im_id+'\n')

def load_batch(filename):
    
    with open(filename, 'rb') as pklf:
        images = cPickle.load(pklf)
    log.info('++ LOADING batch: (len=%d), from %s ++', len(images), filename)
    
    return images

def save_distances(path, cls, distances, points_list, images, nearest_exemplar_indexes):
    """ Save a list of distances, points, images and indexes of the point's 
        nearest exemplar indexes
    """
    log.info('++ SAVING distances: path:%s, cls:%s, distances:%d, pts_list:%d, imgs:%d, NN_ex_idxs:%d', \
        path, cls, len(distances), len(points_list), len(images), len(nearest_exemplar_indexes))
    
    for i,im in enumerate(images):
        log.debug(' + SAVING to file: %s, no_dist: %s, no_pts: %s, no_n.e.i.: %s', \
            path%(im.im_id, cls), distances[i].shape, points_list[i].shape, nearest_exemplar_indexes[i].shape)
        with open(path%(im.im_id, cls), 'wb') as dfile:
            cPickle.dump(distances[i], dfile)
            cPickle.dump(points_list[i], dfile)
            cPickle.dump(images[i], dfile)
            cPickle.dump(nearest_exemplar_indexes[i], dfile)

def load_distances(filename):
    """ Load distances, points and indexes of the point's nearest exemplar
        indexes of a certain image (im_id), or the full file content (im_id=None)
    
        distances and points can be either 2D or 3D (when k>1):
        dimensions: distances[no_descriptors, no_classes, k]
                    nearest_exemplar_indexes[no_descriptors, no_classes, k]
        
    """

    with open(filename, 'rb') as f:
        distances = cPickle.load(f)
        allpoints = cPickle.load(f)
        image = cPickle.load(f)
        nearest_exemplar_indexes = cPickle.load(f)
    log.info('++ LOADING distances from %s, distances:%s, pts_list:%s, im_id:%s, NN_ex_idxs:%s', \
        filename, distances.shape, allpoints.shape, image.im_id, nearest_exemplar_indexes.shape)
    
    return distances, allpoints, image, nearest_exemplar_indexes

def save_exemplars(filename, exemplars):
    ex_stack = np.vstack(exemplars)
    
    log.info('++ SAVING exemplars to %s: len:%d, total:%s', filename, \
        len(exemplars), ex_stack.shape)
    with open(filename, 'wb') as ef:
        np.save(ef, ex_stack)
        # cPickle.dump(exemplars, ef)
    
def load_exemplars(filename, exemplar_indexes = None):
    
    """ Load exemplars of the trained descriptors, and select the indexes needed
    (exemplar_indexes), or return the full array (exemplar_indexes = None)
    
    """
    
    with open(filename, 'rb') as exemplar_f:
        exemplars = np.load(exemplar_f)
    # exemplars is an np.array, nx4(xk), where n=no of exemplars in a class, and k is the amount of NN taken (if >1)
    # the cols are [rel_bb_w, rel_bb_h, rel_x, rel_y]
    if not exemplar_indexes is None:
        log.info('++ LOADING exemplars from %s: total:%s, selecting: %s indexes', \
            filename, exemplars.shape, exemplar_indexes.shape)
        return exemplars[exemplar_indexes]
    else:
        log.info('++ LOADING exemplars from %s: total:%s, selecting: ALL', \
            filename, exemplars.shape)
        return exemplars

def load_imarray(filename):
    """ Open an image file and convert it to an np.array
    
    """
    im = np.asarray(Image.open(filename))
    log.info("++ LOADING imarray from %s: size:%s", filename, im.shape)
    return im

def load_hypotheses(filename):
    """ DEPRECATED
    """
    log.warning('DEPRECATED FUNCTION CALL: save_hypotheses: use save/load_distances instead')
    
    with open(filename, 'rb') as f:
        hypotheses = cPickle.load(f)
        fg_points = cPickle.load(f)
        im_exemplars = cPickle.load(f)
    return (hypotheses, points, im_exemplars)

def save_detections(filename, detections, reflist, descr_distances=None, descr_points=None):
    if descr_distances is None:
        descr_distances = np.array([])
    if descr_points is None:
        descr_points = np.array([])
    log.info('++ SAVING detections to %s: detections:%s, reflist:%s, descr_dist:%s, descr_points:%s', \
        filename, detections.shape, len(reflist), descr_distances.shape, \
        descr_points.shape)
    with open(filename,'wb') as pklfile:
            cPickle.dump(detections, pklfile)
            cPickle.dump(reflist, pklfile)
            cPickle.dump(descr_distances, pklfile)
            cPickle.dump(descr_points, pklfile)

def load_detections(filename, im_id):
    with open(filename, 'rb') as f:
        detections = cPickle.load(f)
        reflist = cPickle.load(f)
        descr_distances = cPickle.load(f)
        descr_points = cPickle.load(f)
    log.info('++ LOADING detections from %s: detections:%s, reflist:%s descr_dist:%s, descr_points:%s', \
        filename, detections.shape, len(reflist), descr_distances.shape, \
        descr_points.shape)
    return detections, reflist, descr_distances, descr_points

def save_voc_results(filename, detections, values, im_ids):
    """Assuming values is array with values higher=more confidence
    
    """
    log.info('++SAVING voc_results file to %s: detections:%s',filename, detections.shape)
    if len(values.shape)>1:
        # If values consist of multiple columns, enumerate to get the confidence
        l = values.shape
        firstval = values[0,:]
        lastval = values[-1,:]
        values = np.arange(l[0])
        log.info('made values of shape: %s into range [0,...,%d]',l, values.shape[0])
        log.info('first entry was: %s, and becomes %d',firstval, values[0])
        log.info('last entry was: %s, and becomes %d',lastval, values[-1])
    with open(filename, 'w') as f:
        for i in xrange(values.shape[0]):
            f.write("%s %f %s\n"%(im_ids[i], values[i], "%f %f %f %f"%tuple(detections[i,:])))
    

def save_to_pickle(filename, datalist):
    log.warning("DEPRECATED function call save_to_pickle")
    
    if not os.path.exists(filename):
        with open(filename,'wb') as pklfile:
            for d in datalist:
                cPickle.dump(d, pklfile)
    else:
        data_old = []
        with open(filename,'rb') as pklfile:
            for d in xrange(len(datalist)):
                data_old.append(cPickle.load(pklfile))
        for o, d in zip(data_old, datalist):
            if len(o.shape) == 1:
                np.hstack([o, d])
            else:
                np.vstack([o,d])
        with open(filename, 'wb') as pklfile:
            for o in data_old:
                cPickle.dump(data_old, pklfile)

def save_results_to_file(file, objects, confidence_values):
    log = logging.getLogger("__name__")
    if isinstance(objects[0], VOCImage) or isinstance(objects[0], CalImage):
        log.info('++ SAVING image classification')
        with open(file, 'a') as f:
            for obj, cv in zip(objects,confidence_values):
                f.write('%s %f\n'%(obj.im_id, cv))
    elif isinstance(objects[0], Object):
        log.info('++ SAVING image detection files (by bbox)')
        with open(file, 'a') as f:
            for obj, cv in zip(objects,confidence_values):
                f.write('%s %f %d %d %d %d\n'%(obj.image.im_id, cv, \
                    obj.xmin, obj.ymin, obj.xmax, obj.ymax))
                
    log.info(" + SAVED results to %s",file)

def load_results(filename, im_id):
    with open(filename, 'r') as f:
        content = f.read()
    imlines = [line for line in content.split('\n') if im_id in line]
    if len(imlines) <1:
        raise Exception('No detections available, exiting')
    detections = []
    confidences = []
    for line in imlines:
        words = line.split(' ')
        detections.append([float(w) for w in words[2:]])
        confidences.append(float(words[1]))
    return detections, confidences