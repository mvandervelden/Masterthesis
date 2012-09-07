import cPickle
import numpy as np
import Image

def save_testinfo(filename, batches, classes):
    """ Save a file with information on how many iterations with how many
    classes, and which ones they are, for the multiple processes that are
    gonna run the tests
    
    """
    with open(filename,'w') as testfile:
        testfile.write("%d\n"%len(batches))
        testfile.write("%d\n"%len(classes))
        for cls in classes:
            testfile.write("%s "%cls)

def save_batch(filename, batch):
    with open(filename, 'wb') as pklfile:
        cPickle.dump(batch, pklfile)
    with open(filename + '.txt', 'w') as txtf:
        for im in batch:
            txtf.write(im.im_id+'\n')

def load_batch(filename):
    with open(filename, 'rb') as pklf:
        images = cPickle.load(pklf)
    return images

def save_distances(path, cls, distances, points_list, images, nearest_exemplar_indexes):
    """ Save a list of distances, points, images and indexes of the point's 
        nearest exemplar indexes
    """
    for i,im in enumerate(images):
        with open(path%(im.im_id, cls), 'wb') as dfile:
            cPickle.dump(distances[i], dfile)
            cPickle.dump(points_list[i], dfile)
            cPickle.dump(images[i], dfile)
            cPickle.dump(nearest_exemplar_indexes[i], dfile)

def load_distances(filename):
    """ Load distances, points and indexes of the point's nearest exemplar
        indexes of a certain image (im_id), or the full file content (im_id=None)
    
    """
    with open(filename, 'rb') as f:
        distances = cPickle.load(f)
        allpoints = cPickle.load(f)
        image = cPickle.load(f)
        nearest_exemplar_indexes = cPickle.load(f)
    return distances, allpoints, image, nearest_exemplar_indexes

def save_exemplars(filename, exemplars):
    with open(filename, 'wb') as ef:
        np.save(ef, np.vstack(exemplars))
        # cPickle.dump(exemplars, ef)
    
def load_exemplars(filename, exemplar_indexes = None):
    """ Load exemplars of the trained descriptors, and select the indexes needed
    (exemplar_indexes), or return the full array (exemplar_indexes = None)
    
    """
    
    with open(filename, 'rb') as exemplar_f:
        exemplars = np.load(exemplar_f)
    # exemplars is an np.array, nx4, where n=no of exemplars in a class
    # the cols are [rel_bb_w, rel_bb_h, rel_x, rel_y]
    if not exemplar_indexes is None:
        return exemplars[exemplar_indexes]
    else:
        return exemplars

def load_imarray(filename):
    """ Open an image file and convert it to an np.array
    
    """
    
    return np.asarray(Image.open(filename))


def load_hypotheses(filename):
    with open(filename, 'rb') as f:
        hypotheses = cPickle.load(f)
        fg_points = cPickle.load(f)
        im_exemplars = cPickle.load(f)
    return (hypotheses, points, im_exemplars)

def save_detections(filename, detections, reflist):
    with open(filename,'wb') as pklfile:
            cPickle.dump(detections, pklfile)
            cPickle.dump(reflist, pklfile)

def load_detections(filename, im_id):
    with open(filename, 'rb') as f:
        detections = cPickle.load(f)
        reflist = cPickle.load(f)
    return detections, reflist

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

def save_to_pickle(filename, datalist):
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
        log.info('Saving image classification')
        with open(file, 'a') as f:
            for obj, cv in zip(objects,confidence_values):
                f.write('%s %f\n'%(obj.im_id, cv))
    elif isinstance(objects[0], Object):
        log.info('Saving image detection files (by bbox)')
        with open(file, 'a') as f:
            for obj, cv in zip(objects,confidence_values):
                f.write('%s %f %d %d %d %d\n'%(obj.image.im_id, cv, \
                    obj.xmin, obj.ymin, obj.xmax, obj.ymax))
                
    log.info("Saved results to %s",file)
