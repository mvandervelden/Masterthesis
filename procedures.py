import cPickle
import logging
import os.path
import numpy as np
from utils import *
from nbnn.voc import *
from file_io import *

log = logging.getLogger(__name__)

def train_voc(descriptor_function, estimator, object_type, VOCopts,\
        train_set='train', descriptor_path=None, exemplar_path=None, cls=None):
    if not cls is None:
        classes = [cls]
    else:
        classes = VOCopts.classes
    
    for i,cls in enumerate(classes):
        if (not object_type == 'fgbg' and not cls in estimator.classes) or \
                (object_type == 'fgbg' and not cls+'_fg' in estimator.classes):
            log.info('==== GET CLASS %d: %s IMAGES ====', i, cls)
            img_set = read_image_set(VOCopts,cls+'_'+ train_set)
            if not descriptor_path is None:
                for im in img_set:
                    if os.path.exists(descriptor_path%im.im_id):
                        # Make sure not to calculate descriptors again if they already exist
                        im.descriptor_file.append(descriptor_path%im.im_id)
            log.info('==== GET %s DESCRIPTORS ====', cls)
            descriptors = get_image_descriptors(img_set, descriptor_function, \
                descriptor_path)
            if not object_type == 'fgbg':
                if object_type == 'bbox':
                    # Get descriptor's objects (bboxes):
                    log.info('==== GET %s OBJECTS ====', cls)
                    objects = get_objects_by_class(img_set, cls)
                    descriptors = get_bbox_descriptors(objects, descriptors)
                elif object_type == 'image':
                    descriptors = [d for p,d in descriptors.values()]
                log.info('==== ADD %s DESCRIPTORS TO ESTIMATOR', cls)
                estimator.add_class(cls, descriptors)
            else:
                # Get descriptor's objects (bboxes):
                log.info('==== GET %s OBJECTS ====', cls)
                objects = get_objects_by_class(img_set, cls)
                bg_descriptors = get_bbox_bg_descriptors(objects, descriptors)
                estimator.add_class(cls+'_bg', bg_descriptors)
                if not exemplar_path is None:
                    fg_descriptors, exemplars = get_bbox_descriptors(objects, descriptors, exemplars=True)
                    estimator.add_class(cls+'_fg', fg_descriptors)
                    log.info('==== SAVING EXEMPLARS to %s ====', exemplar_path)
                    save_exemplars(exemplar_path%cls, exemplars)
                else:
                    fg_descriptors = get_bbox_descriptors(objects, descriptors)
                    estimator.add_class(cls+'_fg', fg_descriptors)

def load_becker_estimator(descriptor_function, estimator, VOCopts, \
        train_set='train', descriptor_path=None, exemplar_path=None):
    cls = 'motorbike'
    # FG images are png: change image path:
    origpath = VOCopts.image_path
    VOCopts.image_path = VOCopts.image_path[:-4]+'.png'
    if not cls in estimator.classes:
        log.info('==== GET CLASS %s IMAGES ====', cls)
        img_set = read_image_set(VOCopts, cls+'_'+train_set)
        log.info('==== LOADING CLASS SEGMENTTION MASKS ====')
        load_object_segmentations(VOCopts.class_mask_path, img_set)
        if not descriptor_path is None:
            for im in img_set:
                if os.path.exists(descriptor_path%im.im_id):
                    # Make sure not to calculate descriptors again if they already exist
                    im.descriptor_file.append(descriptor_path%im.im_id)
        log.info('==== GET %s DESCRIPTORS ====', cls)
        descriptors = get_image_descriptors(img_set, descriptor_function, \
            descriptor_path)
        fg_descr = []
        exemplars = []
        for im in img_set:
            object_idxs = list(np.unique(im.object_segmentation))
            log.info('=== Image %s, bb [%d,%d,%d,%d] partitioning ===', im.im_id, \
                im.objects[0].xmin, im.objects[0].ymin, \
                im.objects[0].xmax, im.objects[0].ymax)
            log.info(' --- object idxs: %s', object_idxs)
            log.info(' --- pts: %s, descr: %s', descriptors[im.im_id][0].shape, descriptors[im.im_id][1].shape)
            if not exemplar_path is None:
                imdescr, impts = partition_descriptors(np.matrix(descriptors[im.im_id][0]), \
                    descriptors[im.im_id][1], im.object_segmentation, exemplars=True)
                exmps = get_exemplars(im.objects[0], np.array(impts[1]))
                log.info(' --- adding %s exemplars, %s descr', len(exmps), len(imdescr[1]))
                exemplars.append(exmps)
            else:
                imdescr, impts = partition_descriptors(np.matrix(descriptors[im.im_id][0]), \
                    descriptors[im.im_id][1], im.object_segmentation, exemplars=False)
                log.info(' --- adding %s descr', len(imdescr))
            fg_descr.append(imdescr[1])


        log.info('--- Adding %s descriptor arrays to class %s', len(fg_descr), cls)
        estimator.add_class(cls, fg_descr) #TODO SHOULD WORK....
        if not exemplar_path is None:
            log.info('==== SAVING EXEMPLARS to %s ====', exemplar_path)
            save_exemplars(exemplar_path%cls, exemplars)
    
    cls = 'background'
    # BG consists of .jpg images, change VOCopts again...
    VOCopts.image_path = origpath
    if not cls in estimator.classes:
        log.info('==== GET CLASS %s IMAGES ====', cls)
        img_set = read_image_set(VOCopts, cls+'_'+train_set)
        log.info('==== LOADING CLASS SEGMENTTION MASKS ====')
        load_object_segmentations(VOCopts.class_mask_path, img_set)
        if not descriptor_path is None:
            for im in img_set:
                if os.path.exists(descriptor_path%im.im_id):
                    # Make sure not to calculate descriptors again if they already exist
                    im.descriptor_file.append(descriptor_path%im.im_id)
        log.info('==== GET %s DESCRIPTORS ====', cls)
        descriptors = get_image_descriptors(img_set, descriptor_function, \
            descriptor_path)
        for im, (p, d) in descriptors.items():
            descriptors[im] = (np.matrix(p), d)
        log.info('==== Select which DESCRIPTORS are %s ====', cls)
        bg_descriptors = get_background_descriptors(img_set, descriptors)
        log.info('--- Adding %s descriptor arrays to class %s', len(bg_descriptors), cls)
        estimator.add_class(cls, bg_descriptors) #TODO SHOULD WORK...

def load_behmo_estimator(descriptor_function, estimator, cls, VOCopts, \
        train_set='train', descriptor_path=None, exemplar_path=None):
    if not cls+'_fg' in estimator.classes:
        log.info('==== GET CLASS %s IMAGES ====', cls)
        img_set = read_image_set(VOCopts,cls+'_'+train_set)
        if not descriptor_path is None:
            for im in img_set:
                if os.path.exists(descriptor_path%im.im_id):
                    # Make sure not to calculate descriptors again if they already exist
                    im.descriptor_file.append(descriptor_path%im.im_id)
        log.info('==== GET %s DESCRIPTORS ====', cls)
        descriptors = get_image_descriptors(img_set, descriptor_function, \
            descriptor_path)
        # Get descriptor's objects (bboxes):
        log.info('==== GET %s OBJECTS ====', cls)
        objects = get_objects_by_class(img_set, cls)
        bg_descriptors = get_bbox_bg_descriptors(objects, descriptors)
        estimator.add_class(cls+'_bg', bg_descriptors)
        if not exemplar_path is None:
            fg_descriptors, exemplars = get_bbox_descriptors(objects, descriptors, exemplars=True)
            estimator.add_class(cls+'_fg', fg_descriptors)
            log.info('==== SAVING EXEMPLARS to %s ====', exemplar_path)
            save_exemplars(exemplar_path%cls, exemplars)
        else:
            fg_descriptors = get_bbox_descriptors(objects, descriptors)
            estimator.add_class(cls+'_fg', fg_descriptors)

def train_behmo_becker(descriptor_function, estimator, VOCopts, val_set, \
        descriptor_path=None):
    cls = 'motorbike'
    log.info('==== BECKER VALIDATE CLASS %s ====', cls)
    # FG images are png: change image path:
    origpath = VOCopts.image_path
    VOCopts.image_path = VOCopts.image_path[:-4]+'.png'
    
    log.info('==== GET CLASS %s IMAGES ====', cls)
    img_set = read_image_set(VOCopts, cls+'_'+val_set)
    log.info('==== LOADING CLASS SEGMENTTION MASKS ====')
    load_object_segmentations(VOCopts.class_mask_path, img_set)
    if not descriptor_path is None:
        for im in img_set:
            if os.path.exists(descriptor_path%im.im_id):
                # Make sure not to calculate descriptors again if they already exist
                im.descriptor_file.append(descriptor_path%im.im_id)
    log.info('==== GET %s DESCRIPTORS ====', cls)
    descriptors = get_image_descriptors(img_set, descriptor_function, \
        descriptor_path)
    fg_descr = []
    for im in img_set:
        object_idxs = list(np.unique(im.object_segmentation))
        log.info('=== Image %s, bb [%d,%d,%d,%d] partitioning ===', im.im_id, \
            im.objects[0].xmin, im.objects[0].ymin, \
            im.objects[0].xmax, im.objects[0].ymax)
        log.info(' --- object idxs: %s', object_idxs)
        log.info(' --- pts: %s, descr: %s', descriptors[im.im_id][0].shape, descriptors[im.im_id][1].shape)
        imdescr, impts = partition_descriptors(np.matrix(descriptors[im.im_id][0]), \
            descriptors[im.im_id][1], im.object_segmentation, exemplars=False)
        log.info(' --- adding %s descr', len(imdescr))
        fg_descr.append(imdescr[1])
    
    cls = 'background'
    # BG consists of .jpg images, change VOCopts again...
    VOCopts.image_path = origpath
    
    log.info('==== GET CLASS %s IMAGES ====', cls)
    img_set = read_image_set(VOCopts, cls+'_'+train_set)
    log.info('==== LOADING CLASS SEGMENTTION MASKS ====')
    load_object_segmentations(VOCopts.class_mask_path, img_set)
    if not descriptor_path is None:
        for im in img_set:
            if os.path.exists(descriptor_path%im.im_id):
                # Make sure not to calculate descriptors again if they already exist
                im.descriptor_file.append(descriptor_path%im.im_id)
    log.info('==== GET %s DESCRIPTORS ====', cls)
    descriptors = get_image_descriptors(img_set, descriptor_function, \
        descriptor_path)
    for im, (p, d) in descriptors.items():
        descriptors[im] = (np.matrix(p), d)
    log.info('==== Select which DESCRIPTORS are %s ====', cls)
    bg_descriptors = get_background_descriptors(img_set, descriptors)
    
    log.info('==== TRAIN BEHMO alphas and betas for %s', cls)
    ground_truth =  ['motorbike' for i in xrange(len(fg_descriptors))] + \
                    ['background' for i in xrange(len(bg_descriptors))]
    
    estimator.train(fg_descr+bg_descriptors, ground_truth)
    
def train_behmo(descriptor_function, estimator, cls, VOCopts, val_set='val', \
        descriptor_path=None):
    
    log.info('==== VALIDATE CLASS %s ====', cls)
    img_set = read_image_set(VOCopts,cls + '_' + val_set)
    if not descriptor_path is None:
        for im in img_set:
            if os.path.exists(descriptor_path%im.im_id):
                # Make sure not to calculate descriptors again if they already exist
                im.descriptor_file.append(descriptor_path%im.im_id)
    log.info('==== GET %s VAL-DESCRIPTORS ====', cls)
    descriptors = get_image_descriptors(img_set, descriptor_function, \
        descriptor_path)

    # Get descriptor's objects (bboxes):
    log.info('==== GET %s VAL-OBJECTS ====', cls)
    objects = get_objects_by_class(img_set, cls)
    bg_descriptors = get_bbox_bg_descriptors(objects, descriptors)
    fg_descriptors = get_bbox_descriptors(objects, descriptors)
    
    log.info('==== TRAIN BEHMO alphas and betas for %s', cls)
    ground_truth =  [cls+'_fg' for i in xrange(len(fg_descriptors))] + \
                    [cls+'_bg' for i in xrange(len(bg_descriptors))]
    
    estimator.train(fg_descriptors+bg_descriptors, ground_truth)
    

def make_voc_batches(descriptor_function, VOCopts, GLOBopts, TESTopts):
    log.info('==== GENERATING TEST IMAGES =====')
    test_images = read_image_set(VOCopts, GLOBopts['test_set'])
    log.info('==== GENERATING AND SAVING TEST DESCRIPTORS =====')
    save_image_descriptors(test_images, descriptor_function, \
        GLOBopts['descriptor_path'])
    batches = get_image_batches(VOCopts, test_images, TESTopts['batch_size'])
    log.info('==== SAVING IMAGE OBJECTS PER BATCH =====')
    for b, batch in enumerate(batches):
        save_batch(TESTopts['img_pickle_path']%(b+1), batch)
    log.info('==== SAVING TESTINFORMATION =====')
    save_testinfo(GLOBopts['tmp_dir']+'/testinfo.txt', batches, VOCopts.classes)

def train_cal(train_images, descriptor_function, estimator, CALopts, TESTopts):
    for cls, images in train_images.items():
        log.info('==== GET %s DESCRIPTORS ====', cls)
        descriptors = get_image_descriptors(images, descriptor_function, \
            TESTopts['descriptor_path'])
        descriptors = [d for p,d in descriptors.values()]
        log.info('==== ADD %s DESCRIPTORS TO ESTIMATOR', cls)
        estimator.add_class(cls, descriptors)

def make_cal_tests(test_images, descriptor_function, CALopts, TESTopts):
    log.info('==== FLATTEN TEST_IMAGES TO LIST ====')
    test_images = [image for clslist in test_images.values() for image in clslist]
    log.info('==== GENERATING AND SAVING TEST DESCRIPTORS =====')
    save_image_descriptors(test_images, descriptor_function, \
        TESTopts['descriptor_path'])
    batches = get_image_batches(CALopts, test_images, TESTopts['batch_size'])
    log.info('==== SAVING IMAGE OBJECTS PER BATCH =====')
    for b,batch in enumerate(batches):
        with open(TESTopts['img_pickle_path']%(b+1), 'wb') as pklfile:
            cPickle.dump(batch, pklfile)
    log.info('==== SAVING TESTINFORMATION =====')
    save_testinfo(TESTopts['infofile'], batches, CALopts.classes)
