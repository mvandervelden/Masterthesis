import cPickle
import logging
import os.path

from utils import *
from nbnn.voc import *
from io import *

log = logging.getLogger("__name__")

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
    save_testinfo(GLOBopts['tmp_path']+'/testinfo.txt', batches, VOCopts.classes)

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
