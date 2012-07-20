import cPickle
import logging
from utils import *
from nbnn.voc import *
import os.path

log = logging.getLogger("__name__")

def train_voc(descriptor_function, estimator, object_type, VOCopts, \
        descriptor_path=None, exemplar_path=None):
    for i,cls in enumerate(VOCopts.classes):
        if not cls in estimator.classes:
            log.info('==== GET CLASS %d: %s IMAGES ====', i, cls)
            img_set = read_image_set(VOCopts,cls+'_train')
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
                    with open(exemplar_path%cls,'wb') as ef:
                        cPickle.dump(exemplars, ef)
                else:
                    fg_descriptors = get_bbox_descriptors(objects, descriptors)
                    estimator.add_class(cls+'_fg', fg_descriptors)


def make_voc_tests(descriptor_function, VOCopts, TESTopts):
    log.info('==== GENERATING TEST IMAGES =====')
    test_images = read_image_set(VOCopts,TESTopts['test_set'])
    log.info('==== GENERATING AND SAVING TEST DESCRIPTORS =====')
    save_image_descriptors(test_images, descriptor_function, \
        TESTopts['descriptor_path'])
    batches = get_image_batches(VOCopts, test_images, TESTopts['batch_size'])
    log.info('==== SAVING IMAGE OBJECTS PER BATCH =====')
    for b,batch in enumerate(batches):
        with open(TESTopts['img_pickle_path']%(b+1), 'wb') as pklfile:
            cPickle.dump(batch, pklfile)
    log.info('==== SAVING TESTINFORMATION =====')
    save_testinfo(TESTopts['infofile'], batches, VOCopts.classes)

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
