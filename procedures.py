import cPickle
import logging
from utils import *
from nbnn.voc import *

log = logging.getLogger("__name__")

def train_voc(descriptor_function, estimator, object_type, VOCopts, \
        descriptor_path=None):
    for i,cls in enumerate(VOCopts.classes):
        log.info('==== GET CLASS %d: %s IMAGES ====', i, cls)
        img_set = read_image_set(VOCopts,cls+'_train')
        log.info('==== GET %s DESCRIPTORS ====', cls)
        descriptors = get_image_descriptors(img_set, descriptor_function, \
            descriptor_path)
        if object_type == 'bbox':
            # Get descriptor's objects (bboxes):
            log.info('==== GET %s OBJECTS ====', cls)
            objects = get_objects_by_class(img_set, cls)
            descriptors = get_bbox_descriptors(objects, descriptors)
        log.info('==== ADD %s DESCRIPTORS TO ESTIMATOR', cls)
        estimator.add_class(cls, descriptors)
    log.info('==== REMOVING TRAIN DESCRIPTORS FROM DISK ====')    

def make_voc_tests(descriptor_function, VOCopts, TESTopts):
    log.info('==== GENERATING TEST IMAGES =====')
    test_images = read_image_set(VOCopts,'test')
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
